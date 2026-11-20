"""Stage B: Stage A 결과로 Random Forest를 학습해 볼륨의 모든 voxel을 best / not-best로 판정한다.

  train_stage_b_forest          — best(Stage A best 샘플) / not-best(볼륨 voxel)로 RF 학습
  classify_voxels_view_features — 모든 voxel의 P(best)로 local mask와 cluster 라벨 볼륨 생성

view-dependent feature(depth/visibility/opacity)는 view_feature_baker.ViewFeatureBaker에서 받는다.
"""

import time

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from src.core.clustering.feature_clustering import ROIFeatureExtractor
from src.core.clustering.rf_classifier import StageBForest, BEST, NOT_BEST, rf_decision

try:
    import torch
    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    _TORCH_CUDA_AVAILABLE = False


def build_full_feature_matrix(feature_layout, *, inr_features=None, depths=None,
                              visibility=None, intensities=None, opacity=None, inr_only=False):
    """feature_layout 순서대로 원본(비정규화) 열을 이어 붙인다.

    inr_only=True면 layout과 무관하게 INR embedding + coords만 사용한다.
    Returns: (N, D) float32
    """
    if inr_only:
        if inr_features is None:
            raise ValueError("INR만 모드인데 INR feature가 없음")
        embedding, coords = ROIFeatureExtractor.split_inr_embedding_and_coords(inr_features)
        return np.concatenate([np.asarray(embedding, dtype=np.float32),
                               np.asarray(coords, dtype=np.float32)], axis=1)

    embedding = coords = None
    if inr_features is not None:
        embedding, coords = ROIFeatureExtractor.split_inr_embedding_and_coords(inr_features)
    by_name = {"intensity": intensities, "opacity": opacity, "depth": depths,
               "visibility": visibility, "inr_embedding": embedding, "coords": coords}

    cols = []
    for name, start, end in feature_layout:
        cols.append(np.asarray(by_name[name], dtype=np.float32).reshape(-1, end - start))
    return np.concatenate(cols, axis=1)


def choose_chunk_z(shape, num_channels, k, budget_mb):
    """청크 하나가 budget_mb 안에 들어오는 Z 슬랩 두께."""
    X, Y, Z = shape[:3]
    bytes_per_z = max(X * Y * (num_channels + k) * 4, 1)
    chunk_z = int(budget_mb * 1024 * 1024 // bytes_per_z)
    return max(1, min(Z, chunk_z))


def stage_a_reference(cluster_result, roi_info):
    """Stage B가 재사용할 Stage A 정보(layout, 라벨, centroid, 정규화 파라미터, 샘플 feature 행렬)."""
    layout = cluster_result["feature_layout"]
    return {
        "layout": layout,
        "best": cluster_result["best_cluster_index"],
        "labels": np.asarray(cluster_result["labels"]),
        "centroids": cluster_result["centroids"].astype(np.float32),
        "feat_mean": cluster_result["feat_mean"].astype(np.float32),
        "feat_std": cluster_result["feat_std"].astype(np.float32),
        "stage_a_full": build_full_feature_matrix(
            layout,
            inr_features=roi_info.get("filtered_inr_features"),
            depths=roi_info.get("filtered_depths"),
            visibility=roi_info.get("filtered_visibility"),
            intensities=roi_info.get("filtered_intensities"),
            opacity=roi_info.get("filtered_opacity"),
        ),
    }


def normalize_with(ref, matrix):
    """Stage A와 같은 z-score 정규화를 적용한다."""
    return (matrix.astype(np.float32, copy=False) - ref["feat_mean"]) / (ref["feat_std"] + np.float32(1e-8))


def distances_to_centroids(flat, centroids):
    """(M, D)와 centroid (k, D) 사이 유클리드 거리. Returns: (M, k) float32"""
    if _TORCH_CUDA_AVAILABLE:
        try:
            device = torch.device('cuda')
            flat_t = torch.as_tensor(flat, dtype=torch.float32, device=device)
            centroids_t = torch.as_tensor(centroids, dtype=torch.float32, device=device)
            return torch.cdist(flat_t, centroids_t).cpu().numpy()
        except Exception as e:
            print(f"⚠️ GPU 거리 계산 실패({e}) — CPU로 대체")

    dists = np.empty((flat.shape[0], centroids.shape[0]), dtype=np.float32)
    for ci in range(centroids.shape[0]):
        dists[:, ci] = np.linalg.norm(flat - centroids[ci], axis=1)
    return dists


def voxelize_world_points(points, volume_shape, voxel_spacing, model_to_world=None):
    """world 좌표 점을 가장 가까운 voxel로 반올림해 표시한다.

    model_to_world: (4,4) 볼륨 actor 변환 행렬. None이면 항등.
    Returns: (X, Y, Z) bool
    """
    out = np.zeros(tuple(volume_shape[:3]), dtype=bool)
    if len(points) == 0:
        return out

    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if model_to_world is not None:
        inv = np.linalg.inv(np.asarray(model_to_world, dtype=np.float64))
        pts = (np.hstack([pts, np.ones((len(pts), 1))]) @ inv.T)[:, :3]

    idx = np.round(pts / np.asarray(voxel_spacing, dtype=np.float64)).astype(int)
    inside = np.all((idx >= 0) & (idx < np.asarray(volume_shape[:3])), axis=1)
    idx = idx[inside]
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def voxel_indices_to_world(idx, voxel_spacing, model_to_world=None):
    """voxel index (M,3) → world 좌표 (M,3). voxelize_world_points의 역변환."""
    pts = np.asarray(idx, dtype=np.float64) * np.asarray(voxel_spacing, dtype=np.float64)
    if model_to_world is None:
        return pts
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (homo @ np.asarray(model_to_world, dtype=np.float64).T)[:, :3]


def stage_a_best_cluster_voxels(cluster_result, roi_info, volume_shape,
                                voxel_spacing, model_to_world=None):
    """Stage A best cluster 샘플이 떨어진 voxel(seed). Returns: (X, Y, Z) bool 또는 None"""
    best_mask = cluster_result.get("best_cluster_mask")
    points = roi_info.get("filtered_points")
    if best_mask is None or points is None:
        return None

    pts = np.asarray(points, dtype=np.float64)[np.asarray(best_mask, dtype=bool)]
    if len(pts) == 0:
        return None
    return voxelize_world_points(pts, volume_shape, voxel_spacing, model_to_world)


def _ray_sample_spacing(points, depths, best):
    """ray 샘플 간격 = max(ray 위 간격, 이웃 ray 간격). 이웃 ray 간격은 ray마다 첫 best 샘플끼리의 최근접 거리 중앙값.

    points/depths는 ray 순서로 편 샘플이다(ray 안에서 depth가 늘고, 다음 ray로 넘어가면 줄어든다).
    """
    d_depth = np.diff(depths)
    gaps = np.linalg.norm(np.diff(points, axis=0), axis=1)[d_depth > 0]
    along = float(np.median(gaps)) if len(gaps) else 0.0
    ray_id = np.r_[0, np.cumsum(d_depth < 0)]
    b_idx = np.flatnonzero(best)
    _, first = np.unique(ray_id[b_idx], return_index=True)
    heads = points[b_idx[first]]
    if len(heads) < 2:
        return along
    return max(along, float(np.median(cKDTree(heads).query(heads, k=2)[0][:, 1])))


def split_best_by_space(cluster_result, roi_info, volume_shape, voxel_spacing, model_to_world=None):
    """Stage A 후처리: best 샘플을 격자에 찍어 26-연결 성분으로 나누고, 샘플이 가장 많은 성분만 best로 남긴다.

    K-means cluster는 feature 공간의 묶음이라 feature가 비슷한 다른 구조(같은 intensity의 두피, 떨어진 부품)가
    소수로 섞여도 떼어내지 못한다. 떨어진 나머지 성분은 best 학습 샘플에서만 빼고 RF not-best로는 쓰지 않는다
    (SAM mask가 같은 구조를 잘라 떨어진 성분이 생겼을 때 같은 구조를 not-best로 학습하지 않도록).

    격자 간격 = 축마다 max(voxel 간격, ray 샘플 간격). ray 샘플이 voxel보다 성기면 voxel 격자로는 같은 ray의 연속 샘플도
    이웃이 아니어서 best가 낱알로 흩어진다.

    cluster_result를 제자리 수정한다: best_cluster_mask = 주 성분 샘플만, detached_best_mask 추가(기록용).
    볼륨 밖 샘플은 건드리지 않는다(best 유지).
    Returns: info dict {components, main, detached, spacing(ray 샘플 간격)}
    """
    best = np.asarray(cluster_result["best_cluster_mask"], dtype=bool)
    pts = np.asarray(roi_info["filtered_points"], dtype=np.float64)
    if model_to_world is not None:
        inv = np.linalg.inv(np.asarray(model_to_world, dtype=np.float64))
        pts = (np.hstack([pts, np.ones((len(pts), 1))]) @ inv.T)[:, :3]
    voxel_spacing = np.asarray(voxel_spacing, dtype=np.float64)
    idx = np.round(pts / voxel_spacing).astype(int)
    inside = np.all((idx >= 0) & (idx < np.asarray(volume_shape[:3])), axis=1)

    sel = np.flatnonzero(best & inside)
    spacing = _ray_sample_spacing(pts, np.asarray(roi_info["filtered_depths"], dtype=np.float64), best)
    grid = np.maximum(voxel_spacing, spacing)
    cell = np.round(pts[sel] / grid).astype(int)   # 볼륨 안 샘플이라 0 이상
    occ = np.zeros(tuple(np.round(np.asarray(volume_shape[:3]) * voxel_spacing / grid).astype(int) + 1), dtype=bool)
    occ[cell[:, 0], cell[:, 1], cell[:, 2]] = True
    labels, n = ndimage.label(occ, structure=np.ones((3, 3, 3), dtype=bool))
    comp = labels[cell[:, 0], cell[:, 1], cell[:, 2]]
    counts = np.bincount(comp, minlength=n + 1)
    counts[0] = 0
    detached = np.zeros_like(best)
    if n > 1:
        main = int(np.argmax(counts))
        detached[sel[comp != main]] = True
    cluster_result["best_cluster_mask"] = best & ~detached
    cluster_result["detached_best_mask"] = detached
    return {"components": int(n), "main": int(best.sum() - detached.sum()), "detached": int(detached.sum()),
            "spacing": spacing}


def nearest_member_cluster(dists, ref, exclude=None):
    """가장 가까운 cluster 라벨. Stage A 멤버가 없는 cluster와 exclude는 후보에서 뺀다.

    Returns: (M,) int16 (후보가 없으면 전부 -1)
    """
    valid = np.unique(ref["labels"])
    if exclude is not None:
        valid = valid[valid != exclude]
    if len(valid) == 0:
        return np.full(len(dists), -1, dtype=np.int16)
    return valid[np.argmin(dists[:, valid], axis=1)].astype(np.int16)


def compose_final_labels(other_labels, final_mask, best_idx):
    """라벨 볼륨에서 final_mask 영역을 best_idx로 덮어쓴다(제자리 수정). Returns: other_labels"""
    other_labels[np.asarray(final_mask, dtype=bool)] = best_idx
    return other_labels


def train_stage_b_forest(cluster_result, roi_info, volume_data, inr_feature_volume, view,
                         model_to_world=None, n_estimators=100, margin=2, max_best=50000,
                         max_not_best=100000, random_state=42, inr_only=False):
    """best / not-best 2클래스 Random Forest를 학습한다.

    best     — Stage A best cluster 샘플(cluster_result["best_cluster_mask"])에서 최대 max_best개.
    not-best — 볼륨 전체 voxel 중 best seed를 margin voxel 팽창한 영역 밖이고
               판정 가능한(view.view_features의 ok) voxel에서 최대 max_not_best개.
    inr_only — True면 INR embedding + coords(비정규화)만으로 학습한다.

    Returns: 학습된 StageBForest
    """
    ref = stage_a_reference(cluster_result, roi_info)
    rs = np.random.RandomState(random_state)
    shape = tuple(volume_data.shape[:3])
    spacing = np.asarray(view.spacing, dtype=np.float64)

    # best
    best_pool = np.flatnonzero(np.asarray(cluster_result["best_cluster_mask"], dtype=bool))
    n_best = min(len(best_pool), int(max_best))
    best_sel = rs.choice(best_pool, n_best, replace=False)
    if inr_only:
        X_best = build_full_feature_matrix(
            None, inr_features=np.asarray(roi_info.get("filtered_inr_features"))[best_sel], inr_only=True)
    else:
        X_best = normalize_with(ref, ref["stage_a_full"][best_sel])

    # not-best 제외 영역: seed + margin (팽창은 seed bounding box 안에서만)
    excluded = np.zeros(shape, dtype=bool)
    seed = stage_a_best_cluster_voxels(cluster_result, roi_info, shape, spacing, model_to_world)
    if seed is not None and seed.any():
        sidx = np.argwhere(seed)
        lo = np.maximum(sidx.min(0) - int(margin), 0)
        hi = np.minimum(sidx.max(0) + int(margin) + 1, shape)
        box = tuple(slice(a, b) for a, b in zip(lo, hi))
        zone = seed[box]
        if margin > 0:
            zone = ndimage.binary_dilation(zone, structure=np.ones((3, 3, 3), dtype=bool),
                                           iterations=int(margin))
        excluded[box] = zone

    # not-best: 무작위 후보 중 제외 영역 밖·판정 가능한 voxel
    total = int(np.prod(shape))
    cand = rs.choice(total, min(total, int(max_not_best) * 4), replace=False)
    cand = cand[~excluded.ravel()[cand]]
    vidx = np.column_stack(np.unravel_index(cand, shape))
    values = volume_data[vidx[:, 0], vidx[:, 1], vidx[:, 2]]
    depth, visibility, opacity, ok = view.view_features(vidx * spacing, values)
    keep = np.flatnonzero(ok)[:int(max_not_best)]
    vidx = vidx[keep]
    inr = (None if inr_feature_volume is None else
           inr_feature_volume[vidx[:, 0], vidx[:, 1], vidx[:, 2]])
    if inr_only:
        X_not = build_full_feature_matrix(None, inr_features=inr, inr_only=True)
    else:
        X_not = normalize_with(ref, build_full_feature_matrix(
            ref["layout"], inr_features=inr, depths=depth[keep], visibility=visibility[keep],
            intensities=values[keep], opacity=opacity[keep]))
    n_not = len(X_not)
    print(f"🌲 [Stage B RF] 학습셋: best {n_best:,}/{len(best_pool):,} / not-best {n_not:,} "
          f"(마진 {margin} voxel, 제외 영역 {int(excluded.sum()):,} voxel)")

    X = np.concatenate([X_best, X_not]).astype(np.float32)
    y = np.concatenate([np.full(n_best, BEST), np.full(n_not, NOT_BEST)]).astype(np.int32)
    forest = StageBForest(n_estimators=n_estimators, random_state=random_state)
    forest.inr_only = bool(inr_only)
    forest.fit(X, y, extra_info={"margin": int(margin),
                                 "margin_excluded": int(excluded.sum()),
                                 "inr_only": bool(inr_only), "n_features": X.shape[1]})
    # "Show RF samples" 오버레이용: best = ROI 샘플 인덱스, not-best = voxel 인덱스 (M,3)
    forest.best_sample_idx, forest.not_best_voxels = best_sel, vidx
    return forest


def _rf_used_groups(forest, layout):
    """RF 입력 feature 그룹 이름 목록."""
    if forest.inr_only:
        return ["inr_embedding", "coords"]
    return [name for name, _, _ in layout]


def _rf_input(forest, normalized_full, inr_features):
    """forest 입력 행렬: INR만 모드면 INR embedding + coords, 아니면 정규화된 layout 행렬."""
    if forest.inr_only:
        return build_full_feature_matrix(None, inr_features=inr_features, inr_only=True)
    return normalized_full


def _p_hist(p):
    """확률 배열 → 0.01 간격 101칸 히스토그램."""
    return np.bincount(np.round(np.asarray(p) * 100).astype(np.int64), minlength=101)


def _rf_debug(forest, p_best_hist, min_best_prob):
    """RF 학습/판정 진단값(format_stage_b 출력용). p_best_hist: best 판정 voxel의 P(best) 분포."""
    total = int(p_best_hist.sum())

    def quantile(frac):
        if total == 0:
            return float("nan")
        return float(np.searchsorted(np.cumsum(p_best_hist), frac * total) / 100.0)

    info = forest.train_info
    return {"n_estimators": forest.n_estimators,
            "train_accuracy": info.get("train_accuracy"),
            "n_best": info.get("n_best"), "n_not_best": info.get("n_not_best"),
            "margin": info.get("margin"), "margin_excluded": info.get("margin_excluded"),
            "inr_only": info.get("inr_only", False), "n_features": info.get("n_features"),
            "min_best_prob": float(min_best_prob),
            "kept_p_best_median": quantile(0.5),
            "kept_p_best_p10": quantile(0.1)}


def classify_voxels_view_features(cluster_result, roi_info, volume_data, inr_feature_volume, view,
                                  forest, min_best_prob=0.0, chunk_budget_mb=256.0):
    """볼륨의 모든 voxel을 forest의 P(best)로 판정한다(Z 슬랩 단위로 처리).

    voxel feature: INR/coords는 INR 볼륨 값, intensity는 볼륨 값,
    depth/visibility/opacity는 view.view_features(p_model, values)가 계산한다.

    Returns: (best_mask, labels, debug_info)
        best_mask  (X,Y,Z) bool  — P(best) > 0.5 이고 P(best) >= min_best_prob
        labels     (X,Y,Z) int16 — best를 제외한 가장 가까운 cluster(판정 불가 voxel은 -1)
        debug_info dict          — used_groups, reduced_dim, num_samples, num_kept, rf,
                                   unclassifiable_count, elapsed_sec, p_best((X,Y,Z) float16)
    """
    t0 = time.time()
    ref = stage_a_reference(cluster_result, roi_info)
    layout, best_idx = ref["layout"], ref["best"]
    if inr_feature_volume is None and (forest.inr_only or
                                       any(n in ("inr_embedding", "coords") for n, _, _ in layout)):
        raise ValueError("INR feature volume이 필요하지만 로드되지 않음")

    X, Y, Z = volume_data.shape[:3]
    k, D = ref["centroids"].shape
    spacing = np.asarray(view.spacing, dtype=np.float64)
    best_mask = np.zeros((X, Y, Z), dtype=bool)
    labels = np.full((X, Y, Z), -1, dtype=np.int16)
    p_vol = np.full((X, Y, Z), -1, dtype=np.float16)   # 판정 불가 = -1
    p_best_hist = np.zeros(101, dtype=np.int64)
    unclassifiable = 0

    chunk_z = choose_chunk_z((X, Y, Z), D, k, chunk_budget_mb)
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    for z0 in range(0, Z, chunk_z):
        z1 = min(Z, z0 + chunk_z)
        kk = np.arange(z0, z1)
        idx = np.stack(np.broadcast_arrays(ii[..., None], jj[..., None], kk[None, None, :]),
                       axis=-1).reshape(-1, 3)
        values = volume_data[:, :, z0:z1].reshape(-1)
        depth, visibility, opacity, ok = view.view_features(idx * spacing, values)
        inr_flat = (None if inr_feature_volume is None else
                    inr_feature_volume[:, :, z0:z1, :].reshape(-1, inr_feature_volume.shape[-1]))
        feats = build_full_feature_matrix(layout, inr_features=inr_flat, depths=depth,
                                          visibility=visibility, intensities=values,
                                          opacity=opacity)

        keep = np.zeros(len(values), dtype=bool)
        lab = np.full(len(values), -1, dtype=np.int16)
        p_flat = np.full(len(values), -1, dtype=np.float16)
        if ok.any():
            normalized = normalize_with(ref, feats[ok])
            p_best = forest.predict_best_proba(_rf_input(
                forest, normalized, None if inr_flat is None else inr_flat[ok]))
            keep_ok = rf_decision(p_best, min_best_prob)
            keep[ok] = keep_ok
            p_flat[ok] = p_best
            dists = distances_to_centroids(normalized, ref["centroids"])
            lab[ok] = nearest_member_cluster(dists, ref, exclude=best_idx)
            p_best_hist += _p_hist(p_best[keep_ok])
        unclassifiable += int((~ok).sum())
        best_mask[:, :, z0:z1] = keep.reshape(X, Y, z1 - z0)
        labels[:, :, z0:z1] = lab.reshape(X, Y, z1 - z0)
        p_vol[:, :, z0:z1] = p_flat.reshape(X, Y, z1 - z0)

    elapsed = time.time() - t0
    print(f"⏱️ [Stage B] RF voxel 판정 ({X * Y * Z:,} voxel, d={D}): {elapsed:.2f}초")
    return best_mask, labels, {
        "used_groups": _rf_used_groups(forest, layout),
        "reduced_dim": forest.train_info.get("n_features"),
        "num_samples": X * Y * Z - unclassifiable,
        "num_kept": int(best_mask.sum()),
        "rf": _rf_debug(forest, p_best_hist, min_best_prob),
        "unclassifiable_count": unclassifiable,
        "elapsed_sec": elapsed,
        "p_best": p_vol,   # (X,Y,Z) float16 voxel별 P(best), -1 = 판정 불가
    }


def sample_label_points(labels, max_points=300000):
    """라벨이 있는(>= 0) voxel을 균등 간격으로 최대 max_points개 고른다(Z 슬랩 단위 처리).

    Returns: (idx (M,3) int64, lab (M,) int16)
    """
    total = int((labels >= 0).sum())
    if total == 0:
        return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int16)
    step = max(1, int(np.ceil(total / max_points)))
    X, Y, Z = labels.shape
    cz = max(1, 4_000_000 // max(X * Y, 1))
    seen = 0
    idx_parts, lab_parts = [], []
    for z0 in range(0, Z, cz):
        slab = labels[:, :, z0:z0 + cz]
        flat = np.flatnonzero(slab.ravel() >= 0)
        pick = flat[(-seen) % step::step]
        seen += len(flat)
        if len(pick) == 0:
            continue
        ix, iy, iz = np.unravel_index(pick, slab.shape)
        idx_parts.append(np.column_stack((ix, iy, iz + z0)))
        lab_parts.append(slab.ravel()[pick])
    return np.vstack(idx_parts), np.concatenate(lab_parts)


if __name__ == "__main__":
    # 자체 검사(python -m src.core.clustering.volume_classification): voxel(1mm)보다 성긴 ray 샘플(2mm)도 한 덩어리로
    # 잇고, 30mm 떨어진 덩어리는 떼어낸다.
    rays = [(x, y) for x in range(0, 10, 2) for y in range(0, 10, 2)] + [(40, 40), (40, 42), (42, 40), (42, 42)]
    z = np.arange(10) * 2.0
    pts = np.concatenate([np.c_[np.full(10, x), np.full(10, y), z] for x, y in rays]).astype(float)
    cr = {"best_cluster_mask": np.ones(len(pts), dtype=bool)}
    info = split_best_by_space(cr, {"filtered_points": pts, "filtered_depths": np.tile(z, len(rays))},
                               (64, 64, 32), (1.0, 1.0, 1.0))
    assert (info["components"], info["main"], info["detached"]) == (2, 250, 40), info
    print("split_best_by_space 자체 검사 통과:", info)
