"""시선에 수직인 깊이 구간(slab)으로 SAM mask ray를 자르고, 구간마다 K-means로 구조를 가르는 Stage A 대안 실험.

1) canonical(canonical_profile): 모든 ray 샘플을 시선축(ray 평균 방향)에 투영한 깊이로 맞춰, 깊이 bin마다 opacity 평균을 낸다.
   ΔO가 아니라 opacity를 쓴다: ΔO는 O가 포화되면 0이라 가려진 뒤쪽 구조가 사라지지만, opacity는 가려짐과 무관해서
   안 보이는 뒤쪽도 구간으로 나뉜다.
2) 구간(cut_slabs): canonical을 살짝 smoothing한 뒤, 값이 empty_thr 이하인 깊이는 빈 곳(구간 아님),
   나머지 연속 구간은 골이 양옆 봉우리 중 낮은 쪽보다 valley_depth 이상 낮으면 골에서 자른다.
   두 기준 모두 절대값이다. canonical은 곡선 하나라 모든 ray에 같은 기준이 적용된다.
3) 구간 안 clustering(cluster_slabs): 구간에 든 샘플만 INR embedding + coords로 z-score(구간 안에서) 후 K-means(k, seed, 재시작).
4) best: cluster의 ΔO 합(화면에 보이는 양)이 가장 큰 것(평균을 쓰면 1~2샘플 조각이 이긴다).
   ΔO는 ray_sampler의 sampled_visibility(렌더러식 step 보정 opacity × 앞쪽 투과율)를 그대로 쓴다.

지금은 앱 UI에 연결돼 있지 않다(결과 창은 canonical_view.CanonicalView).
"""
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from src.core.clustering.feature_clustering import _kmeans_cpu, _kmeans_gpu, _TORCH_CUDA_AVAILABLE

EMPTY_THR = 0.005        # canonical opacity 평균이 이 값 이하인 깊이 = 빈 곳 (절대값)
VALLEY_DEPTH = 0.02      # 골이 낮은 쪽 봉우리보다 이만큼 이상 낮으면 자른다 (절대값, 0이면 빈 곳에서만 자름)
CANON_SMOOTH = 1.0       # canonical smoothing σ (bin)


def canonical_profile(points, opacity, valid):
    """ray 샘플을 시선축 깊이 bin으로 모아 canonical opacity 평균을 만든다.

    Args: points (R, N, 3) world 좌표, opacity (R, N), valid (R, N) bool (박스 안·clipping 통과)
    Returns: dict — bin (R, N): 샘플의 깊이 bin(무효 샘플 -1), op_mean (B,), smooth (B,): 구간 자르기에 쓰는 smoothing 곡선,
             inside_frac (B,): 그 깊이에서 opacity>0인 ray 비율, step: bin 폭(샘플 간격 중앙값), axis (3,): 시선축 단위벡터
    """
    R, N = opacity.shape
    has = valid.any(axis=1)
    first = np.argmax(valid, axis=1)
    last = N - 1 - np.argmax(valid[:, ::-1], axis=1)
    rr = np.flatnonzero(has & (last > first))
    span = points[rr, last[rr]] - points[rr, first[rr]]
    length = np.linalg.norm(span, axis=1)
    axis = (span / length[:, None]).mean(axis=0)
    axis /= np.linalg.norm(axis)
    step = float(np.median(length / (last[rr] - first[rr])))

    z = points @ axis
    z0 = z[valid].min()
    bins = np.where(valid, ((z - z0) / step).astype(np.int64), -1)
    B = int(bins.max()) + 1
    flat_b = bins[valid]
    op_mean = np.bincount(flat_b, weights=opacity[valid], minlength=B) / np.maximum(np.bincount(flat_b, minlength=B), 1)
    pos = valid & (opacity > 0)
    ray_bin = np.unique(np.column_stack([np.nonzero(pos)[0], bins[pos]]), axis=0)   # ray마다 bin당 한 번
    inside_frac = np.bincount(ray_bin[:, 1], minlength=B) / max(int(has.sum()), 1)
    return {"bin": bins, "op_mean": op_mean, "smooth": gaussian_filter1d(op_mean, CANON_SMOOTH, mode="nearest"),
            "inside_frac": inside_frac, "step": step, "axis": axis}


def cut_slabs(curve, empty_thr=EMPTY_THR, valley_depth=VALLEY_DEPTH):
    """canonical 곡선(smoothing) → 구간 목록 [(시작 bin, 끝 bin(미포함)), ...] (빈 곳은 구간이 아님)."""
    occ = curve > empty_thr
    edges = np.flatnonzero(np.diff(occ.astype(np.int8))) + 1
    bounds = np.concatenate([[0], edges, [len(curve)]])
    slabs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if not occ[a]:
            continue
        seg = curve[a:b]
        peaks, _ = find_peaks(np.concatenate([[-np.inf], seg, [-np.inf]]))
        peaks = peaks - 1
        cuts, cur, prev = [], seg[peaks[0]], peaks[0]
        for pk in peaks[1:]:
            v = prev + int(np.argmin(seg[prev:pk + 1]))
            if valley_depth > 0 and min(cur, seg[pk]) - seg[v] >= valley_depth:
                cuts.append(v)
                cur = seg[pk]
            else:
                cur = max(cur, seg[pk])
            prev = pk
        sub = [0] + cuts + [b - a]
        slabs.extend((int(a + u), int(a + w)) for u, w in zip(sub[:-1], sub[1:]))
    return slabs


def cluster_slabs(bins, slabs, inr_features, dO, k, seed=42, n_init=10):
    """구간마다 INR embedding + coords로 K-means (구간 안에서 z-score).

    Args: bins (R, N), slabs [(b0, b1)], inr_features (R, N, C), dO (R, N)
    Returns: slab_id (R, N) (-1 = 구간 밖), cluster_rank (R, N) (-1 = 구간 밖, 0 = best = ΔO 합 1등),
             clusters: 순위별 {slab, size, dO_sum, dO_mean}
    """
    R, N = bins.shape
    slab_id = np.full((R, N), -1, dtype=np.int64)
    for si, (b0, b1) in enumerate(slabs):
        slab_id[(bins >= b0) & (bins < b1)] = si
    cl_global = np.full((R, N), -1, dtype=np.int64)
    cl_slab = []
    for si in range(len(slabs)):
        m = slab_id == si
        n = int(m.sum())
        if n == 0:
            continue
        X = inr_features[m].astype(np.float64)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)
        kk = max(1, min(k, n))
        if kk == 1:
            lab = np.zeros(n, dtype=np.int64)
        elif _TORCH_CUDA_AVAILABLE:
            try:
                lab, _ = _kmeans_gpu(X, kk, seed=seed, n_init=n_init)
            except Exception as e:
                print(f"⚠️ GPU K-means 실패({e}) — CPU로 대체")
                lab, _ = _kmeans_cpu(X, kk, seed=seed, n_init=n_init)
        else:
            lab, _ = _kmeans_cpu(X, kk, seed=seed, n_init=n_init)
        cl_global[m] = len(cl_slab) + np.asarray(lab, dtype=np.int64)
        cl_slab.extend([si] * kk)
    G = len(cl_slab)
    if G == 0:
        return slab_id, cl_global, []
    flat = cl_global.ravel()
    ok = flat >= 0
    size = np.bincount(flat[ok], minlength=G)
    dsum = np.bincount(flat[ok], weights=dO.ravel()[ok], minlength=G)
    order = np.argsort(-dsum, kind="stable")
    rank_of = np.empty(G, dtype=np.int64)
    rank_of[order] = np.arange(G)
    cluster_rank = np.where(cl_global >= 0, rank_of[np.maximum(cl_global, 0)], -1)
    clusters = [{"slab": cl_slab[g], "size": int(size[g]), "dO_sum": float(dsum[g]),
                 "dO_mean": float(dsum[g] / max(size[g], 1))} for g in order]
    return slab_id, cluster_rank, clusters


def run(points, opacity, valid, dO, inr_features, k, seed=42, n_init=10,
        empty_thr=EMPTY_THR, valley_depth=VALLEY_DEPTH):
    """canonical → 구간 → 구간별 K-means → best.

    opacity: (R, N) TF opacity(구간 자르기), dO: (R, N) step 보정 ΔO(best 선택, ray_sampler sampled_visibility),
    inr_features: (R, N, C) 또는 None(구간까지만). Returns: dict
    """
    t0 = time.perf_counter()
    can = canonical_profile(points, opacity, valid)
    slabs = cut_slabs(can["smooth"], empty_thr, valley_depth)
    if inr_features is not None:
        slab_id, cluster_rank, clusters = cluster_slabs(can["bin"], slabs, inr_features, dO, k, seed, n_init)
    else:
        slab_id, cluster_rank, clusters = cluster_slabs(can["bin"], slabs, np.zeros(opacity.shape + (1,)), dO, 1)
    print(f"[깊이 구간] canonical {len(can['op_mean'])} bin(폭 {can['step']:.2f}), 빈 곳 ≤ {empty_thr:g}, 골 깊이 ≥ {valley_depth:g}"
          f" → 구간 {len(slabs)}개 {slabs}, cluster {len(clusters)}개  ({time.perf_counter() - t0:.1f}초)")
    return {"canonical": can, "slabs": slabs, "slab_id": slab_id, "cluster_rank": cluster_rank,
            "clusters": clusters, "dO": dO, "params": (empty_thr, valley_depth)}


def format_result(out, opacity, n=10):
    """구간·cluster 표 문자열 (cluster 순위 = ΔO 합 순, #1 = best)."""
    lines = []
    sid, rank = out["slab_id"], out["cluster_rank"]
    for si, (a, b) in enumerate(out["slabs"]):
        m = sid == si
        lines.append(f"  구간 {si}: bin {a}–{b - 1}  샘플 {int(m.sum()):,} (opacity>0 {int((m & (opacity > 0)).sum()):,}), "
                     f"ΔO 합 {out['dO'][m].sum():.1f}")
    lines.append("  순위   구간       샘플   opacity>0     ΔO 합    ΔO 평균")
    for rk, c in enumerate(out["clusters"][:n]):
        m = rank == rk
        lines.append(f"  #{rk + 1:<4} {c['slab']:>4} {c['size']:>10,} {int((m & (opacity > 0)).sum()):>11,} "
                     f"{c['dO_sum']:>9.1f} {c['dO_mean']:>10.4f}")
    return "\n".join(lines)
