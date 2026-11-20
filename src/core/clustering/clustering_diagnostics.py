"""Stage A / Stage B 결과를 콘솔 로그용 문자열로 요약한다."""

import numpy as np
from scipy import ndimage


def _best_vs_nearest_contribution(centroids, feature_layout, best_idx):
    """best cluster와 가장 가까운 cluster 사이 제곱거리를 feature 그룹별로 나눈다.

    Returns: (nearest_idx, {group: 제곱거리}, 전체 제곱거리, 거리). cluster가 1개면 전부 None.
    """
    if centroids.shape[0] < 2:
        return None, None, None, None

    dists = np.linalg.norm(centroids - centroids[best_idx], axis=1)
    dists[best_idx] = np.inf
    nearest_idx = int(np.argmin(dists))

    d2 = (centroids[best_idx] - centroids[nearest_idx]) ** 2
    totals = {name: float(d2[s:e].sum()) for name, s, e in feature_layout}
    return nearest_idx, totals, float(d2.sum()), float(dists[nearest_idx])


def format_stage_a(cluster_result, max_clusters=12):
    """Stage A 요약: best vs 최근접 cluster의 그룹별 분리 기여도 + cluster별 통계 표."""
    layout = cluster_result.get("feature_layout")
    centroids = cluster_result.get("centroids")
    if not layout or centroids is None:
        return "[Stage A] feature_layout/centroids 없음"

    k = centroids.shape[0]
    stats = cluster_result.get("cluster_stats", [])
    scores = cluster_result.get("scores")
    best = cluster_result.get("best_cluster_index")

    lines = [f"┌─ Stage A: K-means (k={k}, d={centroids.shape[1]})"]

    nearest_idx, totals, grand, nearest_dist = _best_vs_nearest_contribution(centroids, layout, best)
    if nearest_idx is not None:
        lines.append(f"│  ROI(cluster {best}) vs 최근접 cluster {nearest_idx} "
                     f"(거리 {nearest_dist:.2f}) — 그룹별 분리 기여도")
        if grand > 1e-6:
            for name, s, e in layout:
                share = 100.0 * totals[name] / grand
                bar = "█" * int(round(share / 4)) if share > 0 else ""
                lines.append(f"│    {name:<14} {e-s}D {share:>5.1f}% {bar:<25}")
        else:
            lines.append("│    ⚠ 두 centroid가 사실상 같은 지점")

    lines.append("│")
    lines.append(f"│  {'cl':>3} {'count':>8} {'반경p90':>8} {'최근접':>8} {'분리비':>7} "
                 f"{'meanVis':>8} {'score':>8}")
    for i in range(min(k, max_clusters)):
        st = stats[i] if i < len(stats) else {}
        cnt = st.get("count", 0)
        if cnt == 0:
            lines.append(f"│  {i:>3} {'(비어있음)':>8}")
            continue
        radius = st.get("radius_p90")
        nearest = st.get("nearest_centroid_dist")
        if radius is None or nearest is None:
            ratio_txt = "     -"
        elif radius < 1e-3:
            ratio_txt = "  한점"
        else:
            ratio_txt = f"{nearest / radius:>6.2f}"
        sc = float(scores[i]) if scores is not None and i < len(scores) else float("nan")
        flag = "  ◀ best" if i == best else ""
        lines.append(
            f"│  {i:>3} {cnt:>8} "
            f"{radius if radius is not None else float('nan'):>8.2f} "
            f"{nearest if nearest is not None else float('nan'):>8.2f} "
            f"{ratio_txt:>7} "
            f"{st.get('mean_visibility', 0.0):>8.4f} {sc:>8.4f}{flag}")
    if k > max_clusters:
        lines.append(f"│  ... ({k - max_clusters}개 더)")
    lines.append("│  분리비 = 최근접 centroid 거리 / 자기 반경")
    lines.extend(_pairwise_contribution_lines(centroids, layout, max_clusters))
    lines.append("└─")
    return "\n".join(lines)


def _pairwise_contribution_lines(centroids, feature_layout, max_clusters=12):
    """모든 cluster 쌍의 centroid 거리와 그룹별 분리 기여도(제곱거리 비율, %) 표."""
    k = min(centroids.shape[0], max_clusters)
    if k < 2:
        return []
    names = [name for name, _, _ in feature_layout]
    lines = ["│", "│  cluster 쌍별 분리 기여도 (centroid 제곱거리 중 그룹 비율 %)",
             "│  " + f"{'쌍':>6} {'거리':>6} " + " ".join(f"{n[:10]:>10}" for n in names)]
    for a in range(k):
        for b in range(a + 1, k):
            d2 = (centroids[a] - centroids[b]) ** 2
            total = float(d2.sum())
            shares = [100.0 * float(d2[s:e].sum()) / total if total > 1e-12 else 0.0
                      for _, s, e in feature_layout]
            lines.append("│  " + f"{f'{a}-{b}':>6} {np.sqrt(total):>6.2f} " +
                         " ".join(f"{sh:>10.1f}" for sh in shares))
    return lines


def format_stage_b(debug_info, mask=None):
    """Stage B 요약: 사용 feature, RF 학습/판정 통계, 최종 mask 크기와 형태."""
    lines = ["┌─ Stage B: 볼륨 전체 분류 [Random Forest, voxel 단위]",
             f"│  사용 feature: {debug_info.get('used_groups')} (d={debug_info.get('reduced_dim')})"]
    if debug_info.get("unclassifiable_count"):
        lines.append(f"│  판정 불가(화면 밖/카메라 뒤/클리핑) {debug_info['unclassifiable_count']:,} voxel")

    rf = debug_info.get("rf")
    if rf:
        lines.append(f"│  학습: tree {rf['n_estimators']}{', INR만' if rf.get('inr_only') else ''}, "
                     f"best {rf['n_best']:,} : not-best {rf['n_not_best']:,}, "
                     f"학습셋 정확도 {rf['train_accuracy']:.3f}")
        lines.append(f"│    not-best 제외 영역: best seed+마진 {rf['margin']} voxel "
                     f"= {rf['margin_excluded']:,} voxel")
        prob_txt = f", P(best)≥{rf['min_best_prob']:.2f}" if rf["min_best_prob"] > 0 else ""
        lines.append(f"│  best 판정(P(best)>0.5{prob_txt}) P(best) 중앙값 "
                     f"{rf['kept_p_best_median']:.2f}, 하위10% {rf['kept_p_best_p10']:.2f}")

    frac = debug_info.get("voxel_fraction")
    if frac is not None:
        lines.append(f"│  최종 mask {debug_info.get('voxel_count'):,} voxel ({100.0*frac:.3f}% of volume)")

    shape = debug_info.get("mask_shape")
    if shape:
        lines.append(f"│  형태: 연결성분 {shape['components']}개  "
                     f"최대성분 {shape['largest_pct']:.1f}%  "
                     f"자기이웃점유(occ) {shape['occupancy']:.3f}")
        if shape["occupancy"] < 0.35:
            lines.append("│    ⚠ occ가 낮음 — mask가 얇거나 흩어져 있음")

    if debug_info.get("warning"):
        lines.append(f"│  ⚠ {debug_info['warning']}")
    lines.append("└─")
    return "\n".join(lines)


def mask_shape_summary(mask):
    """mask 형태 요약. Returns: {components, largest_pct, occupancy} 또는 None(빈 mask)

    occupancy: mask voxel마다 3×3×3 이웃 중 mask 비율의 평균.
    """
    m = mask > 0
    n = int(m.sum())
    if n == 0:
        return None
    structure = np.ones((3, 3, 3), dtype=bool)
    labeled, num = ndimage.label(m, structure=structure)
    sizes = np.bincount(labeled.ravel())[1:]
    occ = ndimage.convolve(m.astype(np.float32), np.ones((3, 3, 3), dtype=np.float32),
                           mode="constant")[m] / 27.0
    return {"components": int(num),
            "largest_pct": 100.0 * float(sizes.max()) / n,
            "occupancy": float(occ.mean())}
