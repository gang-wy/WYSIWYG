"""Stage A: SAM ray 샘플을 K-means로 clustering하고 ROI(best cluster)를 고른다."""

import time
import numpy as np
from sklearn.cluster import KMeans

from src.core.clustering import sampling_rules

try:
    import torch
    _TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    torch = None
    _TORCH_CUDA_AVAILABLE = False


def _kmeanspp_init_torch(X, k, generator):
    """GPU 텐서 X에서 kmeans++ 방식으로 초기 centroid k개를 고른다. Returns: (k, D) 텐서"""
    n = X.shape[0]
    first_idx = torch.randint(0, n, (1,), generator=generator, device=X.device)
    centroids = [X[first_idx]]
    closest_dist_sq = torch.full((n,), float('inf'), device=X.device, dtype=X.dtype)
    for _ in range(1, k):
        dist_sq = torch.sum((X - centroids[-1]) ** 2, dim=1)
        closest_dist_sq = torch.minimum(closest_dist_sq, dist_sq)
        total = closest_dist_sq.sum()
        if total <= 0:
            next_idx = torch.randint(0, n, (1,), generator=generator, device=X.device)
        else:
            next_idx = torch.multinomial(closest_dist_sq / total, 1, generator=generator)
        centroids.append(X[next_idx])
    return torch.cat(centroids, dim=0)


def _lloyd_torch(X, centroids, max_iter=300, tol=1e-4):
    """Lloyd 반복. 빈 cluster는 이전 centroid를 유지한다. Returns: (labels, centroids)"""
    k = centroids.shape[0]
    labels = torch.zeros(X.shape[0], dtype=torch.long, device=X.device)
    for _ in range(max_iter):
        new_labels = torch.cdist(X, centroids).argmin(dim=1)
        new_centroids = centroids.clone()
        for ci in range(k):
            member_mask = new_labels == ci
            if member_mask.any():
                new_centroids[ci] = X[member_mask].mean(dim=0)
        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            break
    return labels, centroids


KMEANS_RESTART_SUBSAMPLE = 50000


def _kmeans_gpu(features, k, seed=42, max_iter=300, tol=1e-4, n_init=1):
    """GPU K-means(kmeans++ 초기화 + Lloyd 반복).

    n_init > 1이면 샘플 최대 KMEANS_RESTART_SUBSAMPLE개를 뽑아 그 위에서 n_init번 초기화부터 다시 돌리고,
    SSE가 가장 낮은 centroid에서 시작해 전체 샘플로 Lloyd를 한 번 더 돈다(샘플 대부분이 공기일 때
    초기화 한 번으로는 조직을 한 cluster로 묶는 지역해에 빠지기 쉽다). n_init=1이면 초기화 1번.

    Returns: (labels (N,) int64, centroids (k, D) float64)
    """
    device = torch.device('cuda')
    X = torch.as_tensor(features, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)

    if n_init <= 1:
        start = _kmeanspp_init_torch(X, k, generator)
    else:
        n = X.shape[0]
        pick = torch.randperm(n, generator=generator, device=device)[:min(n, KMEANS_RESTART_SUBSAMPLE)]
        sub = X[pick]
        best_sse = None
        for _ in range(int(n_init)):
            sub_labels, sub_centroids = _lloyd_torch(sub, _kmeanspp_init_torch(sub, k, generator), max_iter, tol)
            sse = float(((sub - sub_centroids[sub_labels]) ** 2).sum())
            if best_sse is None or sse < best_sse:
                best_sse, start = sse, sub_centroids

    labels, centroids = _lloyd_torch(X, start, max_iter, tol)
    return labels.cpu().numpy(), centroids.double().cpu().numpy()


def _kmeans_cpu(features, k, seed=42, n_init=1):
    """CPU K-means(sklearn). n_init번 초기화해 SSE가 가장 낮은 해를 고른다."""
    km = KMeans(n_clusters=k, n_init=max(1, int(n_init)), random_state=seed).fit(features)
    return km.labels_, km.cluster_centers_


class ROIFeatureExtractor:
    """ray 샘플로 feature 행렬을 만들고 K-means로 ROI cluster와 intensity 구간을 구한다.

    feature 그룹: INR embedding, coords, visibility, depth, intensity, opacity (각각 on/off).
    정규화: 열마다 z-score. sparse_minmax면 visibility·opacity 열만 min-max(min 0, max = 0이 아닌 값의 p99)
    — 샘플 대부분이 0인 열을 z-score하면 0이 아닌 소수 샘플이 거대한 값이 되어 거리를 지배하기 때문.
    """

    SPARSE_GROUPS = ("visibility", "opacity")

    def __init__(self, volume_range, k=10, beta1=1e-4,
                 percentile_low=5.0, percentile_high=95.0,
                 use_inr=True, use_coords=True, use_visibility=True, use_depth=True,
                 use_intensity=False, use_opacity=True, seed=42, sparse_minmax=False, n_init=1):
        self.vol_min = float(volume_range[0])
        self.vol_max = float(volume_range[1])
        self.vol_span = max(self.vol_max - self.vol_min, 1e-8)

        self.k = k
        self.beta1 = beta1
        self.seed = seed
        self.n_init = max(1, int(n_init))       # K-means 재시작 횟수(_kmeans_gpu 참고)
        self.sparse_minmax = sparse_minmax
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        self.use_inr = use_inr
        self.use_coords = use_coords
        self.use_visibility = use_visibility
        self.use_depth = use_depth
        self.use_intensity = use_intensity
        self.use_opacity = use_opacity

    @staticmethod
    def split_inr_embedding_and_coords(inr_features):
        """INR feature의 채널 축을 (embedding, 마지막 3채널 정규화 좌표)로 나눈다.

        (N, C) 샘플과 (X, Y, Z, C) 볼륨 모두 사용 가능. Returns: (embedding, coords)
        """
        num_channels = inr_features.shape[-1]
        coord_dims = min(3, num_channels)
        embedding = inr_features[..., :num_channels - coord_dims]
        coords = inr_features[..., num_channels - coord_dims:]
        return embedding, coords

    def build_named_feature_groups(self, inr_features, visibility, depths, intensities, opacity):
        """켜진 feature 그룹만 모은다. Returns: [(name, (N, d) 배열), ...]"""
        named_groups = []

        if inr_features is not None and (self.use_inr or self.use_coords):
            embedding, coords = self.split_inr_embedding_and_coords(inr_features)
            if self.use_inr and embedding.shape[-1] > 0:
                named_groups.append(("inr_embedding", embedding))
            if self.use_coords and coords.shape[-1] > 0:
                named_groups.append(("coords", coords))

        for name, arr, enabled in (
            ("visibility", visibility, self.use_visibility),
            ("depth", depths, self.use_depth),
            ("intensity", intensities, self.use_intensity),
            ("opacity", opacity, self.use_opacity),
        ):
            if enabled and arr is not None:
                named_groups.append((name, arr.reshape(-1, 1)))

        return named_groups

    @staticmethod
    def feature_layout_from_groups(named_groups):
        """그룹별로 결합 행렬에서 차지하는 열 범위. Returns: [(name, start_col, end_col), ...]"""
        layout = []
        col = 0
        for name, arr in named_groups:
            width = arr.shape[-1]
            layout.append((name, col, col + width))
            col += width
        return layout

    def to_norm(self, value):
        """볼륨 실제 값 → [0, 1] 정규화 값."""
        return float(np.clip((value - self.vol_min) / self.vol_span, 0.0, 1.0))

    def range_to_norm(self, value_range):
        low, high = self.to_norm(value_range[0]), self.to_norm(value_range[1])
        return (min(low, high), max(low, high))

    def normalize_features(self, features, feature_layout):
        """열마다 z-score (sparse_minmax면 visibility·opacity 열은 mean 0, std = 0이 아닌 값의 p99).

        Stage B가 같은 (x - feat_mean) / feat_std 식을 다시 쓴다. Returns: (feat_mean, feat_std, normalized)
        """
        features = np.asarray(features, dtype=np.float64)
        feat_mean = features.mean(axis=0)
        feat_std = features.std(axis=0)
        if self.sparse_minmax:
            for name, start, _ in feature_layout:
                if name in self.SPARSE_GROUPS:
                    col = features[:, start]
                    nonzero = col[col > 0]
                    feat_mean[start] = 0.0
                    feat_std[start] = float(np.percentile(nonzero, 99)) if len(nonzero) else 1.0
        return feat_mean, feat_std, (features - feat_mean) / (feat_std + 1e-8)

    def cluster_and_score(self, clustering_features, feature_layout, depths=None, visibility=None):
        """정규화(클래스 설명) 후 K-means를 돌리고 점수 S(i) = 평균 visibility + beta1 × 평균 depth가
        가장 높은 cluster를 best로 고른다. 정규화는 (x - feat_mean) / feat_std 한 식으로 표현해 Stage B가 그대로 쓴다
        (min-max 열은 feat_mean 0, feat_std p99).

        Returns: dict 또는 None(샘플 부족/실패)
            k, labels, centroids, feat_mean, feat_std, feature_layout,
            scores, cluster_stats, best_cluster_index, best_cluster_mask
        """
        n = len(clustering_features)
        if n < 2:
            return None
        k = max(1, min(self.k, n))

        feat_mean, feat_std, normalized = self.normalize_features(clustering_features, feature_layout)

        try:
            t0 = time.time()
            if _TORCH_CUDA_AVAILABLE:
                try:
                    labels, centroids = _kmeans_gpu(normalized, k, seed=self.seed, n_init=self.n_init)
                    backend = "GPU"
                except Exception as e:
                    print(f"⚠️ GPU K-means 실패({e}) — CPU로 대체")
                    labels, centroids = _kmeans_cpu(normalized, k, seed=self.seed, n_init=self.n_init)
                    backend = "CPU"
            else:
                labels, centroids = _kmeans_cpu(normalized, k, seed=self.seed, n_init=self.n_init)
                backend = "CPU"
            sse = float(((normalized - centroids[labels]) ** 2).sum())
            print(f"⏱️ [Stage A] K-means [{backend}] n={n}, d={normalized.shape[1]}, k={k}, seed={self.seed}, "
                  f"재시작 {self.n_init}: {time.time() - t0:.2f}초, SSE {sse:,.0f}")
        except Exception as e:
            print(f"⚠️ K-means 실패: {e}")
            return None

        scores = np.full(k, -np.inf, dtype=np.float32)
        cluster_stats = []
        for i in range(k):
            member_mask = labels == i
            count = int(np.sum(member_mask))
            if count == 0:
                cluster_stats.append({"count": 0, "mean_visibility": 0.0, "mean_depth": 0.0})
                continue

            mean_vis = float(np.mean(visibility[member_mask])) if visibility is not None else 0.0
            mean_depth = float(np.mean(depths[member_mask])) if depths is not None else 0.0

            # 진단용: 멤버 반경(p90)과 가장 가까운 다른 centroid까지 거리
            member_d = np.linalg.norm(normalized[member_mask] - centroids[i], axis=1)
            other = np.delete(np.arange(k), i)
            nearest_dist = (float(np.min(np.linalg.norm(centroids[other] - centroids[i], axis=1)))
                            if len(other) else float("nan"))

            scores[i] = mean_vis + self.beta1 * mean_depth
            cluster_stats.append({
                "count": count,
                "mean_visibility": mean_vis,
                "mean_depth": mean_depth,
                "radius_p90": float(np.percentile(member_d, 90)),
                "nearest_centroid_dist": nearest_dist,
            })

        best_idx = int(np.argmax(scores))
        return {
            "k": k,
            "labels": labels,
            "centroids": centroids,
            "feat_mean": feat_mean,
            "feat_std": feat_std,
            "feature_layout": feature_layout,
            "normalized": normalized,          # 진단 덤프용 (K-means 입력 그대로)
            "scores": scores,
            "cluster_stats": cluster_stats,
            "best_cluster_index": best_idx,
            "best_cluster_mask": labels == best_idx,
        }

    def extract(self, analyzer_results):
        """FeatureAnalyzer.analyze_roi_profile() 결과로 Stage A를 수행한다.

        Returns: roi_info dict 또는 None(샘플 없음)
            filtered_*           Stage A 입력 샘플 (points/intensities/inr_features/depths/visibility/opacity)
            cluster_result       cluster_and_score() 결과 (feature 그룹이 없으면 None)
            cluster_intensities  best cluster 샘플의 intensity
            center_norm          best cluster intensity 중앙값 (0~1)
            range_norm           best cluster intensity percentile 구간 (0~1)
            num_samples          전체 샘플 수
        """
        if analyzer_results is None:
            return None

        def as_array(key, dtype=np.float32):
            value = analyzer_results.get(key)
            return None if value is None else np.asarray(value, dtype=dtype)

        intensities = as_array("sampled_intensities")
        if intensities is None or len(intensities) == 0:
            return None
        points = as_array("sampled_points", dtype=np.float64)
        inr_features = as_array("sampled_inr_features")
        depths = as_array("sampled_depths")
        visibility = as_array("sampled_visibility")
        opacity = as_array("sampled_opacity")
        # feature 열은 V 또는 T, 점수는 항상 V (Stage B도 filtered_visibility = feature 열을 쓴다)
        vis_feature = (as_array("sampled_transmittance")
                       if sampling_rules.VISIBILITY_FEATURE == "T" else visibility)

        cluster_result = None
        named_groups = self.build_named_feature_groups(
            inr_features, vis_feature, depths, intensities, opacity)
        if named_groups:
            cluster_result = self.cluster_and_score(
                np.concatenate([arr for _, arr in named_groups], axis=1),
                self.feature_layout_from_groups(named_groups),
                depths=depths, visibility=visibility)

        cluster_intensities = intensities
        if cluster_result is not None and cluster_result["best_cluster_mask"].any():
            cluster_intensities = intensities[cluster_result["best_cluster_mask"]]

        range_real = (float(np.percentile(cluster_intensities, self.percentile_low)),
                      float(np.percentile(cluster_intensities, self.percentile_high)))
        center_real = float(np.median(cluster_intensities))

        return {
            "filtered_points": points,
            "filtered_intensities": intensities,
            "filtered_inr_features": inr_features,
            "filtered_depths": depths,
            "filtered_visibility": vis_feature,
            "filtered_opacity": opacity,
            "cluster_result": cluster_result,
            "cluster_intensities": cluster_intensities,
            "center_norm": self.to_norm(center_real),
            "range_norm": self.range_to_norm(range_real),
            "num_samples": int(len(intensities)),
        }
