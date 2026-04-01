import numpy as np

from src.core.utils.tf_utils import find_target_range_from_tents


class ROIFeatureExtractor:
    """
    FeatureAnalyzer의 raw 결과를 받아
    WYSIWYG 편집에 적합한 ROI intensity support 정보로 정리한다.
    """

    def __init__(self, volume_range):
        self.vol_min = float(volume_range[0])
        self.vol_max = float(volume_range[1])
        self.vol_span = max(self.vol_max - self.vol_min, 1e-8)

    def _to_norm(self, value):
        return float(np.clip((value - self.vol_min) / self.vol_span, 0.0, 1.0))

    def _range_to_norm(self, value_range):
        low, high = value_range
        low_n = self._to_norm(low)
        high_n = self._to_norm(high)
        return (min(low_n, high_n), max(low_n, high_n))

    def _robust_filter(self, picked_intensities, picked_points=None, sam_weights=None):
        """
        1차 버전:
        median ± std 범위만 남기는 간단한 robust filtering
        """
        picked_intensities = np.asarray(picked_intensities, dtype=np.float32)

        if len(picked_intensities) == 0:
            return picked_intensities, picked_points, sam_weights, None

        median = float(np.median(picked_intensities))
        std = float(np.std(picked_intensities))

        # std가 거의 0이면 그대로 반환
        if std < 1e-8:
            return picked_intensities, picked_points, sam_weights, {
                "median": median,
                "std": std,
                "valid_ratio": 1.0,
            }

        valid_mask = (
            (picked_intensities >= median - std) &
            (picked_intensities <= median + std)
        )

        filtered_intensities = picked_intensities[valid_mask]

        filtered_points = None
        if picked_points is not None:
            picked_points = np.asarray(picked_points)
            filtered_points = picked_points[valid_mask]

        filtered_weights = None
        if sam_weights is not None:
            sam_weights = np.asarray(sam_weights)
            filtered_weights = sam_weights[valid_mask]

        # 너무 많이 날아가면 원본 유지
        if len(filtered_intensities) == 0:
            filtered_intensities = picked_intensities
            filtered_points = np.asarray(picked_points) if picked_points is not None else None
            filtered_weights = np.asarray(sam_weights) if sam_weights is not None else None
            valid_ratio = 0.0
        else:
            valid_ratio = float(len(filtered_intensities) / len(picked_intensities))

        return filtered_intensities, filtered_points, filtered_weights, {
            "median": median,
            "std": std,
            "valid_ratio": valid_ratio,
        }

    def extract(self, analyzer_results, tf_nodes, sam_weights=None):
        """
        Args:
            analyzer_results: FeatureAnalyzer.analyze_roi_profile() 결과
            tf_nodes: 현재 TF node list [[x, r, g, b, a], ...]
            sam_weights: (optional) mask confidence / logits 기반 가중치

        Returns:
            roi_info dict or None
        """
        if analyzer_results is None:
            return None

        picked_intensities = analyzer_results.get("picked_intensities", None)
        picked_points = analyzer_results.get("picked_points", None)

        if picked_intensities is None or len(picked_intensities) == 0:
            return None

        picked_intensities = np.asarray(picked_intensities, dtype=np.float32)
        picked_points = np.asarray(picked_points) if picked_points is not None else None

        filtered_intensities, filtered_points, filtered_weights, filter_stats = self._robust_filter(
            picked_intensities=picked_intensities,
            picked_points=picked_points,
            sam_weights=sam_weights,
        )

        # TF tent 기반 support range 계산
        low_q = float(np.percentile(filtered_intensities, 20))
        high_q = float(np.percentile(filtered_intensities, 80))
        range_real = (low_q, high_q)

        center_real = float(np.median(filtered_intensities))
        sigma_real = float(np.std(filtered_intensities))
        min_real = float(np.min(filtered_intensities))
        max_real = float(np.max(filtered_intensities))

        # 히스토그램은 디버깅/시각화/도구 판단용
        hist, bin_edges = np.histogram(
            filtered_intensities,
            bins=64,
            range=(self.vol_min, self.vol_max)
        )

        roi_info = {
            # raw
            "picked_intensities": picked_intensities,
            "picked_points": picked_points,

            # filtered
            "filtered_intensities": filtered_intensities,
            "filtered_points": filtered_points,
            "weights": filtered_weights,

            # statistics (real scale)
            "center_real": center_real,
            "range_real": (float(range_real[0]), float(range_real[1])),
            "min_real": min_real,
            "max_real": max_real,
            "sigma_real": sigma_real,

            # normalized (0~1)
            "center_norm": self._to_norm(center_real),
            "range_norm": self._range_to_norm(range_real),
            "min_norm": self._to_norm(min_real),
            "max_norm": self._to_norm(max_real),

            # debug / visualization
            "histogram": hist,
            "histogram_bin_edges": bin_edges,
            "filter_stats": filter_stats,
            "num_samples_raw": int(len(picked_intensities)),
            "num_samples_filtered": int(len(filtered_intensities)),
        }

        return roi_info