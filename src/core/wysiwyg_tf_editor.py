import numpy as np


class WysiwygTFEditor:
    """
    ROI intensity support(range_norm)를 기반으로
    TF를 국소적으로 수정하는 간단한 편집 엔진.

    1차 구현:
    - opacity increase
    - opacity decrease
    """

    def __init__(self, lut_size=256):
        self.lut_size = lut_size

    def _sample_nodes_to_lut(self, tf_nodes):
        """
        tf_nodes: [[x, r, g, b, a], ...]
        -> color_lut: (N, 3)
        -> opacity_lut: (N,)
        """
        if tf_nodes is None or len(tf_nodes) == 0:
            raise ValueError("tf_nodes is empty")

        nodes = np.array(tf_nodes, dtype=np.float32)
        xs_nodes = nodes[:, 0]
        rs_nodes = nodes[:, 1]
        gs_nodes = nodes[:, 2]
        bs_nodes = nodes[:, 3]
        alphas_nodes = nodes[:, 4]

        xs = np.linspace(0.0, 1.0, self.lut_size, dtype=np.float32)

        r = np.interp(xs, xs_nodes, rs_nodes)
        g = np.interp(xs, xs_nodes, gs_nodes)
        b = np.interp(xs, xs_nodes, bs_nodes)
        a = np.interp(xs, xs_nodes, alphas_nodes)

        color_lut = np.stack([r, g, b], axis=1)
        opacity_lut = a
        return xs, color_lut, opacity_lut

    def _lut_to_nodes(self, xs, color_lut, opacity_lut, num_nodes=32):
        """
        수정된 LUT를 다시 node 형태로 다운샘플링
        """
        if len(xs) != len(opacity_lut) or len(xs) != len(color_lut):
            raise ValueError("LUT size mismatch")

        indices = np.linspace(0, len(xs) - 1, num_nodes).astype(int)

        nodes = []
        for i in indices:
            nodes.append([
                float(xs[i]),
                float(color_lut[i, 0]),
                float(color_lut[i, 1]),
                float(color_lut[i, 2]),
                float(opacity_lut[i]),
            ])
        return nodes

    def _make_range_weight(self, xs, low, high, feather=0.03):
        """
        ROI 구간 주변만 0~1 weight를 갖도록 생성.
        feather는 양끝 부드러운 전이 폭.
        """
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))

        if high < low:
            low, high = high, low

        weights = np.zeros_like(xs, dtype=np.float32)

        # 중심 구간
        inner = (xs >= low) & (xs <= high)
        weights[inner] = 1.0

        # 왼쪽 feather
        left_start = max(0.0, low - feather)
        left_mask = (xs >= left_start) & (xs < low)
        if np.any(left_mask) and feather > 1e-8:
            weights[left_mask] = (xs[left_mask] - left_start) / feather

        # 오른쪽 feather
        right_end = min(1.0, high + feather)
        right_mask = (xs > high) & (xs <= right_end)
        if np.any(right_mask) and feather > 1e-8:
            weights[right_mask] = (right_end - xs[right_mask]) / feather

        return np.clip(weights, 0.0, 1.0)

    def apply_opacity_delta(self, tf_nodes, roi_info, delta=0.15, feather=0.03, num_nodes=32):
        """
        ROI support 구간에 opacity delta를 더함.
        delta > 0 : increase
        delta < 0 : decrease
        """
        if roi_info is None:
            raise ValueError("roi_info is None")

        if "range_norm" not in roi_info:
            raise ValueError("roi_info does not contain 'range_norm'")

        low, high = roi_info["range_norm"]

        xs, color_lut, opacity_lut = self._sample_nodes_to_lut(tf_nodes)
        weights = self._make_range_weight(xs, low, high, feather=feather)

        new_opacity = np.clip(opacity_lut + delta * weights, 0.0, 1.0)

        new_nodes = self._lut_to_nodes(xs, color_lut, new_opacity, num_nodes=num_nodes)

        debug_info = {
            "range_norm": (float(low), float(high)),
            "delta": float(delta),
            "feather": float(feather),
            "max_weight": float(np.max(weights)),
            "mean_weight": float(np.mean(weights)),
            "opacity_before_mean": float(np.mean(opacity_lut)),
            "opacity_after_mean": float(np.mean(new_opacity)),
        }

        return new_nodes, debug_info

    def apply_opacity_increase(self, tf_nodes, roi_info, strength=0.15, feather=0.03, num_nodes=32):
        return self.apply_opacity_delta(
            tf_nodes=tf_nodes,
            roi_info=roi_info,
            delta=abs(strength),
            feather=feather,
            num_nodes=num_nodes,
        )

    def apply_opacity_decrease(self, tf_nodes, roi_info, strength=0.15, feather=0.03, num_nodes=32):
        return self.apply_opacity_delta(
            tf_nodes=tf_nodes,
            roi_info=roi_info,
            delta=-abs(strength),
            feather=feather,
            num_nodes=num_nodes,
        )