import numpy as np


class WysiwygTFEditor:
    """
    ROI intensity support(range_norm)를 기반으로
    TF를 국소적으로 수정할 때 사용하는 공통 편집 유틸.

    역할:
    - TF node 정렬
    - 특정 위치 node 보간 생성
    - ROI 내부 node가 없을 때 helper node 삽입
    - (선택적으로) LUT 샘플링/복원 유틸 제공

    주의:
    - 실제 도구 로직(eraser / brightness / colorization / contrast ...)은
      각 전용 tool 파일에 둔다.
    """

    def __init__(self, lut_size=256):
        self.lut_size = lut_size

    def _sample_nodes_to_lut(self, tf_nodes):
        """
        현재 직접 사용하지는 않지만,
        추후 LUT 기반 연속 편집이 필요할 때 사용 가능.
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
        현재 직접 사용하지는 않지만,
        추후 LUT 기반 연속 편집이 필요할 때 사용 가능.
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

    def _sort_nodes(self, tf_nodes):
        return sorted(tf_nodes, key=lambda n: n[0])

    def _interpolate_node_at(self, tf_nodes, x):
        """
        tf_nodes: [x, r, g, b, a]
        x 위치의 node 값을 선형보간으로 생성
        """
        if tf_nodes is None or len(tf_nodes) == 0:
            raise ValueError("tf_nodes is empty")

        x = float(np.clip(x, 0.0, 1.0))
        nodes = self._sort_nodes([[float(v) for v in node] for node in tf_nodes])

        if x <= nodes[0][0]:
            return [x, nodes[0][1], nodes[0][2], nodes[0][3], nodes[0][4]]

        if x >= nodes[-1][0]:
            return [x, nodes[-1][1], nodes[-1][2], nodes[-1][3], nodes[-1][4]]

        for i in range(len(nodes) - 1):
            x0, r0, g0, b0, a0 = nodes[i]
            x1, r1, g1, b1, a1 = nodes[i + 1]

            if x0 <= x <= x1:
                if abs(x1 - x0) < 1e-8:
                    return [x, r0, g0, b0, a0]

                t = (x - x0) / (x1 - x0)
                r = r0 + t * (r1 - r0)
                g = g0 + t * (g1 - g0)
                b = b0 + t * (b1 - b0)
                a = a0 + t * (a1 - a0)
                return [x, r, g, b, a]

        return [x, nodes[-1][1], nodes[-1][2], nodes[-1][3], nodes[-1][4]]

    def ensure_nodes_in_roi(self, tf_nodes, roi_range, insert_mode="three"):
        """
        ROI 안에 기존 TF node가 하나도 없을 때만 helper node 삽입.
        기본: left / center / right

        Returns:
            nodes, inserted(bool), inserted_positions(list)
        """
        if tf_nodes is None or len(tf_nodes) == 0:
            raise ValueError("tf_nodes is empty")

        if roi_range is None or len(roi_range) != 2:
            raise ValueError("roi_range must be (low, high)")

        low, high = roi_range
        low = float(np.clip(low, 0.0, 1.0))
        high = float(np.clip(high, 0.0, 1.0))

        if high < low:
            low, high = high, low

        nodes = self._sort_nodes([[float(v) for v in node] for node in tf_nodes])

        roi_nodes = [node for node in nodes if low <= node[0] <= high]
        if len(roi_nodes) > 0:
            return nodes, False, []

        center = 0.5 * (low + high)

        if insert_mode == "three":
            insert_positions = [low, center, high]
        else:
            insert_positions = [center]

        eps = 1e-6
        existing_x = [node[0] for node in nodes]
        inserted_positions = []

        for x in insert_positions:
            duplicated = any(abs(x - ex) < eps for ex in existing_x)
            if duplicated:
                continue

            new_node = self._interpolate_node_at(nodes, x)
            nodes.append(new_node)
            inserted_positions.append(float(x))
            existing_x.append(float(x))

        nodes = self._sort_nodes(nodes)
        return nodes, len(inserted_positions) > 0, inserted_positions