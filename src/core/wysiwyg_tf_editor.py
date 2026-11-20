"""TF node list([x, r, g, b, a], x는 0~1 정규화 intensity) 공통 연산."""
import numpy as np


class WysiwygTFEditor:
    """node 정렬·보간, LUT 샘플링, ROI 보조 node 삽입."""

    def __init__(self, lut_size=256):
        self.lut_size = lut_size

    def sample_nodes_to_lut(self, tf_nodes, num_bins=None):
        """[0, 1]을 num_bins(기본 lut_size)개로 균등 샘플링해 node를 선형 보간한다.

        Returns: (xs (num_bins,), color_lut (num_bins, 3), opacity_lut (num_bins,)) float32
        """
        if tf_nodes is None or len(tf_nodes) == 0:
            raise ValueError("tf_nodes is empty")

        num_bins = num_bins or self.lut_size
        nodes = np.array(self.sort_nodes(tf_nodes), dtype=np.float32)
        xs_nodes = nodes[:, 0]
        rs_nodes = nodes[:, 1]
        gs_nodes = nodes[:, 2]
        bs_nodes = nodes[:, 3]
        alphas_nodes = nodes[:, 4]

        xs = np.linspace(0.0, 1.0, num_bins, dtype=np.float32)

        r = np.interp(xs, xs_nodes, rs_nodes)
        g = np.interp(xs, xs_nodes, gs_nodes)
        b = np.interp(xs, xs_nodes, bs_nodes)
        a = np.interp(xs, xs_nodes, alphas_nodes)

        color_lut = np.stack([r, g, b], axis=1).astype(np.float32)
        opacity_lut = a.astype(np.float32)
        return xs, color_lut, opacity_lut

    @staticmethod
    def sort_nodes(tf_nodes):
        return sorted(tf_nodes, key=lambda n: n[0])

    @staticmethod
    def interpolate_node_at(tf_nodes, x):
        """Returns: x(0~1로 clip) 위치의 선형 보간 node [x, r, g, b, a]. 양 끝 밖은 끝 node 값."""
        if tf_nodes is None or len(tf_nodes) == 0:
            raise ValueError("tf_nodes is empty")

        x = float(np.clip(x, 0.0, 1.0))
        nodes = WysiwygTFEditor.sort_nodes([[float(v) for v in node] for node in tf_nodes])

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
        """roi_range 안에 node가 하나도 없을 때만 보간 node를 넣는다(insert_mode="three"면 low/center/high, 아니면 center).

        Returns: (정렬된 nodes, 삽입 여부, 삽입한 x 목록)
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

        nodes = self.sort_nodes([[float(v) for v in node] for node in tf_nodes])

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

            new_node = self.interpolate_node_at(nodes, x)
            nodes.append(new_node)
            inserted_positions.append(float(x))
            existing_x.append(float(x))

        nodes = self.sort_nodes(nodes)
        return nodes, len(inserted_positions) > 0, inserted_positions