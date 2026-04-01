def _clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _compute_binary_weight(x, range_min, range_max):
    return 1.0 if range_min <= x <= range_max else 0.0


def preview_eraser(tf_editor, base_tf_nodes, roi_info, strength=0.5, feather=0.1, mode="decrease"):
    """
    ROI 내부 TF node의 alpha를 증가/감소시킨다.

    보강:
    - ROI 안에 node가 없으면 helper node(left/center/right) 삽입
    - feather는 현재 실사용 0 고정이므로 binary selection 중심
    """
    if base_tf_nodes is None or len(base_tf_nodes) == 0:
        raise ValueError("base_tf_nodes is empty.")

    if roi_info is None or "range_norm" not in roi_info:
        raise ValueError("roi_info['range_norm'] is required.")

    range_min, range_max = roi_info["range_norm"]
    normalized_mode = str(mode).strip().lower()

    if normalized_mode not in ("increase", "decrease"):
        normalized_mode = "decrease"

    new_nodes, inserted, inserted_positions = tf_editor.ensure_nodes_in_roi(
        tf_nodes=base_tf_nodes,
        roi_range=(range_min, range_max),
        insert_mode="three",
    )

    new_nodes = [node[:] for node in new_nodes]

    before_opacities = []
    after_opacities = []
    weights = []
    affected_count = 0

    for node in new_nodes:
        x, r, g, b, a = node

        # 현재 feather는 안 쓰는 방향이므로 binary range selection
        weight = _compute_binary_weight(x, range_min, range_max)
        weights.append(weight)

        before_opacities.append(float(a))

        if weight > 0.0:
            affected_count += 1

            if normalized_mode == "increase":
                new_a = a + (1.0 - a) * strength * weight
            else:
                new_a = a * (1.0 - strength * weight)

            node[4] = _clamp(new_a)

        after_opacities.append(float(node[4]))

    debug_info = {
        "tool": "eraser",
        "mode": normalized_mode,
        "range_norm": (range_min, range_max),
        "feather": feather,
        "strength": strength,
        "affected_count": affected_count,
        "mean_weight": sum(weights) / len(weights) if weights else 0.0,
        "opacity_before_mean": sum(before_opacities) / len(before_opacities) if before_opacities else 0.0,
        "opacity_after_mean": sum(after_opacities) / len(after_opacities) if after_opacities else 0.0,
        "inserted": inserted,
        "inserted_positions": inserted_positions,
    }

    return new_nodes, debug_info