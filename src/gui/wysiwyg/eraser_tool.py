from src.gui.wysiwyg.color_math import clamp, compute_binary_weight, require_roi_range


def preview_eraser(tf_editor, base_tf_nodes, roi_info, strength=0.5, feather=0.1, mode="decrease"):
    """ROI intensity 구간(range_norm) 안 node의 opacity를 높이거나(increase) 낮춘다(decrease).

    구간 안에 node가 없으면 보조 node 3개를 먼저 넣는다. feather는 사용하지 않는다(구간 안 = 가중치 1).
    Returns: (new_nodes, debug_info)
    """
    range_min, range_max = require_roi_range(base_tf_nodes, roi_info)
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
        weight = compute_binary_weight(x, range_min, range_max)
        weights.append(weight)

        before_opacities.append(float(a))

        if weight > 0.0:
            affected_count += 1

            if normalized_mode == "increase":
                new_a = a + (1.0 - a) * strength * weight
            else:
                new_a = a * (1.0 - strength * weight)

            node[4] = clamp(new_a)

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