from src.gui.wysiwyg.color_math import clamp, rgb_to_lab, lab_to_rgb, compute_binary_weight, require_roi_range


def preview_brightness(tf_editor, base_tf_nodes, roi_info, strength=0.15, mode="increase", feather=0.0):
    """ROI intensity 구간 안 node 색의 Lab L(밝기)을 strength × 100만큼 올리거나 내린다(opacity 유지).

    구간 안에 node가 없으면 보조 node 3개를 먼저 넣는다.
    Returns: (new_nodes, debug_info)
    """
    range_min, range_max = require_roi_range(base_tf_nodes, roi_info)
    normalized_mode = str(mode).strip().lower()

    if normalized_mode not in ("increase", "decrease"):
        normalized_mode = "increase"

    new_nodes, inserted, inserted_positions = tf_editor.ensure_nodes_in_roi(
        tf_nodes=base_tf_nodes,
        roi_range=(range_min, range_max),
        insert_mode="three",
    )

    new_nodes = [node[:] for node in new_nodes]

    l_before_values = []
    l_after_values = []
    affected_count = 0
    weights = []

    for node in new_nodes:
        x, r, g, b, a = node
        weight = compute_binary_weight(x, range_min, range_max)
        weights.append(weight)

        if weight <= 0.0:
            continue

        affected_count += 1

        l_val, a_val, b_val = rgb_to_lab(r, g, b)
        l_before_values.append(l_val)
        delta_l = strength * 100.0 * weight

        if normalized_mode == "decrease":
            new_l = clamp(l_val - delta_l, 0.0, 100.0)
        else:
            new_l = clamp(l_val + delta_l, 0.0, 100.0)

        new_r, new_g, new_b = lab_to_rgb(new_l, a_val, b_val)

        node[1] = new_r
        node[2] = new_g
        node[3] = new_b

        l_after_values.append(new_l)

    debug_info = {
        "tool": "brightness",
        "mode": normalized_mode,
        "range_norm": (range_min, range_max),
        "feather": feather,
        "strength": strength,
        "affected_count": affected_count,
        "mean_weight": sum(weights) / len(weights) if weights else 0.0,
        "l_before_mean": sum(l_before_values) / len(l_before_values) if l_before_values else 0.0,
        "l_after_mean": sum(l_after_values) / len(l_after_values) if l_after_values else 0.0,
        "inserted": inserted,
        "inserted_positions": inserted_positions,
    }

    return new_nodes, debug_info