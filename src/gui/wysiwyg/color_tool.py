from src.gui.wysiwyg.color_math import clamp, rgb_to_lab, lab_to_rgb, compute_binary_weight, require_roi_range


def _normalize_input_color(color):
    """(r,g,b) 0~1 / (r,g,b) 0~255 / "#RRGGBB" → (r, g, b) 0~1. None이면 빨강."""
    if color is None:
        return 1.0, 0.0, 0.0

    if isinstance(color, str):
        color = color.strip()
        if color.startswith("#") and len(color) == 7:
            r = int(color[1:3], 16) / 255.0
            g = int(color[3:5], 16) / 255.0
            b = int(color[5:7], 16) / 255.0
            return r, g, b
        raise ValueError(f"Unsupported color string: {color}")

    if isinstance(color, (list, tuple)) and len(color) == 3:
        r, g, b = color
        if max(r, g, b) > 1.0:
            return clamp(r / 255.0), clamp(g / 255.0), clamp(b / 255.0)
        return clamp(r), clamp(g), clamp(b)

    raise ValueError(f"Unsupported color format: {color}")


def preview_colorization(
    tf_editor,
    base_tf_nodes,
    roi_info,
    strength=0.25,
    feather=0.0,
    mode="apply",
    color=None,
):
    """ROI intensity 구간 안 node 색을 목표 색 쪽으로 Lab 공간에서 strength만큼 보간한다(opacity 유지).

    구간 안에 node가 없으면 보조 node 3개를 먼저 넣는다. mode="restore"면 색을 바꾸지 않는다.
    Returns: (new_nodes, debug_info)
    """
    range_min, range_max = require_roi_range(base_tf_nodes, roi_info)
    target_r, target_g, target_b = _normalize_input_color(color)
    target_l, target_a, target_b_lab = rgb_to_lab(target_r, target_g, target_b)

    normalized_mode = str(mode).strip().lower()

    new_nodes, inserted, inserted_positions = tf_editor.ensure_nodes_in_roi(
        tf_nodes=base_tf_nodes,
        roi_range=(range_min, range_max),
        insert_mode="three",
    )

    new_nodes = [node[:] for node in new_nodes]

    affected_count = 0
    weights = []

    before_lab = []
    after_lab = []

    for node in new_nodes:
        x, r, g, b, alpha = node

        weight = compute_binary_weight(x, range_min, range_max)
        weights.append(weight)

        if weight <= 0.0:
            continue

        affected_count += 1

        src_l, src_a, src_b = rgb_to_lab(r, g, b)
        before_lab.append((src_l, src_a, src_b))

        t = clamp(strength * weight)

        if normalized_mode == "restore":
            t = 0.0

        new_l = (1.0 - t) * src_l + t * target_l
        new_a = (1.0 - t) * src_a + t * target_a
        new_b = (1.0 - t) * src_b + t * target_b_lab

        out_r, out_g, out_b = lab_to_rgb(new_l, new_a, new_b)

        node[1] = out_r
        node[2] = out_g
        node[3] = out_b

        after_lab.append((new_l, new_a, new_b))

    debug_info = {
        "tool": "colorization",
        "mode": normalized_mode,
        "range_norm": (range_min, range_max),
        "feather": feather,
        "strength": strength,
        "affected_count": affected_count,
        "target_rgb": (target_r, target_g, target_b),
        "mean_weight": sum(weights) / len(weights) if weights else 0.0,
        "l_before_mean": sum(v[0] for v in before_lab) / len(before_lab) if before_lab else 0.0,
        "l_after_mean": sum(v[0] for v in after_lab) / len(after_lab) if after_lab else 0.0,
        "inserted": inserted,
        "inserted_positions": inserted_positions,
    }

    return new_nodes, debug_info