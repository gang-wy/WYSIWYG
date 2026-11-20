from src.gui.wysiwyg.color_math import rgb_to_lab, lab_to_rgb, compute_binary_weight, require_roi_range


def _expand_indices_for_minimum_count(all_nodes, selected_indices, minimum_count=3):
    """선택 node가 minimum_count보다 적으면 좌우 이웃 index를 번갈아 추가한다. Returns: 정렬된 index 목록"""
    if len(selected_indices) >= minimum_count:
        return sorted(selected_indices)

    selected = set(selected_indices)

    if not selected:
        return []

    left = min(selected) - 1
    right = max(selected) + 1

    while len(selected) < minimum_count and (left >= 0 or right < len(all_nodes)):
        if left >= 0:
            selected.add(left)
            left -= 1
            if len(selected) >= minimum_count:
                break

        if right < len(all_nodes):
            selected.add(right)
            right += 1

    return sorted(selected)


def preview_contrast(tf_editor, base_tf_nodes, roi_info, strength=0.2, feather=0.0, mode="increase"):
    """ROI intensity 구간 안 node들의 Lab L을 평균 기준으로 (1 ± strength)배 벌리거나 좁힌다(opacity 유지).

    구간 안 node가 3개 미만이면 이웃 node를 포함해 3개로 맞춘다.
    Returns: (new_nodes, debug_info)
    """
    range_min, range_max = require_roi_range(base_tf_nodes, roi_info)
    normalized_mode = str(mode).strip().lower()

    new_nodes = [node[:] for node in base_tf_nodes]

    roi_indices = []
    l_values = []

    for i, node in enumerate(new_nodes):
        x, r, g, b, a = node
        weight = compute_binary_weight(x, range_min, range_max)
        if weight > 0.0:
            roi_indices.append(i)

    original_roi_count = len(roi_indices)

    roi_indices = _expand_indices_for_minimum_count(new_nodes, roi_indices, minimum_count=3)

    if not roi_indices:
        debug_info = {
            "tool": "contrast",
            "mode": normalized_mode,
            "range_norm": (range_min, range_max),
            "feather": feather,
            "strength": strength,
            "affected_count": 0,
            "original_roi_count": 0,
            "expanded": False,
            "l_mean_before": 0.0,
            "l_std_before": 0.0,
            "l_std_after": 0.0,
            "scale": 1.0,
        }
        return new_nodes, debug_info

    lab_cache = {}
    for idx in roi_indices:
        _, r, g, b, _ = new_nodes[idx]
        l_val, a_val, b_val = rgb_to_lab(r, g, b)
        lab_cache[idx] = (l_val, a_val, b_val)
        l_values.append(l_val)

    l_mean = sum(l_values) / len(l_values)
    l_var = sum((v - l_mean) ** 2 for v in l_values) / len(l_values)
    l_std_before = l_var ** 0.5

    if normalized_mode == "decrease":
        scale = max(0.0, 1.0 - strength)
    else:
        scale = 1.0 + strength

    l_after_values = []

    for idx in roi_indices:
        l_val, a_val, b_val = lab_cache[idx]
        new_l = l_mean + (l_val - l_mean) * scale
        new_l = max(0.0, min(100.0, new_l))

        out_r, out_g, out_b = lab_to_rgb(new_l, a_val, b_val)

        new_nodes[idx][1] = out_r
        new_nodes[idx][2] = out_g
        new_nodes[idx][3] = out_b

        l_after_values.append(new_l)

    l_after_mean = sum(l_after_values) / len(l_after_values)
    l_after_var = sum((v - l_after_mean) ** 2 for v in l_after_values) / len(l_after_values)
    l_std_after = l_after_var ** 0.5

    debug_info = {
        "tool": "contrast",
        "mode": normalized_mode,
        "range_norm": (range_min, range_max),
        "feather": feather,
        "strength": strength,
        "affected_count": len(roi_indices),
        "original_roi_count": original_roi_count,
        "expanded": len(roi_indices) != original_roi_count,
        "l_mean_before": l_mean,
        "l_std_before": l_std_before,
        "l_std_after": l_std_after,
        "scale": scale,
    }

    return new_nodes, debug_info