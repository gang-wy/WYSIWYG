def _clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def _srgb_to_linear(c):
    c = _clamp(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _linear_to_srgb(c):
    c = max(0.0, c)
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def _rgb_to_xyz(r, g, b):
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)

    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    return x, y, z


def _xyz_to_rgb(x, y, z):
    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    r = _clamp(_linear_to_srgb(r_lin))
    g = _clamp(_linear_to_srgb(g_lin))
    b = _clamp(_linear_to_srgb(b_lin))
    return r, g, b


def _f_xyz_to_lab(t):
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return t ** (1.0 / 3.0)
    return (t / (3.0 * delta ** 2)) + (4.0 / 29.0)


def _f_lab_to_xyz(t):
    delta = 6.0 / 29.0
    if t > delta:
        return t ** 3
    return 3.0 * (delta ** 2) * (t - 4.0 / 29.0)


def _xyz_to_lab(x, y, z):
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    fx = _f_xyz_to_lab(x / xn)
    fy = _f_xyz_to_lab(y / yn)
    fz = _f_xyz_to_lab(z / zn)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return l, a, b


def _lab_to_xyz(l, a, b):
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    xn, yn, zn = 0.95047, 1.00000, 1.08883
    x = xn * _f_lab_to_xyz(fx)
    y = yn * _f_lab_to_xyz(fy)
    z = zn * _f_lab_to_xyz(fz)
    return x, y, z


def _rgb_to_lab(r, g, b):
    x, y, z = _rgb_to_xyz(r, g, b)
    return _xyz_to_lab(x, y, z)


def _lab_to_rgb(l, a, b):
    x, y, z = _lab_to_xyz(l, a, b)
    return _xyz_to_rgb(x, y, z)


def _compute_binary_weight(x, range_min, range_max):
    return 1.0 if range_min <= x <= range_max else 0.0


def _expand_indices_for_minimum_count(all_nodes, selected_indices, minimum_count=3):
    """
    선택된 node 수가 너무 적을 때, 가장 가까운 좌우 이웃 node를 추가해서
    최소 개수를 맞춘다.
    """
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
    """
    ROI 내부 TF node들의 Lab L 분포 대비를 평균 기준으로 확대/축소한다.
    ROI에 걸린 node가 너무 적으면 인접 node를 포함해 최소 3개로 보정한다.
    alpha는 유지.
    """
    if base_tf_nodes is None or len(base_tf_nodes) == 0:
        raise ValueError("base_tf_nodes is empty.")

    if roi_info is None or "range_norm" not in roi_info:
        raise ValueError("roi_info['range_norm'] is required.")

    range_min, range_max = roi_info["range_norm"]
    normalized_mode = str(mode).strip().lower()

    new_nodes = [node[:] for node in base_tf_nodes]

    roi_indices = []
    l_values = []

    for i, node in enumerate(new_nodes):
        x, r, g, b, a = node
        weight = _compute_binary_weight(x, range_min, range_max)
        if weight > 0.0:
            roi_indices.append(i)

    original_roi_count = len(roi_indices)

    # 핵심 보정: node가 너무 적으면 주변 node까지 포함
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
        l_val, a_val, b_val = _rgb_to_lab(r, g, b)
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

        out_r, out_g, out_b = _lab_to_rgb(new_l, a_val, b_val)

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