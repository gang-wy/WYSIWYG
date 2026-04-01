import math


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

    # sRGB D65
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
    # D65 white point
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


def preview_brightness(tf_editor, base_tf_nodes, roi_info, strength=0.15, mode="increase", feather=0.0):
    """
    ROI intensity range에 해당하는 TF node의 RGB를 Lab로 변환한 뒤
    L 채널만 증감시키고 다시 RGB로 복원한다.

    보강:
    - ROI 내부 node가 0개면 helper node를 먼저 삽입

    tf node format: [x, r, g, b, a]
    """
    if base_tf_nodes is None or len(base_tf_nodes) == 0:
        raise ValueError("base_tf_nodes is empty.")

    if roi_info is None or "range_norm" not in roi_info:
        raise ValueError("roi_info['range_norm'] is required.")

    range_min, range_max = roi_info["range_norm"]
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

        # feather는 현재 0 고정이므로 binary selection
        weight = _compute_binary_weight(x, range_min, range_max)
        weights.append(weight)

        if weight <= 0.0:
            continue

        affected_count += 1

        l_val, a_val, b_val = _rgb_to_lab(r, g, b)
        l_before_values.append(l_val)

        # L 범위는 보통 [0, 100]
        delta_l = strength * 100.0 * weight

        if normalized_mode == "decrease":
            new_l = _clamp(l_val - delta_l, 0.0, 100.0)
        else:
            new_l = _clamp(l_val + delta_l, 0.0, 100.0)

        new_r, new_g, new_b = _lab_to_rgb(new_l, a_val, b_val)

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