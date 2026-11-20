"""WYSIWYG 도구 공통 유틸: sRGB(0~1) ↔ CIE Lab(D65) 변환, ROI 입력 검증, 구간 가중치."""


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def srgb_to_linear(c):
    c = clamp(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def linear_to_srgb(c):
    c = max(0.0, c)
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055


def rgb_to_xyz(r, g, b):
    r_lin = srgb_to_linear(r)
    g_lin = srgb_to_linear(g)
    b_lin = srgb_to_linear(b)

    # sRGB D65
    x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
    y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
    z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041
    return x, y, z


def xyz_to_rgb(x, y, z):
    r_lin = x * 3.2404542 + y * -1.5371385 + z * -0.4985314
    g_lin = x * -0.9692660 + y * 1.8760108 + z * 0.0415560
    b_lin = x * 0.0556434 + y * -0.2040259 + z * 1.0572252

    r = clamp(linear_to_srgb(r_lin))
    g = clamp(linear_to_srgb(g_lin))
    b = clamp(linear_to_srgb(b_lin))
    return r, g, b


def f_xyz_to_lab(t):
    delta = 6.0 / 29.0
    if t > delta ** 3:
        return t ** (1.0 / 3.0)
    return (t / (3.0 * delta ** 2)) + (4.0 / 29.0)


def f_lab_to_xyz(t):
    delta = 6.0 / 29.0
    if t > delta:
        return t ** 3
    return 3.0 * (delta ** 2) * (t - 4.0 / 29.0)


def xyz_to_lab(x, y, z):
    # D65 white point
    xn, yn, zn = 0.95047, 1.00000, 1.08883

    fx = f_xyz_to_lab(x / xn)
    fy = f_xyz_to_lab(y / yn)
    fz = f_xyz_to_lab(z / zn)

    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return l, a, b


def lab_to_xyz(l, a, b):
    fy = (l + 16.0) / 116.0
    fx = fy + (a / 500.0)
    fz = fy - (b / 200.0)

    xn, yn, zn = 0.95047, 1.00000, 1.08883

    x = xn * f_lab_to_xyz(fx)
    y = yn * f_lab_to_xyz(fy)
    z = zn * f_lab_to_xyz(fz)
    return x, y, z


def rgb_to_lab(r, g, b):
    """Returns: (L 0~100, a, b)"""
    x, y, z = rgb_to_xyz(r, g, b)
    return xyz_to_lab(x, y, z)


def lab_to_rgb(l, a, b):
    """Returns: (r, g, b) 0~1로 clamp"""
    x, y, z = lab_to_xyz(l, a, b)
    return xyz_to_rgb(x, y, z)


def compute_binary_weight(x, range_min, range_max):
    """Returns: x가 [range_min, range_max] 안이면 1.0, 아니면 0.0"""
    return 1.0 if range_min <= x <= range_max else 0.0


def require_roi_range(base_tf_nodes, roi_info):
    """도구 입력을 검증한다. Returns: roi_info["range_norm"] (min, max)"""
    if base_tf_nodes is None or len(base_tf_nodes) == 0:
        raise ValueError("base_tf_nodes is empty.")
    if roi_info is None or "range_norm" not in roi_info:
        raise ValueError("roi_info['range_norm'] is required.")
    return roi_info["range_norm"]
