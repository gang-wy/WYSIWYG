"""
공통 유틸리티 함수들
- TF LUT 변환
- Grid 샘플링
- 3D→2D 투영
"""

import numpy as np


def tf_nodes_to_opacity_lut(tf_nodes, lut_size=256):
    """
    TF 노드 리스트로부터 Opacity LUT 생성
    
    Args:
        tf_nodes: list of [intensity, R, G, B, A] (intensity: 0~1 normalized)
        lut_size: LUT 크기 (default: 256)
    
    Returns:
        numpy array of shape (lut_size,) with opacity values
    """
    lut = np.zeros(lut_size, dtype=np.float32)
    
    if not tf_nodes or len(tf_nodes) < 2:
        return lut
    
    sorted_nodes = sorted(tf_nodes, key=lambda x: x[0])
    
    for i in range(lut_size):
        intensity = i / (lut_size - 1)
        
        # 선형 보간
        if intensity <= sorted_nodes[0][0]:
            alpha = sorted_nodes[0][4]
        elif intensity >= sorted_nodes[-1][0]:
            alpha = sorted_nodes[-1][4]
        else:
            for j in range(len(sorted_nodes) - 1):
                if sorted_nodes[j][0] <= intensity <= sorted_nodes[j + 1][0]:
                    t = (intensity - sorted_nodes[j][0]) / (sorted_nodes[j + 1][0] - sorted_nodes[j][0])
                    alpha = sorted_nodes[j][4] * (1 - t) + sorted_nodes[j + 1][4] * t
                    break
        
        lut[i] = alpha
    
    return lut


def sample_grid_representative_points(projection_results, grid_size=4):
    """
    Grid 기반 대표점 샘플링
    
    투영된 점들의 Bounding Box를 기준으로 Grid를 나누고,
    각 Grid 격자점(Node)에서 가장 가까운 '실제 투영된 점'을 선택합니다.
    
    Args:
        projection_results: list of (original_idx, (x, y))
        grid_size: grid_size x grid_size 격자 (default: 4 -> 최대 16개 포인트)
        
    Returns:
        list of dict: [{'idx': original_3d_idx, 'pt': (x, y)}, ...]
    """
    if not projection_results:
        return []

    # 좌표 추출
    coords = np.array([p[1] for p in projection_results])  # (N, 2)
    original_indices = np.array([p[0] for p in projection_results])
    
    min_x, min_y = coords.min(axis=0)
    max_x, max_y = coords.max(axis=0)
    
    # Grid 생성
    x_grid = np.linspace(min_x, max_x, grid_size)
    y_grid = np.linspace(min_y, max_y, grid_size)
    
    sampled_candidates = []
    
    # 각 Grid 교차점에 대해 가장 가까운 실제 점 찾기
    for gx in x_grid:
        for gy in y_grid:
            grid_point = np.array([gx, gy])
            
            # 거리 계산 (Euclidean)
            dists = np.linalg.norm(coords - grid_point, axis=1)
            nearest_idx_in_arr = np.argmin(dists)
            
            real_idx = original_indices[nearest_idx_in_arr]
            real_pt = tuple(coords[nearest_idx_in_arr])
            
            sampled_candidates.append({
                'idx': int(real_idx),
                'pt': real_pt
            })
    
    # 중복 제거
    unique_samples = []
    seen_indices = set()
    for cand in sampled_candidates:
        if cand['idx'] not in seen_indices:
            seen_indices.add(cand['idx'])
            unique_samples.append(cand)
            
    return unique_samples


def project_points_to_screen(points_3d, vtk_renderer, render_window):
    """
    3D world coordinates → 2D screen coordinates (with original indices)
    
    Args:
        points_3d: list/array of (x, y, z) world coordinates
        vtk_renderer: VTK renderer object
        render_window: VTK render window
    
    Returns:
        list of (original_index, (x_pixel, y_pixel))
    """
    import vtk
    
    if points_3d is None or len(points_3d) == 0:
        return []
    
    width, height = render_window.GetSize()
    
    coordinate = vtk.vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    
    results = []
    
    for i, point_3d in enumerate(points_3d):
        coordinate.SetValue(float(point_3d[0]), float(point_3d[1]), float(point_3d[2]))
        display_pos = coordinate.GetComputedDisplayValue(vtk_renderer)
        
        x_pixel = int(display_pos[0])
        y_pixel = int(height - display_pos[1] - 1)  # VTK → Image 좌표 변환
        
        # 화면 범위 체크
        if 0 <= x_pixel < width and 0 <= y_pixel < height:
            results.append((i, (x_pixel, y_pixel)))
    
    return results