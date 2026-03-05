"""
TF Tent 경계 기반 target_range 계산 헬퍼 함수
"""
import numpy as np
from collections import Counter


def extract_tents_from_nodes(nodes):
    """TF nodes에서 tent 구조 추출"""
    tents = []
    n = len(nodes)
    i = 0
    
    while i < n:
        if nodes[i][4] > 0:  # opacity > 0 → peak node
            peak_node = nodes[i]
            left_node = nodes[i-1] if i > 0 else nodes[i]
            right_node = nodes[i+1] if i < n-1 else nodes[i]
            
            tents.append({
                'left': left_node[0],
                'mu': peak_node[0],
                'right': right_node[0],
                'peak': peak_node[4]
            })
            i += 2
        else:
            i += 1
    
    return tents


def find_target_range_from_tents(picked_intensities, tf_nodes, volume_range=None):
    """
    Picked intensities가 속하는 TF tent의 경계를 target_range로 반환
    
    Args:
        picked_intensities: array of intensity values (실제 볼륨 값)
        tf_nodes: List of [intensity, R, G, B, A] (intensity는 0~1 normalized)
        volume_range: tuple (min, max) - 볼륨의 실제 intensity 범위
    
    Returns:
        tuple: (min_intensity, max_intensity) - 실제 볼륨 스케일
    """
    if volume_range is None:
        volume_range = (0, 255)
    
    vol_min, vol_max = volume_range
    vol_span = vol_max - vol_min
    
    tents = extract_tents_from_nodes(tf_nodes)
    
    if not tents:
        print("⚠️ No tents found, falling back to intensity min/max")
        return (np.min(picked_intensities), np.max(picked_intensities))
    
    print(f"\n🎯 Finding target range from {len(tents)} tent(s):")
    for i, tent in enumerate(tents):
        left_real = tent['left'] * vol_span + vol_min
        right_real = tent['right'] * vol_span + vol_min
        print(f"   Tent {i}: [{left_real:.4f}, {right_real:.4f}]")
    
    # 각 intensity가 어느 tent에 속하는지 카운트
    tent_counts = Counter()
    for intensity in picked_intensities:
        normalized = (intensity - vol_min) / vol_span
        for i, tent in enumerate(tents):
            if tent['left'] <= normalized <= tent['right']:
                tent_counts[i] += 1
                break
    
    print(f"📊 Points per tent: {dict(tent_counts)}")
    
    if not tent_counts:
        print("⚠️ No points in any tent, falling back to intensity min/max")
        return (np.min(picked_intensities), np.max(picked_intensities))
    
    # 가장 많은 포인트가 속한 tent 선택
    dominant_tent_idx = tent_counts.most_common(1)[0][0]
    dominant_tent = tents[dominant_tent_idx]
    
    target_min = dominant_tent['left'] * vol_span + vol_min
    target_max = dominant_tent['right'] * vol_span + vol_min
    
    print(f"✅ Selected Tent {dominant_tent_idx}: [{target_min:.4f}, {target_max:.4f}]")
    
    return (target_min, target_max)