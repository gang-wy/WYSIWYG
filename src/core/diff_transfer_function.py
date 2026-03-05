"""
Differentiable Transfer Function for PyTorch-based Optimization

Key Design:
- Dynamic LUT size based on volume data range (8-bit: 256, CT: 4096, etc.)
- Binning support to reduce learnable parameters
- Avoids gradient discontinuity from if/else statements
- Can convert back to VTK-compatible node format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableTF(nn.Module):
    """
    Learnable Transfer Function with Dynamic Range and Binning
    """
    
    def __init__(self, initial_nodes, volume_range=None, num_bins=256, device='cuda'):
        """
        Args:
            initial_nodes: List of [intensity, R, G, B, A]
            volume_range: tuple (min_intensity, max_intensity)
            num_bins: Number of learnable bins (default: 256)
            device: 'cuda' or 'cpu'
        """
        super().__init__()
        self.device = device
        self.num_bins = num_bins
        
        # Store volume intensity range
        if volume_range is None:
            self.volume_min = 0.0
            self.volume_max = 1.0
        else:
            self.volume_min = float(volume_range[0])
            self.volume_max = float(volume_range[1])
        
        self.volume_range = self.volume_max - self.volume_min
        
        # Convert initial nodes to num_bins-element LUT
        opacity_lut_np, color_lut_np = self._nodes_to_lut(initial_nodes, num_bins)        
        
        # [수정 1] 값이 있는 인덱스만 추출 (Sparse Optimization)
        active_indices = np.where(opacity_lut_np > 1e-6)[0]
        
        # [수정 2] Buffer 등록 (학습되지 않는 고정 값들)
        self.register_buffer(
            'active_indices', 
            torch.tensor(active_indices, dtype=torch.long, device=device)
        )
        self.register_buffer(
            'base_opacity_lut', 
            torch.zeros(num_bins, dtype=torch.float32, device=device)
        )

        # [수정 3] 진짜 학습할 파라미터만 등록
        self.learnable_opacity = nn.Parameter(
            torch.tensor(opacity_lut_np[active_indices], dtype=torch.float32, device=device)
        )
        
        # Color is fixed (not learnable)
        self.register_buffer(
            'color_lut',
            torch.tensor(color_lut_np, dtype=torch.float32, device=device)
        )
        
        # Store original nodes for reference
        self.original_nodes = initial_nodes
        
        print(f"✅ DifferentiableTF initialized (Sparse Mode):")
        print(f"   Volume range: [{self.volume_min:.1f}, {self.volume_max:.1f}]")
        print(f"   Num bins: {self.num_bins}")
        print(f"   Active Learnable Bins: {len(active_indices)}")

    # [수정 4] 외부 코드 호환성을 위한 Property 추가 (핵심!)
    @property
    def opacity_lut(self):
        """
        외부에서 tf.opacity_lut를 호출할 때 자동으로 실행됩니다.
        학습 중인 파라미터(active)를 전체 LUT(base)에 끼워 넣어서 리턴합니다.
        """
        # 1. 0으로 채워진 베이스 복사
        full_lut = self.base_opacity_lut.clone()
        # 2. 학습된 값들을 제자리에 끼워 넣기
        full_lut[self.active_indices] = self.learnable_opacity
        return full_lut
    
    def get_full_opacity_lut(self):
        """내부 편의용 함수 (property와 동일)"""
        return self.opacity_lut


    def _nodes_to_lut(self, nodes, num_bins):
        """Convert VTK-style nodes to num_bins-element LUT"""
        if not nodes or len(nodes) < 2:
            opacity_lut = np.linspace(0, 1, num_bins)
            color_lut = np.ones((num_bins, 3))
            return opacity_lut, color_lut
        
        sorted_nodes = sorted(nodes, key=lambda x: x[0])
        
        opacity_lut = np.zeros(num_bins)
        color_lut = np.zeros((num_bins, 3))
        
        # [추가] Node 위치 → bin index 매핑 (정확히 스냅)
        node_bin_indices = []
        for node in sorted_nodes:
            bin_idx = int(round(node[0] * (num_bins - 1)))
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            node_bin_indices.append(bin_idx)
        
        # [추가] Node 위치의 bin에는 정확한 값 먼저 넣기
        for node, bin_idx in zip(sorted_nodes, node_bin_indices):
            opacity_lut[bin_idx] = node[4]
            color_lut[bin_idx] = node[1:4]
        
        # 나머지 bin은 보간으로 채우기
        for i in range(num_bins):
            # 이미 node가 있는 bin이면 스킵
            if i in node_bin_indices:
                continue
                
            normalized_intensity = i / (num_bins - 1)
            
            if normalized_intensity <= sorted_nodes[0][0]:
                opacity_lut[i] = sorted_nodes[0][4]
                color_lut[i] = sorted_nodes[0][1:4]
            elif normalized_intensity >= sorted_nodes[-1][0]:
                opacity_lut[i] = sorted_nodes[-1][4]
                color_lut[i] = sorted_nodes[-1][1:4]
            else:
                for j in range(len(sorted_nodes) - 1):
                    node_a = sorted_nodes[j]
                    node_b = sorted_nodes[j + 1]
                    
                    if node_a[0] <= normalized_intensity <= node_b[0]:
                        t = (normalized_intensity - node_a[0]) / (node_b[0] - node_a[0])
                        
                        opacity_lut[i] = node_a[4] * (1 - t) + node_b[4] * t
                        color_lut[i, 0] = node_a[1] * (1 - t) + node_b[1] * t 
                        color_lut[i, 1] = node_a[2] * (1 - t) + node_b[2] * t 
                        color_lut[i, 2] = node_a[3] * (1 - t) + node_b[3] * t 
                        break
        
        return opacity_lut, color_lut
    
    
    def forward(self, intensities):
        """Apply transfer function to intensity values"""
        # [수정 5] Property를 통해 전체 LUT 가져오기
        current_lut = self.opacity_lut

        # Normalize intensities to [0, 1]
        normalized = (intensities - self.volume_min) / (self.volume_range + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        # Map to bin indices [0, num_bins-1]
        float_indices = normalized * (self.num_bins - 1)
        
        # Floor and ceil indices
        idx_low = torch.floor(float_indices).long()
        idx_high = torch.clamp(idx_low + 1, max=self.num_bins - 1)
        idx_low = torch.clamp(idx_low, 0, self.num_bins - 1)
        
        t = float_indices - idx_low.float() 
        
        # Linear interpolation
        opacities = current_lut[idx_low] * (1 - t) + current_lut[idx_high] * t        

        colors_low = self.color_lut[idx_low]
        colors_high = self.color_lut[idx_high]
        colors = colors_low * (1 - t).unsqueeze(-1) + colors_high * t.unsqueeze(-1)
        
        rgba = torch.cat([colors, opacities.unsqueeze(-1)], dim=-1)
        return rgba
    
    def get_opacity_only(self, intensities):
        """Get only opacity values"""
        normalized = (intensities - self.volume_min) / (self.volume_range + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        float_indices = normalized * (self.num_bins - 1)
        idx_low = torch.floor(float_indices).long()
        idx_high = torch.clamp(idx_low + 1, max=self.num_bins - 1)
        idx_low = torch.clamp(idx_low, 0, self.num_bins - 1)
        
        t = float_indices - idx_low.float()
        
        # [수정 6] Property 사용
        current_lut = self.opacity_lut
        opacities = current_lut[idx_low] * (1 - t) + current_lut[idx_high] * t
        return opacities
    
    def get_color_only(self, intensities):
        """Get only color values (RGB)"""
        normalized = (intensities - self.volume_min) / (self.volume_range + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        float_indices = normalized * (self.num_bins - 1)
        idx_low = torch.floor(float_indices).long()
        idx_high = torch.clamp(idx_low + 1, max=self.num_bins - 1)
        idx_low = torch.clamp(idx_low, 0, self.num_bins - 1)
        
        t = float_indices - idx_low.float()
        
        colors_low = self.color_lut[idx_low]
        colors_high = self.color_lut[idx_high]
        colors = colors_low * (1 - t).unsqueeze(-1) + colors_high * t.unsqueeze(-1)
        
        return colors  # (N, 3) RGB

    def to_nodes_direct(self):
        """LUT를 그대로 num_bins개 nodes로 변환"""
        # [수정 8] Property 사용
        opacity_np = self.opacity_lut.detach().cpu().numpy()
        color_np = self.color_lut.detach().cpu().numpy()
        
        nodes = []
        for i in range(self.num_bins):
            normalized_intensity = i / (self.num_bins - 1)
            nodes.append([
                normalized_intensity,
                float(color_np[i, 0]),
                float(color_np[i, 1]),
                float(color_np[i, 2]),
                float(opacity_np[i])
            ])
        return nodes

    def get_lut_arrays(self):
        """Get LUT as numpy arrays for visualization"""
        # [수정 9] Property 사용
        return (
            self.opacity_lut.detach().cpu().numpy(),
            self.color_lut.detach().cpu().numpy()
        )
    
    def set_opacity_constraint(self, min_opacity=0.0, max_opacity=1.0):
        """Apply constraints to opacity values"""
        with torch.no_grad():
            # [수정 10] 학습 중인 파라미터만 Clamp
            self.learnable_opacity.clamp_(min_opacity, max_opacity)    

    def get_num_parameters(self):
        return self.learnable_opacity.numel()