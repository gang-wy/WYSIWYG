"""
Differentiable Volume Renderer using PyTorch

Key Design Principles:
1. Fully vectorized - NO for loops over pixels
2. Uses grid_sample for differentiable interpolation
3. Front-to-back compositing for physically accurate rendering
4. Compatible with VTK camera parameters

Architecture:
  Camera → Ray Generation (vectorized) → Sampling (vectorized) 
  → Interpolation (grid_sample) → TF Application → Compositing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DifferentiableVolumeRenderer:
    """
    PyTorch-based differentiable volume renderer
    
    All operations are vectorized for GPU acceleration
    Supports gradient computation through entire rendering pipeline
    """
    
    def __init__(self, volume_data, spacing, resolution=256, num_samples=128, device='cuda'):
        """
        Args:
            volume_data: numpy array of shape (X, Y, Z)
            spacing: tuple (sx, sy, sz) voxel spacing in world units
            resolution: output image resolution (width=height)
            num_samples: number of samples along each ray
            device: 'cuda' or 'cpu'
        """
        self.device = device
        self.resolution = resolution
        self.num_samples = num_samples
        
        # Convert volume to PyTorch tensor (normalized to [-1, 1] for grid_sample)
        # Shape: (1, 1, D, H, W) for grid_sample convention
        volume_tensor = torch.tensor(volume_data, dtype=torch.float32, device=device)
        
        # Normalize volume to [0, 1] for TF application
        self.volume_min = volume_tensor.min()
        self.volume_max = volume_tensor.max()
        self.actual_volume_range = (self.volume_min.item(), self.volume_max.item())  # ⭐ 추가!

        volume_normalized = (volume_tensor - self.volume_min) / (self.volume_max - self.volume_min + 1e-8)
        
        # Reshape for grid_sample: (Batch, Channel, Depth, Height, Width)
        self.volume = volume_normalized.unsqueeze(0).unsqueeze(0)
        
        # Volume properties
        self.dims = torch.tensor(volume_data.shape, dtype=torch.float32, device=device)
        self.spacing = torch.tensor(spacing, dtype=torch.float32, device=device)
        self.volume_bounds = self.dims * self.spacing
        
        # Camera parameters (initialized to None, set via set_camera)
        self.camera_position = None
        self.camera_forward = None
        self.camera_right = None
        self.camera_up = None
        self.camera_fov = 30.0  # Field of view in degrees
        
        # ⭐ Add camera matrices (for projection)
        self.view_matrix = torch.eye(4, device=device)
        self.proj_matrix = torch.eye(4, device=device)

        print(f"✅ DifferentiableVolumeRenderer initialized:")
        print(f"   Volume shape: {volume_data.shape}")
        print(f"   Actual intensity range: [{self.actual_volume_range[0]:.1f}, {self.actual_volume_range[1]:.1f}]")  # ⭐ 추가!

        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Samples per ray: {num_samples}")
        print(f"   Device: {device}")
    
    def set_camera(self, camera_info):
        """
        Set camera from VTK camera parameters
        
        Args:
            camera_info: dict with keys:
                - 'position': (x, y, z)
                - 'focal_point': (x, y, z)
                - 'view_up': (x, y, z)
                - 'view_angle': float (optional)
        """
        position = torch.tensor(camera_info['position'], dtype=torch.float32, device=self.device)
        focal_point = torch.tensor(camera_info['focal_point'], dtype=torch.float32, device=self.device)
        view_up = torch.tensor(camera_info['view_up'], dtype=torch.float32, device=self.device)
        
        # Compute camera coordinate system
        forward = F.normalize(focal_point - position, dim=0)
        right = F.normalize(torch.cross(forward, view_up, dim=0), dim=0)
        up = torch.cross(right, forward, dim=0)
        
        self.camera_position = position
        self.camera_forward = forward
        self.camera_right = right
        self.camera_up = up
        
        if 'view_angle' in camera_info:
            self.camera_fov = camera_info['view_angle']

        # ⭐⭐⭐ [추가] View Matrix 계산 ⭐⭐⭐
        self.view_matrix = self._compute_view_matrix(position, focal_point, view_up)
        
        # ⭐⭐⭐ [추가] Projection Matrix 계산 ⭐⭐⭐
        aspect_ratio = 1.0  # Square viewport
        near = 0.1
        far = 1000.0

        self.proj_matrix = self._compute_projection_matrix(self.camera_fov, aspect_ratio, near, far)
        
        print(f"📷 Camera set: pos={position.cpu().numpy()}, fov={self.camera_fov:.1f}°")
    
    # ⭐⭐⭐ [새로운 메서드 1] View Matrix 계산 ⭐⭐⭐
    def _compute_view_matrix(self, eye, target, up):
        """
        Compute view matrix (world to camera space)
        
        Args:
            eye: camera position (3,)
            target: look-at target (3,)
            up: up vector (3,)
            
        Returns:
            view_matrix: (4, 4) torch tensor
        """
        # Forward vector (camera looks toward -Z in OpenGL convention)
        forward = F.normalize(target - eye, dim=0)
        
        # Right vector
        right = F.normalize(torch.cross(forward, up, dim=0), dim=0)
        
        # Recalculate up (orthogonal)
        up_ortho = torch.cross(right, forward, dim=0)
        
        # Build rotation part (3x3)
        rotation = torch.stack([right, up_ortho, -forward], dim=0)  # (3, 3)
        
        # Build translation part
        translation = -torch.matmul(rotation, eye)
        
        # Combine into 4x4 matrix
        view_matrix = torch.eye(4, device=self.device)
        view_matrix[:3, :3] = rotation
        view_matrix[:3, 3] = translation
        
        return view_matrix
    
    # ⭐⭐⭐ [새로운 메서드 2] Projection Matrix 계산 ⭐⭐⭐
    def _compute_projection_matrix(self, fov_degrees, aspect_ratio, near, far):
        """
        Compute perspective projection matrix
        
        Args:
            fov_degrees: vertical field of view in degrees
            aspect_ratio: width / height
            near: near clipping plane
            far: far clipping plane
            
        Returns:
            proj_matrix: (4, 4) torch tensor
        """
        fov_rad = torch.deg2rad(torch.tensor(fov_degrees, dtype=torch.float32, device=self.device))
        f = 1.0 / torch.tan(fov_rad / 2.0)
        
        proj_matrix = torch.zeros(4, 4, device=self.device)
        proj_matrix[0, 0] = f / aspect_ratio
        proj_matrix[1, 1] = f
        proj_matrix[2, 2] = (far + near) / (near - far)
        proj_matrix[2, 3] = (2.0 * far * near) / (near - far)
        proj_matrix[3, 2] = -1.0
        
        return proj_matrix

    def _generate_rays(self):
        """
        Generate all rays for the image (vectorized)
        
        Returns:
            ray_origins: (N, 3) tensor where N = resolution^2
            ray_directions: (N, 3) tensor, normalized
        """
        if self.camera_position is None:
            raise ValueError("Camera not set! Call set_camera() first.")
        
        # Generate pixel grid
        half_res = self.resolution / 2.0
        aspect_ratio = 1.0  # Square image
        
        # Compute image plane size based on FOV
        fov_rad = torch.deg2rad(torch.tensor(self.camera_fov, device=self.device))
        focal_length = half_res / torch.tan(fov_rad / 2.0)
        
        # Pixel coordinates (centered at origin)
        x = torch.linspace(-half_res, half_res - 1, self.resolution, device=self.device)
        y = torch.linspace(half_res - 1, -half_res, self.resolution, device=self.device)
        
        # Create grid: (H, W, 2)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Flatten to (N, 2)
        pixel_coords = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        
        # Convert pixel coords to world rays
        # Ray direction in camera space: (x/focal, y/focal, 1)
        cam_space_dirs = torch.stack([
            pixel_coords[:, 0] / focal_length,
            pixel_coords[:, 1] / focal_length,
            torch.ones(pixel_coords.shape[0], device=self.device)
        ], dim=1)
        
        # Transform to world space using camera basis
        ray_directions = (
            cam_space_dirs[:, 0:1] * self.camera_right +
            cam_space_dirs[:, 1:2] * self.camera_up +
            cam_space_dirs[:, 2:3] * self.camera_forward
        )
        
        # Normalize directions
        ray_directions = F.normalize(ray_directions, dim=1)
        
        # All rays originate from camera position
        ray_origins = self.camera_position.unsqueeze(0).expand(ray_directions.shape[0], -1)
        
        return ray_origins, ray_directions
    
    def _intersect_volume(self, ray_origins, ray_directions):
        """
        Compute ray-volume intersection (vectorized)
        
        Args:
            ray_origins: (N, 3)
            ray_directions: (N, 3)
        
        Returns:
            t_min: (N,) entry points
            t_max: (N,) exit points
            valid_mask: (N,) bool mask for rays that hit the volume
        """
        # Volume bounds: [0, 0, 0] to [dims * spacing]
        bounds_min = torch.zeros(3, device=self.device)
        bounds_max = self.volume_bounds
        
        # Slab method for ray-box intersection
        inv_dir = 1.0 / (ray_directions + 1e-8)
        
        t1 = (bounds_min - ray_origins) * inv_dir
        t2 = (bounds_max - ray_origins) * inv_dir
        
        t_min = torch.max(torch.minimum(t1, t2), dim=1)[0]
        t_max = torch.min(torch.maximum(t1, t2), dim=1)[0]
        
        # Valid rays: t_max > t_min and t_max > 0
        valid_mask = (t_max > t_min) & (t_max > 0)
        
        # Clamp t_min to 0 (don't start behind camera)
        t_min = torch.clamp(t_min, min=0.0)
        
        return t_min, t_max, valid_mask
    
    def _sample_along_rays(self, ray_origins, ray_directions, t_min, t_max):
        """
        Sample points along rays (vectorized)
        
        Args:
            ray_origins: (N, 3)
            ray_directions: (N, 3)
            t_min: (N,) entry distances
            t_max: (N,) exit distances
        
        Returns:
            sample_points: (N, S, 3) where S = num_samples
            sample_t: (N, S) distances along ray
        """
        # Generate sample distances: (N, S)
        t_vals = torch.linspace(0, 1, self.num_samples, device=self.device)
        sample_t = t_min.unsqueeze(1) + (t_max - t_min).unsqueeze(1) * t_vals.unsqueeze(0)
        
        # Compute 3D sample points: (N, S, 3)
        sample_points = ray_origins.unsqueeze(1) + ray_directions.unsqueeze(1) * sample_t.unsqueeze(2)
        
        return sample_points, sample_t
    
    def _interpolate_volume(self, sample_points):
        """
        Interpolate volume at sample points using grid_sample (differentiable!)
        
        Args:
            sample_points: (N, S, 3) in world coordinates
        
        Returns:
            intensities: (N, S) normalized intensity values [0, 1]
        """
        # Convert world coords to normalized grid coords [-1, 1] for grid_sample
        # grid_sample expects: (x, y, z) → (W, H, D) convention
        # Our volume is (D, H, W) so we need to swap axes
        
        # Normalize to [0, 1]
        normalized_coords = sample_points / self.volume_bounds
        
        # Convert to [-1, 1] for grid_sample
        grid_coords = normalized_coords * 2.0 - 1.0
        
        # Swap axes: (x, y, z) → (z, y, x) for (D, H, W) volume
        grid_coords = torch.stack([
            grid_coords[..., 2],  # z
            grid_coords[..., 1],  # y
            grid_coords[..., 0]   # x
        ], dim=-1)
        
        # Reshape for grid_sample: (1, N*S, 1, 1, 3)
        N, S = sample_points.shape[0], sample_points.shape[1]
        grid_coords_flat = grid_coords.view(1, N * S, 1, 1, 3)
        
        # Interpolate: (1, 1, N*S, 1, 1)
        sampled = F.grid_sample(
            self.volume,
            grid_coords_flat,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )
        
        # Reshape back: (N, S)
        intensities = sampled.view(N, S)
        
        return intensities
    
    def _front_to_back_composite(self, colors, opacities, sample_distances):
        """
        Front-to-back compositing (vectorized)
        
        Args:
            colors: (N, S, 3) RGB values
            opacities: (N, S) opacity values
            sample_distances: (N, S) distances between samples
        
        Returns:
            final_colors: (N, 3) accumulated RGB
            final_alphas: (N,) accumulated alpha
        """
        # Compute step size (distance between consecutive samples)
        step_sizes = sample_distances[:, 1:] - sample_distances[:, :-1]
        step_sizes = torch.cat([
            step_sizes,
            step_sizes[:, -1:]  # Duplicate last step
        ], dim=1)
        
        # Opacity correction for step size
        # alpha_corrected = 1 - (1 - alpha)^step_size
        corrected_opacities = 1.0 - torch.pow(1.0 - opacities + 1e-8, step_sizes)
        
        # Compute transmittance (product of (1 - alpha) up to each point)
        # transmittance[i] = prod((1 - alpha[j]) for j < i)
        one_minus_alpha = 1.0 - corrected_opacities
        transmittance = torch.cumprod(
            torch.cat([torch.ones(colors.shape[0], 1, device=self.device), one_minus_alpha[:, :-1]], dim=1),
            dim=1
        )
        
        # Weight of each sample
        weights = transmittance * corrected_opacities
        
        # Accumulate color
        final_colors = torch.sum(weights.unsqueeze(2) * colors, dim=1)
        
        # Accumulate alpha
        final_alphas = torch.sum(weights, dim=1)
        
        return final_colors, final_alphas
    
    def render(self, tf_module, background_color=(1.0, 1.0, 1.0)):
        """
        Render the volume (fully differentiable)
        
        Args:
            tf_module: DifferentiableTF module
            background_color: (R, G, B) background color
        
        Returns:
            image: (H, W, 3) rendered image tensor
        """
        if self.camera_position is None:
            raise ValueError("Camera not set! Call set_camera() first.")
        
        # 1. Generate all rays (vectorized)
        ray_origins, ray_directions = self._generate_rays()
        N = ray_origins.shape[0]
        
        # 2. Ray-volume intersection
        t_min, t_max, valid_mask = self._intersect_volume(ray_origins, ray_directions)
        
        # 3. Sample along rays
        sample_points, sample_t = self._sample_along_rays(ray_origins, ray_directions, t_min, t_max)
        
        # 4. Interpolate volume (differentiable!)
        intensities = self._interpolate_volume(sample_points)
        
        # 5. Apply transfer function (differentiable!)
        rgba = tf_module(intensities)  # (N, S, 4)
        colors = rgba[..., :3]
        opacities = rgba[..., 3]
        
        # 6. Compositing
        final_colors, final_alphas = self._front_to_back_composite(colors, opacities, sample_t)
        
        # 7. Apply background
        bg_color = torch.tensor(background_color, dtype=torch.float32, device=self.device)
        final_colors = final_colors + (1.0 - final_alphas.unsqueeze(1)) * bg_color
        
        # 8. Handle invalid rays (outside volume)
        final_colors[~valid_mask] = bg_color
        
        # 9. Reshape to image
        image = final_colors.view(self.resolution, self.resolution, 3)
        
        # Clamp to valid range
        image = torch.clamp(image, 0.0, 1.0)
        
        return image
    
    def render_to_numpy(self, tf_module, background_color=(0.0, 0.0, 0.0)):
        """
        Render and convert to numpy array (for saving/display)
        
        Returns:
            image: (H, W, 3) numpy array in [0, 255]
        """
        with torch.no_grad():
            image_tensor = self.render(tf_module, background_color)
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
        return image_np

    def check_visibility(self, point_3d, tf_module, opacity_threshold=0.95, 
                            initial_ray_direction=None, point_intensity=None,
                            target_range=None, clipping_ranges=None, return_debug=False):
            """
            PyTorch 기반 differentiable visibility check (Target Range Masking 적용)
            
            Args:
                point_3d: (3,) torch.Tensor - world coordinates of target point
                tf_module: DifferentiableTF - learnable transfer function
                opacity_threshold: float - visibility threshold (default: 0.95)
                initial_ray_direction: (Not Used in this logic, but kept for interface compatibility)
                point_intensity: (Not Used)
                target_range: tuple (min, max) - Normalized [0, 1] range to skip (Self-occlusion ignore)
                clipping_ranges: (optional) Clipping ranges for volume rendering
                return_debug: bool - if True, return debug data for visualization
            Returns:
                accumulated_opacity: torch.Tensor (scalar)
                is_visible: bool
                debug_data: dict (only if return_debug=True)
            """
            if self.camera_position is None:
                raise ValueError("Camera not set! Call set_camera() first.")
            
            # 디버그용 빈 데이터 (early return 시 사용)
            empty_debug = {
                't_values': np.array([]),
                'intensities': np.array([]),
                'raw_opacities': np.array([]),
                'effective_opacities': np.array([]),
                'cumulative_opacity': np.array([]),
                'is_target_tissue': np.array([]),
                'is_clipped': np.array([]),
                'target_range': target_range,
                'opacity_threshold': opacity_threshold,
            }
            
            # 포인트 자체가 clipping 범위 밖이면 즉시 occluded 처리
            # if clipping_ranges is not None:
            #     if self._is_point_clipped(point_3d, clipping_ranges):
            #         if return_debug:
            #             empty_debug['early_exit_reason'] = 'point_clipped'
            #             return torch.tensor(1.0, device=self.device, requires_grad=True), False, empty_debug
            #         return torch.tensor(1.0, device=self.device, requires_grad=True), False


            # 1. Ray setup (Camera -> point)
            ray_origin = self.camera_position
            ray_target = point_3d 
            ray_dir = ray_target - ray_origin
            ray_length = torch.norm(ray_dir)
            
            if ray_length < 1e-6:
                if return_debug:
                    empty_debug['early_exit_reason'] = 'zero_ray_length'
                    return torch.tensor(0.0, device=self.device, requires_grad=True), True, empty_debug
                return torch.tensor(0.0, device=self.device, requires_grad=True), True
            
            ray_dir = ray_dir / ray_length
            
            # 2. Ray sampling setup
            step_size = min(self.spacing).item() * 0.5
            num_steps = max(1, int((ray_length.item() - step_size) / step_size))
            
            if num_steps <= 1:
                if return_debug:
                    empty_debug['early_exit_reason'] = 'insufficient_steps'
                    return torch.tensor(0.0, device=self.device, requires_grad=True), True, empty_debug
                return torch.tensor(0.0, device=self.device, requires_grad=True), True
            
            # 3. Sampling points generation
            t_values = torch.linspace(step_size, ray_length.item() - step_size, num_steps, device=self.device)
            sample_positions = ray_origin.unsqueeze(0) + ray_dir.unsqueeze(0) * t_values.unsqueeze(1)

            # 4. Normalize coordinates for grid_sample
            normalized_coords = sample_positions / self.volume_bounds
            grid_coords = normalized_coords * 2.0 - 1.0
            grid_coords = grid_coords.view(1, num_steps, 1, 1, 3)
            grid_coords = torch.stack([
                grid_coords[..., 2],
                grid_coords[..., 1],
                grid_coords[..., 0]
            ], dim=-1)
            
            # 5. Volume interpolation (Get Intensities)
            sampled = F.grid_sample(
                self.volume,
                grid_coords,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            
            intensities = sampled.view(num_steps)  # Normalized intensity [0, 1]
            
            # 6. Apply Transfer Function (Get Opacities)
            opacities = tf_module.get_opacity_only(intensities)
            colors = tf_module.get_color_only(intensities)  # (num_steps, 3) RGB

            
            # 디버그용: raw opacities 저장
            raw_opacities = opacities.clone()
            
            # Target Range Skip Logic
            effective_opacities = opacities.clone()
            is_target_tissue = torch.zeros(num_steps, dtype=torch.bool, device=self.device)
            
            if target_range is not None:
                t_min, t_max = target_range
                is_target_tissue = (intensities >= t_min) & (intensities <= t_max)
                effective_opacities = torch.where(
                    is_target_tissue, 
                    torch.tensor(0.0, device=self.device), 
                    opacities
                )

            # Clipping mask
            is_clipped = torch.zeros(num_steps, dtype=torch.bool, device=self.device)
            if clipping_ranges is not None:
                is_clipped = self._get_clipping_mask(sample_positions, clipping_ranges)
                effective_opacities = torch.where(
                    is_clipped,
                    torch.tensor(0.0, device=self.device),
                    effective_opacities
                )    

            # 7. Front-to-back compositing
            one_minus_alpha = torch.clamp(1.0 - effective_opacities, min=1e-6, max=1.0)
            transmittance = torch.cumprod(one_minus_alpha, dim=0)
            cumulative_opacity = 1.0 - transmittance  # 각 스텝에서의 누적 opacity
            accumulated_opacity = cumulative_opacity[-1]
            
            is_visible = accumulated_opacity.item() < opacity_threshold


            # Optical depth (normalized)
            optical_depth_threshold = -torch.log(torch.tensor(1.0 - opacity_threshold + 1e-8, device=self.device))
            optical_depth = -torch.log(transmittance[-1] + 1e-8)
            normalized_depth = optical_depth / optical_depth_threshold



            if return_debug:
                debug_data = {
                    't_values': t_values.detach().cpu().numpy(),
                    'intensities': intensities.detach().cpu().numpy(),
                    'raw_opacities': raw_opacities.detach().cpu().numpy(),
                    'effective_opacities': effective_opacities.detach().cpu().numpy(),
                    'cumulative_opacity': cumulative_opacity.detach().cpu().numpy(),
                    'is_target_tissue': is_target_tissue.detach().cpu().numpy(),
                    'is_clipped': is_clipped.detach().cpu().numpy(),
                    'colors': colors.detach().cpu().numpy(),  # NEW: (num_steps, 3) RGB
                    'target_range': target_range,
                    'opacity_threshold': opacity_threshold,
                    'ray_length': ray_length.item(),
                    'num_steps': num_steps,
                }
                return normalized_depth, is_visible, debug_data
            
            return normalized_depth, is_visible

    def check_visibility_batch(self, points_3d, tf_module, opacity_threshold=0.95):
        """
        Batch version of visibility check (for multiple points)
        
        Args:
            points_3d: (N, 3) torch.Tensor - world coordinates
            tf_module: DifferentiableTF
            opacity_threshold: float
            
        Returns:
            normalized_depths: (N,) torch.Tensor - differentiable
            visibility_mask: (N,) bool tensor
        """
        N = points_3d.shape[0]
        
        accumulated_opacities = []
        visibility_mask = []
        
        for i in range(N):
            acc_opacity, is_visible = self.check_visibility(
                points_3d[i], tf_module, opacity_threshold
            )
            accumulated_opacities.append(acc_opacity)
            visibility_mask.append(is_visible)
        
        # Stack into tensors
        accumulated_opacities = torch.stack(accumulated_opacities)  # (N,)
        visibility_mask = torch.tensor(visibility_mask, device=self.device)  # (N,)
        
        return accumulated_opacities, visibility_mask


    def _is_point_clipped(self, point_3d, clipping_ranges):
        """
        단일 포인트가 clipping 범위 밖인지 체크
        
        Args:
            point_3d: (3,) torch.Tensor - world coordinates
            clipping_ranges: dict {'x': [min, max], 'y': [min, max], 'z': [min, max]}
        
        Returns:
            bool: True if clipped (범위 밖)
        """
        # 정규화된 좌표로 변환 (0~1)
        normalized = point_3d / self.volume_bounds
        
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            if axis in clipping_ranges:
                min_val, max_val = clipping_ranges[axis]
                coord = normalized[i].item()
                if coord < min_val or coord > max_val:
                    return True  # 범위 밖 = clipped
        
        return False  # 범위 안 = visible


    def _get_clipping_mask(self, sample_positions, clipping_ranges):
        """
        샘플 포인트들의 clipping 마스크 생성 (배치 처리)
        
        Args:
            sample_positions: (N, 3) torch.Tensor - world coordinates
            clipping_ranges: dict {'x': [min, max], 'y': [min, max], 'z': [min, max]}
        
        Returns:
            mask: (N,) bool tensor - True if clipped (범위 밖)
        """
        # 정규화된 좌표로 변환 (0~1)
        normalized = sample_positions / self.volume_bounds  # (N, 3)
        
        # 초기 마스크: 모두 False (범위 안)
        clipped_mask = torch.zeros(sample_positions.shape[0], dtype=torch.bool, device=self.device)
        
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            if axis in clipping_ranges:
                min_val, max_val = clipping_ranges[axis]
                # 범위 밖이면 True
                out_of_range = (normalized[:, i] < min_val) | (normalized[:, i] > max_val)
                clipped_mask = clipped_mask | out_of_range
        
        return clipped_mask