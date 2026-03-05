"""
Differentiable Optimizer for Transfer Function

Integrates:
- DifferentiableVolumeRenderer (PyTorch rendering)
- DifferentiableTF (learnable transfer function)
- Multiple optimization methods (Adam, Nelder-Mead)

Key Design:
- Adam: Optimizes binned LUT (64 bins)
- Nelder-Mead: Optimizes original TF node peaks (Tent structure)
"""

import os
import tempfile
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.core.diff_renderer import DifferentiableVolumeRenderer
    from src.core.diff_transfer_function import DifferentiableTF
    from src.core.utils.common import sample_grid_representative_points
except ImportError:
    from diff_renderer import DifferentiableVolumeRenderer
    from diff_transfer_function import DifferentiableTF
    try:
        from utils.common import sample_grid_representative_points
    except ImportError:
        def sample_grid_representative_points(projection_results, grid_size=4):
            if not projection_results:
                return []
            coords = np.array([p[1] for p in projection_results])
            original_indices = np.array([p[0] for p in projection_results])
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            x_grid = np.linspace(min_x, max_x, grid_size)
            y_grid = np.linspace(min_y, max_y, grid_size)
            sampled_candidates = []
            for gx in x_grid:
                for gy in y_grid:
                    grid_point = np.array([gx, gy])
                    dists = np.linalg.norm(coords - grid_point, axis=1)
                    nearest_idx_in_arr = np.argmin(dists)
                    real_idx = original_indices[nearest_idx_in_arr]
                    real_pt = tuple(coords[nearest_idx_in_arr])
                    sampled_candidates.append({'idx': int(real_idx), 'pt': real_pt})
            unique_samples = []
            seen_indices = set()
            for cand in sampled_candidates:
                if cand['idx'] not in seen_indices:
                    seen_indices.add(cand['idx'])
                    unique_samples.append(cand)
            return unique_samples


class DiffOptimizer:
    """
    PyTorch-based Transfer Function Optimizer
    
    Supports multiple optimization methods:
    - 'adam': Gradient-based, optimizes binned LUT
    - 'nelder-mead': Derivative-free, optimizes original TF node peaks
    """
        
    def __init__(self, volume_data, spacing, initial_nodes, camera_info,
                sam_wrapper, projection_points_3d, device='cuda',
                feature_analyzer=None, use_vtk_rendering=True, vtk_resolution=None,
                target_range=None, initial_ray_directions=None, clipping_ranges=None,
                color_reference=None, gt_mask=None,
                point_certainty_weights=None, sam_confidence=None):
        """
        Args:
            volume_data: numpy array (X, Y, Z)
            spacing: tuple (sx, sy, sz)
            initial_nodes: list of [intensity, R, G, B, A]
            camera_info: dict with camera parameters
            sam_wrapper: SAM model wrapper for mask prediction
            projection_points_3d: list/array of 3D world coordinates
            device: 'cuda' or 'cpu'
            point_certainty_weights: (N,) numpy array — sigmoid(logit) per 3D point, 1.0=certain, 0.5=boundary
            sam_confidence: float — global SAM mask confidence (mean certainty inside mask)
        """
        self.device = device
        self.sam_wrapper = sam_wrapper
        self.feature_analyzer = feature_analyzer
        self.use_vtk_rendering = use_vtk_rendering
        self.camera_info = camera_info
        self.target_range = target_range
        self.clipping_ranges = clipping_ranges
        self.initial_nodes = initial_nodes
        self.volume_data = volume_data
        self.spacing = spacing
        
        # Color reference setup
        self.color_reference = None
        self.gt_mask = None
        
        if color_reference is not None and gt_mask is not None:
            self.color_reference = {
                'mean': torch.tensor(color_reference['mean'], dtype=torch.float32, device=device),
                'std': torch.tensor(color_reference['std'], dtype=torch.float32, device=device),
                'source_resolution': color_reference['source_resolution']
            }
            self.gt_mask = torch.tensor(gt_mask, dtype=torch.bool, device=device)
            
            print(f"✅ Color reference loaded:")
            print(f"   Mean RGB: {color_reference['mean']}")
            print(f"   Std RGB: {color_reference['std']}")

        self._prev_loss = None  # for convergence detection

        # SAM certainty weights (precomputed, not updated during optimization)
        self.sam_confidence = sam_confidence if sam_confidence is not None else 1.0
        if point_certainty_weights is not None:
            self.point_certainty_weights = torch.tensor(
                point_certainty_weights, dtype=torch.float32, device=device
            )
            print(f"   SAM confidence (global): {self.sam_confidence:.3f}")
            print(f"   Point certainty weights: {len(point_certainty_weights)} points, "
                  f"mean={np.mean(point_certainty_weights):.3f}")
        else:
            self.point_certainty_weights = None

        if initial_ray_directions is not None:
            self.initial_ray_directions = np.array(initial_ray_directions)
        else:
            self.initial_ray_directions = None

        # VTK resolution
        if use_vtk_rendering:
            if vtk_resolution is None:
                raise ValueError("vtk_resolution is required when use_vtk_rendering=True")
            self.vtk_resolution = vtk_resolution
            print(f"🔍 VTK resolution: {vtk_resolution[0]}x{vtk_resolution[1]}")
        else:
            self.vtk_resolution = None

        # Store 3D points
        self.points_3d = torch.tensor(projection_points_3d, dtype=torch.float32, device=device)
        
        # Volume range for TF
        self.volume_range = (volume_data.min(), volume_data.max())
        
        # Create differentiable components (for Adam mode)
        self.tf = DifferentiableTF(
            initial_nodes, 
            volume_range=self.volume_range,
            num_bins=128, #64,
            device=device
        )

        self.renderer = DifferentiableVolumeRenderer(
            volume_data=volume_data,
            spacing=spacing,
            resolution=256,
            num_samples=128,
            device=device
        )
        self.renderer.set_camera(camera_info)
        
        # Extract Tent structure for Nelder-Mead mode
        self.tents = self._extract_tents(initial_nodes)
        print(f"📊 Extracted {len(self.tents)} Tents from TF nodes")
        
        # External image for VTK rendering
        self.external_image_path = None
        self.external_image_ready = False
        
        # Render request callback (for VTK sync)
        self.render_request_callback = None

        # Temp directory
        self.temp_dir = tempfile.mkdtemp(prefix='diffdvr_')
        
        # Statistics
        self.iteration = 0
        self.best_loss = float('inf')
        self.initial_intensities = None
        self.best_nodes = initial_nodes
        self.converged = False
        self.initial_opacity_lut = self.tf.opacity_lut.detach().clone()
        
        # Mask storage
        self.last_full_mask = None
        self.last_visible_mask = None
        self.last_visible_samples = []
        self.last_grid_samples = []
        self.last_visibility_results = {}

        # Adam optimizer (미리 생성) learning_rate 조정하는 곳
        self.optimizer = torch.optim.Adam(
            self.tf.parameters(),
            lr=0.01,
            betas=(0.9, 0.999)
        )

        print(f"✅ DiffOptimizer initialized:")
        print(f"   Device: {device}")
        print(f"   3D Points: {len(projection_points_3d)}")
        print(f"   Temp dir: {self.temp_dir}")

    def _extract_tents(self, nodes):
        """
        Extract Tent structures from TF nodes
        
        Returns:
            list of dicts: [{'mu': center, 'left': left_bound, 'right': right_bound, 
                            'r': R, 'g': G, 'b': B, 'peak': opacity}, ...]
        """
        tents = []
        n = len(nodes)
        i = 0
        
        while i < n:
            if nodes[i][4] > 0:  # Has opacity (peak)
                peak_node = nodes[i]
                left_node = nodes[i-1] if i > 0 else nodes[i]
                right_node = nodes[i+1] if i < n-1 else nodes[i]
                
                tents.append({
                    'mu': peak_node[0],
                    'left': left_node[0],
                    'right': right_node[0],
                    'r': peak_node[1],
                    'g': peak_node[2],
                    'b': peak_node[3],
                    'peak': peak_node[4]
                })
                i += 2  # Skip peak and right boundary
            else:
                i += 1
        
        return tents

    def _tents_to_nodes(self, peak_opacities):
        """
        Convert Tent peak opacities back to TF nodes
        
        Args:
            peak_opacities: list/array of opacity values for each tent
            
        Returns:
            list of [intensity, R, G, B, A] nodes
        """
        nodes = []
        
        for i, tent in enumerate(self.tents):
            opacity = float(np.clip(peak_opacities[i], 0.0, 1.0))
            
            # Left boundary (transparent)
            nodes.append([tent['left'], 0.0, 0.0, 0.0, 0.0])
            # Peak
            nodes.append([tent['mu'], tent['r'], tent['g'], tent['b'], opacity])
            # Right boundary (transparent)
            nodes.append([tent['right'], 0.0, 0.0, 0.0, 0.0])
        
        # Sort by intensity
        nodes.sort(key=lambda x: x[0])
        
        # Remove duplicates (same intensity)
        unique_nodes = []
        seen_intensities = set()
        for node in nodes:
            key = round(node[0], 6)
            if key not in seen_intensities:
                seen_intensities.add(key)
                unique_nodes.append(node)
        
        return unique_nodes

    def _create_tf_from_nodes(self, nodes):
        """Create a new DifferentiableTF from nodes"""
        return DifferentiableTF(
            nodes,
            volume_range=self.volume_range,
            num_bins=64,
            device=self.device
        )

    def set_external_image(self, image_path):
        """Set external VTK rendered image"""
        self.external_image_path = image_path
        self.external_image_ready = True
        print(f"📷 External VTK image set: {image_path}")

    def set_render_request_callback(self, callback):
        """Set callback for requesting VTK render"""
        self.render_request_callback = callback

    def _project_points_to_2d(self, points_3d):
        """Project 3D world coordinates to 2D screen coordinates"""
        view_matrix = self.renderer.view_matrix
        proj_matrix = self.renderer.proj_matrix
        
        N = points_3d.shape[0]
        points_hom = torch.cat([
            points_3d, 
            torch.ones(N, 1, device=self.device)
        ], dim=1)
        
        points_view = points_hom @ view_matrix.T
        points_clip = points_view @ proj_matrix.T
        points_ndc = points_clip[:, :3] / points_clip[:, 3:4]
        
        if self.use_vtk_rendering and self.vtk_resolution is not None:
            resolution_w, resolution_h = self.vtk_resolution
        else:
            resolution_w = resolution_h = self.renderer.resolution
            
        screen_x = (points_ndc[:, 0] + 1.0) * 0.5 * resolution_w
        screen_y = (1.0 - points_ndc[:, 1]) * 0.5 * resolution_h
        
        projection_results = []
        for i, (x, y) in enumerate(zip(screen_x, screen_y)):
            px, py = x.item(), y.item()
            if 0 <= px < resolution_w and 0 <= py < resolution_h:
                projection_results.append((i, (int(px), int(py))))
        
        return projection_results

    def _save_image(self, image_tensor):
        """Save PyTorch tensor as PNG image"""
        image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_path = os.path.join(self.temp_dir, f"render_{timestamp}.png")
        Image.fromarray(image_np).save(image_path)
        return image_path

    def _save_tf_visualization(self):
        """Save TF visualization"""
        tf_save_dir = "./resources/TF_graphs"
        os.makedirs(tf_save_dir, exist_ok=True)
        
        opacity_np = self.tf.opacity_lut.detach().cpu().numpy()
        color_np = self.tf.color_lut.detach().cpu().numpy()
        initial_opacity_np = self.initial_opacity_lut.cpu().numpy()
        num_bins = self.tf.num_bins
        
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(num_bins), initial_opacity_np, 'k--', linewidth=1, label='Initial', alpha=0.3)
        
        for i in range(num_bins):
            color = color_np[i]
            ax.bar(i, opacity_np[i], width=1, color=color, alpha=0.8)
        
        ax.set_xlabel(f'Bin Index')
        ax.set_ylabel('Opacity (0-1)')
        ax.set_title(f'Transfer Function - Iteration {self.iteration}')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, num_bins - 1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        tf_graph_path = os.path.join(tf_save_dir, f"tf_iter_{self.iteration:03d}.png")
        plt.savefig(tf_graph_path, dpi=100)
        plt.close()

    def _save_ray_profiles_visualization(self, grid_samples, debug_data_list, visibility_flags):
        """
        Grid Sample별 Ray Opacity 프로파일 시각화
        
        Args:
            grid_samples: list of {'idx': int, 'pt': (x, y)} 
            debug_data_list: list of debug_data dicts from check_visibility
            visibility_flags: list of bool (is_visible)
        """
        ray_profile_dir = "./resources/ray_profiles"
        os.makedirs(ray_profile_dir, exist_ok=True)
        
        num_samples = len(grid_samples)
        if num_samples == 0:
            return
        
        # Grid layout 계산 (4x4 또는 적절한 크기)
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_samples == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (sample, debug_data, is_visible) in enumerate(zip(grid_samples, debug_data_list, visibility_flags)):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            idx = sample['idx']
            pt = sample['pt']
            
            # Early exit case
            if 'early_exit_reason' in debug_data:
                ax.text(0.5, 0.5, f"Early Exit:\n{debug_data['early_exit_reason']}", 
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Sample #{i} (idx={idx})\n2D: {pt}", fontsize=10)
                continue
            
            t_values = debug_data['t_values']
            intensities = debug_data['intensities']
            raw_opacities = debug_data['raw_opacities']
            effective_opacities = debug_data['effective_opacities']
            cumulative_opacity = debug_data['cumulative_opacity']
            is_target_tissue = debug_data['is_target_tissue']
            is_clipped = debug_data['is_clipped']
            opacity_threshold = debug_data['opacity_threshold']
            
            if len(t_values) == 0:
                ax.text(0.5, 0.5, "No samples", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Sample #{i} (idx={idx})", fontsize=10)
                continue
            
            # Plot lines
            ax.plot(t_values, intensities, 'gray', linewidth=1, alpha=0.7, label='Intensity')
            ax.plot(t_values, raw_opacities, 'b-', linewidth=1.5, alpha=0.7, label='Raw Opacity')
            ax.plot(t_values, effective_opacities, 'orange', linewidth=2, label='Effective Opacity')
            ax.plot(t_values, cumulative_opacity, 'r-', linewidth=2, label='Cumulative Opacity')
            
            # Threshold line
            ax.axhline(y=opacity_threshold, color='red', linestyle='--', alpha=0.5, label=f'Threshold ({opacity_threshold})')
            
            # Shade target tissue skip regions
            if np.any(is_target_tissue):
                for j in range(len(t_values)):
                    if is_target_tissue[j]:
                        ax.axvspan(t_values[max(0,j-1)], t_values[min(len(t_values)-1, j+1)], 
                                alpha=0.2, color='green')
            
            # Shade clipped regions
            if np.any(is_clipped):
                for j in range(len(t_values)):
                    if is_clipped[j]:
                        ax.axvspan(t_values[max(0,j-1)], t_values[min(len(t_values)-1, j+1)], 
                                  alpha=0.2, color='purple')
            
            # Final accumulated opacity text
            final_acc = cumulative_opacity[-1] if len(cumulative_opacity) > 0 else 0
            visibility_str = "VISIBLE" if is_visible else "OCCLUDED"
            color = 'green' if is_visible else 'red'
            
            ax.set_xlabel('Distance along ray')
            ax.set_ylabel('Value (0-1)')
            ax.set_ylim(-0.05, 1.1)
            ax.set_title(f"Sample #{i} (idx={idx}) | {visibility_str}\n"
                        f"Acc Opacity: {final_acc:.3f} | 2D: ({int(pt[0])}, {int(pt[1])})", 
                        fontsize=9, color=color)
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Only show legend on first subplot
                ax.legend(loc='upper left', fontsize=7)
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(ray_profile_dir, f"ray_profiles_iter_{self.iteration:03d}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        
        print(f"   📊 Ray profiles saved: {save_path}")

    def _compute_color_loss(self, image_tensor):
        """Compute color loss for visible mask region in LAB space"""
        import torch.nn.functional as F
        import kornia
        
        if self.color_reference is None or self.last_visible_mask is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        render_h, render_w = image_tensor.shape[:2]
        
        visible_mask_tensor = torch.tensor(self.last_visible_mask, dtype=torch.float32, device=self.device)
        visible_mask_resized = F.interpolate(
            visible_mask_tensor.unsqueeze(0).unsqueeze(0),
            size=(render_h, render_w),
            mode='nearest'
        ).squeeze().bool()
        
        # RGB → LAB 변환 (H,W,3) → (1,3,H,W) → LAB
        rgb_image = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        lab_image = kornia.color.rgb_to_lab(rgb_image)  # (1,3,H,W)
        lab_image = lab_image.squeeze(0).permute(1, 2, 0)  # (H,W,3)
        
        masked_pixels = lab_image[visible_mask_resized]
        
        if masked_pixels.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        current_mean = masked_pixels.mean(dim=0)
        current_std = masked_pixels.std(dim=0)
        
        # 각 채널을 reference std로 정규화 (상대적 오차)
        ref_mean = self.color_reference['mean']
        ref_std = self.color_reference['std']
        
        mean_loss = (
            0.3 * (((current_mean[0] - ref_mean[0]) / (ref_std[0] + 1e-6)) ** 2) +  # L
            1.0 * (((current_mean[1] - ref_mean[1]) / (ref_std[1] + 1e-6)) ** 2) +  # a
            1.0 * (((current_mean[2] - ref_mean[2]) / (ref_std[2] + 1e-6)) ** 2)    # b
        )
        
        std_loss = (
            0.3 * (((current_std[0] - ref_std[0]) / (ref_std[0] + 1e-6)) ** 2) +
            1.0 * (((current_std[1] - ref_std[1]) / (ref_std[1] + 1e-6)) ** 2) +
            1.0 * (((current_std[2] - ref_std[2]) / (ref_std[2] + 1e-6)) ** 2)
        )
        
        return mean_loss + 0.5 * std_loss

    def _calculate_loss(self, image_tensor, image_path, tf_module=None):
        """
        Redesigned loss function (v2) — No SAM in loop, fixed weights

        L_total = L_transmittance * 1.0
                + L_target       * 10.0
                + L_reg          * 5.0
                + L_smooth       * 1.0

        Components:
            L_transmittance: continuous acc_opacity² for ALL sample points
            L_target:        preserve target tissue opacity at initial values
            L_reg:           asymmetric (penalize opacity increase >> decrease)
            L_smooth:        penalize sharp discontinuities in TF
        """
        if tf_module is None:
            tf_module = self.tf

        def soft_invisible_weight(acc_opacity, threshold=0.95, temperature=10.0):
            """
            acc_opacity가 높을수록 (막힘) → weight 높게
            acc_opacity가 낮을수록 (보임) → weight 낮게
            
            Args:
                acc_opacity: tensor of accumulated opacities (0~1)
                threshold: visibility threshold (default 0.95)
                temperature: sharpness of sigmoid (higher = more binary-like)
            
            Returns:
                tensor of soft weights (0~1)
            """
            return torch.sigmoid(temperature * (acc_opacity - threshold))

        # 1. Project 3D points to 2D
        projection_results = self._project_points_to_2d(self.points_3d)

        if not projection_results:
            loss_tensor = torch.tensor(10.0, device=self.device, requires_grad=True)
            return 10.0, loss_tensor

        # 2. Grid Sampling (4x4 = up to 16 representative points)
        grid_samples = sample_grid_representative_points(projection_results, grid_size=4)

        if not grid_samples:
            loss_tensor = torch.tensor(10.0, device=self.device, requires_grad=True)
            return 10.0, loss_tensor

        print(f"   Grid sampled {len(grid_samples)} representative points")

        # 3. Visibility check for ALL sample points (continuous, no binary cutoff for loss)
        visibility_opacities = []
        visibility_flags = []
        debug_data_list = []

        for sample in grid_samples:
            idx = sample['idx']
            point_3d = self.points_3d[idx]

            initial_ray_dir = None
            if self.initial_ray_directions is not None and idx < len(self.initial_ray_directions):
                initial_ray_dir = self.initial_ray_directions[idx]
                if isinstance(initial_ray_dir, np.ndarray):
                    initial_ray_dir = torch.tensor(initial_ray_dir, dtype=torch.float32, device=self.device)

            result = self.renderer.check_visibility( #threshold 조정하는 곳 (기본 0.95)
                point_3d, tf_module, opacity_threshold=0.95,
                initial_ray_direction=initial_ray_dir,
                point_intensity=None,
                target_range=self.target_range,
                clipping_ranges=self.clipping_ranges,
                return_debug=True
            )

            acc_opacity, is_visible, debug_data = result

            visibility_opacities.append(acc_opacity)
            visibility_flags.append(is_visible)
            debug_data_list.append(debug_data)

        # Store debug data for ray profile visualization
        self.last_debug_data_list = debug_data_list
        self.last_grid_samples = grid_samples
        self.last_visible_samples = [s for s, flag in zip(grid_samples, visibility_flags) if flag]

        # Store visibility results (for points overlay visualization)
        self.last_visibility_results = {}
        for sample, opacity, is_vis in zip(grid_samples, visibility_opacities, visibility_flags):
            idx = sample['idx']
            opacity_val = opacity.item() if hasattr(opacity, 'item') else float(opacity)
            self.last_visibility_results[idx] = (is_vis, opacity_val)

        # 초기 visible depth 저장 (첫 iteration에서만)
        if not hasattr(self, 'initial_visible_depth_map'):
            self.initial_visible_depth_map = {}
            for sample, opacity, is_vis in zip(grid_samples, visibility_opacities, visibility_flags):
                if is_vis and opacity.item() < 0.5:
                    idx = sample['idx']
                    self.initial_visible_depth_map[idx] = opacity.detach().clone()

        # Visible preserve loss: 초기에 visible이었던 포인트의 depth 유지
        visible_preserve_pairs = []
        for sample, opacity, is_vis in zip(grid_samples, visibility_opacities, visibility_flags):
            idx = sample['idx']
            if idx in self.initial_visible_depth_map:
                initial_depth = self.initial_visible_depth_map[idx]
                visible_preserve_pairs.append((opacity - initial_depth).pow(2))

        if len(visible_preserve_pairs) > 0:
            visible_preserve_loss = torch.stack(visible_preserve_pairs).mean()
        else:
            visible_preserve_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        visible_count = sum(visibility_flags)
        total_count = len(visibility_flags)
        vis_ratio = visible_count / total_count if total_count > 0 else 0

        print(f"   Visibility: {visible_count}/{total_count} ({vis_ratio:.1%})")

        # =============================================
        # (1) L_transmittance: continuous loss for ALL points
        #     acc_opacity² → gradient always pushes toward transparency
        #     Weighted by per-point SAM certainty:
        #       Interior points (certainty ~1.0) → full weight
        #       Boundary points (certainty ~0.5) → half weight
        # =============================================
        all_opacities = torch.stack(visibility_opacities)

        # Per-point certainty weighting (local)
        point_weights = torch.ones(len(grid_samples), device=self.device)
        if self.point_certainty_weights is not None:
            for i, sample in enumerate(grid_samples):
                idx = sample['idx']
                if idx < len(self.point_certainty_weights):
                    point_weights[i] = self.point_certainty_weights[idx]

        # ===== 수정 부분 시작 =====
        # Soft invisible weighting (opacity 높을수록 weight 높음)
        soft_invisible_weights = soft_invisible_weight(
            all_opacities, 
            threshold=0.95, 
            temperature=10.0
        )
        
        # Combined weighting: SAM certainty × soft invisible weight
        combined_weights = point_weights * soft_invisible_weights
        
        # Transmittance loss with soft weighting
        weighted_opacities = (all_opacities ** 2) * combined_weights
        transmittance_loss = weighted_opacities.sum() / (combined_weights.sum() + 1e-8)
        
        # weighted_opacities = (all_opacities ** 2) * point_weights
        # transmittance_loss = weighted_opacities.sum() / point_weights.sum()

        # =============================================
        # (2) L_target: preserve target tissue opacity
        #     Without this, optimizer would make EVERYTHING transparent
        # =============================================
        current_lut = tf_module.opacity_lut
        initial_lut = self.initial_opacity_lut
        num_bins = tf_module.num_bins

        target_loss = torch.tensor(0.0, device=self.device)
        if self.target_range is not None:
            t_min, t_max = self.target_range
            bin_min = max(0, int(t_min * (num_bins - 1)))
            bin_max = min(num_bins - 1, int(t_max * (num_bins - 1)))

            if bin_max > bin_min:
                target_bins_current = current_lut[bin_min:bin_max + 1]
                target_bins_initial = initial_lut[bin_min:bin_max + 1]
                # Keep target tissue opacity at initial values
                target_loss = ((target_bins_current - target_bins_initial) ** 2).mean()

        # =============================================
        # (3) L_reg: asymmetric regularization
        #     Opacity INCREASE (new occlusion) → strong penalty (1.0x)
        #     Opacity DECREASE (remove occlusion) → weak penalty (0.1x)
        #     Target range bins excluded from regularization
        # =============================================
        delta = current_lut - initial_lut

        # Mask out target bins from regularization (handled by L_target)
        reg_mask = torch.ones(num_bins, device=self.device)
        if self.target_range is not None:
            t_min, t_max = self.target_range
            bin_min = max(0, int(t_min * (num_bins - 1)))
            bin_max = min(num_bins - 1, int(t_max * (num_bins - 1)))
            reg_mask[bin_min:bin_max + 1] = 0.0

        increase_penalty = torch.clamp(delta, min=0.0) ** 2   # opacity went up
        decrease_penalty = torch.clamp(-delta, min=0.0) ** 2   # opacity went down

        reg_per_bin = (increase_penalty * 1.0 + decrease_penalty * 0.1) * reg_mask
        regularization_loss = reg_per_bin.mean()

        # =============================================
        # (4) L_smooth: penalize sharp TF discontinuities
        # =============================================
        smoothness_loss = ((current_lut[1:] - current_lut[:-1]) ** 2).mean()

        # =============================================
        # Convergence check
        #   - All points visible → immediate stop
        #   - vis_ratio > 0.8 + loss plateau → stop
        # =============================================
        if vis_ratio >= 1.0:
            self.converged = True
            print(f"   ** All points visible — converged **")
        elif vis_ratio > 0.8 and self._prev_loss is not None:
            loss_change = abs(transmittance_loss.item() - self._prev_loss)
            if loss_change < 1e-4:
                self.converged = True
        self._prev_loss = transmittance_loss.item()

        # =============================================
        # Combine with fixed weights
        #   reg/smooth raised to better preserve surrounding structures
        #   Global SAM confidence scales transmittance weight:
        #     High confidence (0.95+) → trust the ROI, full optimization
        #     Low confidence (0.6)    → SAM may be wrong, be conservative
        # =============================================
        w_trans = 1.0 * self.sam_confidence  # scaled by global SAM confidence
        w_target = 10.0
        w_reg = 15.0
        w_smooth = 5.0

        visible_preserve_weight = 50.0
        final_visible_preserve_loss = visible_preserve_loss * visible_preserve_weight


        loss_tensor = (
            transmittance_loss * w_trans +
            target_loss * w_target +
            regularization_loss * w_reg +
            smoothness_loss * w_smooth +
            final_visible_preserve_loss
        )

        loss_value = loss_tensor.item()

        print(f"   Loss: {loss_value:.4f} (SAM conf={self.sam_confidence:.3f})")
        print(f"     transmittance = {transmittance_loss.item() * w_trans:.4f} (w={w_trans:.3f})")
        print(f"     target        = {target_loss.item() * w_target:.4f}")
        print(f"     reg(asym)     = {regularization_loss.item() * w_reg:.4f}")
        print(f"     smooth        = {smoothness_loss.item() * w_smooth:.4f}")
        print(f"     visible_preserve = {final_visible_preserve_loss.item():.4f}")
        # SAM masks no longer computed in loop
        self.last_full_mask = None
        self.last_visible_mask = None

        return loss_value, loss_tensor

    # ============== Adam Optimization ==============
    
    def optimize_step(self):
        """Single optimization step for Adam"""
        self.optimizer.zero_grad()
        
        if self.use_vtk_rendering:
            if not self.external_image_ready:
                return {
                    'loss': 0.0,
                    'nodes': self.tf.to_nodes_direct(),
                    'image_path': None,
                    'iteration': self.iteration,
                    'converged': False,
                    'waiting_for_render': True
                }
            
            image_path = self.external_image_path
            self.external_image_ready = False
            image_tensor = self.renderer.render(self.tf)
            self._save_tf_visualization()
        else:
            image_tensor = self.renderer.render(self.tf)
            image_path = self._save_image(image_tensor)
        
        loss_value, loss_tensor = self._calculate_loss(image_tensor, image_path)
        
        if loss_tensor.requires_grad:
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.tf.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.tf.set_opacity_constraint(min_opacity=0.0, max_opacity=1.0)
        
        self.iteration += 1
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_nodes = self.tf.to_nodes_direct()
        
        return {
            'loss': loss_value,
            'nodes': self.tf.to_nodes_direct(),
            'image_path': image_path,
            'iteration': self.iteration,
            'converged': self.converged
        }

    def _optimize_adam(self, num_iterations, callback=None):
        """Adam optimization on binned LUT"""
        
        consecutive_converged = 0
        
        for i in range(num_iterations):
            result = self.optimize_step()
            
            if result.get('waiting_for_render'):
                continue
                
            print(f"[Adam] Iteration {i + 1}/{num_iterations}: Loss={result['loss']:.4f}")
            
            if callback:
                callback(result)
            
            if self.converged:
                consecutive_converged += 1
                if consecutive_converged >= 3:
                    print(f"✅ Early stopping at iteration {i + 1}")
                    break
            else:
                consecutive_converged = 0
        
        return self.best_nodes

    # ============== Nelder-Mead Optimization ==============

    def _nelder_mead_loss(self, peak_opacities, callback=None):
        """
        Loss function for Nelder-Mead (Simplified & Robust Version)
        VTK 렌더링 설정과 상관없이, 최적화 계산은 PyTorch 렌더러를 사용하여
        안정적으로 수행합니다.
        """
        self.iteration += 1
        
        # 1. Optimizer가 제안한 Opacity 값 (0~1 클리핑)
        clipped_opacities = np.clip(peak_opacities, 0.0, 1.0) 
        nodes = self._tents_to_nodes(clipped_opacities)
        
        # 2. 임시 TF 생성
        temp_tf = self._create_tf_from_nodes(nodes)
        
        # 3. [핵심] 무조건 PyTorch로 렌더링하고 이미지 저장
        # (VTK 설정을 무시하고 내부 연산용 이미지를 만듭니다)
        with torch.no_grad():
            image_tensor = self.renderer.render(temp_tf)
        
        # 이미지 파일 저장 (디버그 시각화용)
        image_path = self._save_image(image_tensor)
        
        # 4. Loss 계산 (여기서 직접 계산하므로 값이 보장됨)
        loss_value, _ = self._calculate_loss(image_tensor, image_path, tf_module=temp_tf)
        
        # 5. Best 기록 갱신
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_nodes = nodes
            print(f"✨ New Best! Loss={loss_value:.4f}")
        
        # 6. GUI 업데이트용 콜백 (단순 시각화용, 계산 결과는 안 기다림)
        if callback:
            callback({
                'nodes': nodes,
                'iteration': self.iteration,
                'image_path': image_path,
                'loss': loss_value,
                'full_mask': None,
                'visible_mask': None
            })
            
        print(f"[Nelder-Mead] Iter {self.iteration}: Loss={loss_value:.4f}, Peaks={clipped_opacities}")
        
        return loss_value

    def _optimize_nelder_mead(self, num_iterations, callback=None):
        """Nelder-Mead optimization on original TF node peaks"""
        
        if len(self.tents) == 0:
            print("⚠️ No tents found in TF nodes!")
            return self.initial_nodes
        
        # Initial peak values
        x0 = np.array([tent['peak'] for tent in self.tents])
        
        print(f"🔧 Nelder-Mead optimizing {len(x0)} peak(s): {x0}")
        
        # Bounds (0 to 1)
        bounds = [(0.0, 1.0) for _ in x0]
        
        # Run optimization
        result = minimize(
            self._nelder_mead_loss,
            x0,
            args=(callback,),
            method='Nelder-Mead',
            options={
                'maxiter': num_iterations,
                'xatol': 1e-3,
                'fatol': 1e-3,
                'disp': True
            }
        )
        
        # Final nodes
        final_nodes = self._tents_to_nodes(result.x)
        
        print(f"\n✅ Nelder-Mead complete!")
        print(f"   Initial peaks: {x0}")
        print(f"   Final peaks: {result.x}")
        print(f"   Final loss: {result.fun:.4f}")
        
        return final_nodes

    # ============== Main Entry Point ==============

    def optimize(self, num_iterations=50, method='adam', callback=None):
        """
        Run optimization
        
        Args:
            num_iterations: max iterations
            method: 'adam' or 'nelder-mead'
            callback: optional callback function
        """
        print(f"\n🚀 Starting {method.upper()} optimization ({num_iterations} iterations)...")
        
        self.iteration = 0
        self.best_loss = float('inf')
        self.converged = False
        self._prev_loss = None
        
        if method == 'adam':
            return self._optimize_adam(num_iterations, callback)
        elif method == 'nelder-mead':
            return self._optimize_nelder_mead(num_iterations, callback)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'adam' or 'nelder-mead'")

    def get_current_nodes(self):
        """Get current TF nodes"""
        return self.tf.to_nodes_direct()
    
    def get_best_nodes(self):
        """Get best TF nodes"""
        return self.best_nodes