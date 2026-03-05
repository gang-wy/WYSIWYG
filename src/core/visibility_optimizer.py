"""
Visibility-based Transfer Function Optimizer

Comparison method that:
1. Loads a 3D binary segmentation mask (same size as volume)
2. Computes visibility V = Σ T(p) × O(p) via 2D ray-based approach (Eq. 4 & 5)
   - Casts all rays (like render()) in a single vectorized pass
   - At each sample point, checks if it belongs to ROI via grid_sample on segmentation volume
   - T(p): transmittance at each sample = cumprod(1 - α) along the ray
   - O(p): opacity defined by the current TF at the sample's intensity
   - Only accumulates T(p) × O(p) for samples inside the ROI
3. Sets target_visibility = 1.2 × initial_visibility
4. Optimizes TF using Adam (128-bin LUT) until target is reached or max_iter

Key difference from SAM-based method: cost function is purely visibility-based,
using a pre-defined 3D segmentation mask instead of SAM-generated ROI.

Performance: All rays processed in parallel on GPU via vectorized operations.
No per-voxel for loop — segmentation lookup is done via grid_sample.
"""

import os
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.core.diff_renderer import DifferentiableVolumeRenderer
    from src.core.diff_transfer_function import DifferentiableTF
except ImportError:
    from diff_renderer import DifferentiableVolumeRenderer
    from diff_transfer_function import DifferentiableTF


class VisibilityOptimizer:
    """
    Visibility-based TF Optimizer (Comparison Method)
    
    Uses V = Σ T(p) × O(p) as the optimization objective.
    Computes visibility via 2D ray casting (vectorized, GPU-parallel).
    Optimizes 128-bin LUT with Adam to increase ROI visibility.
    """
    
    def __init__(self, volume_data, spacing, initial_nodes, camera_info,
                 segmentation_mask, device='cuda',
                 num_samples_per_ray=128, render_resolution=256):
        """
        Args:
            volume_data: numpy array (X, Y, Z) - raw volume
            spacing: tuple (sx, sy, sz) - voxel spacing
            initial_nodes: list of [intensity, R, G, B, A] - initial TF
            camera_info: dict with camera parameters
            segmentation_mask: numpy array (X, Y, Z) - binary mask, same shape as volume_data
            device: 'cuda', 'mps', or 'cpu'
            num_samples_per_ray: samples along each ray
            render_resolution: resolution for ray casting (width=height)
        """
        self.device = device
        self.camera_info = camera_info
        self.volume_data = volume_data
        self.spacing = spacing
        self.initial_nodes = initial_nodes
        self.render_resolution = render_resolution
        
        # Volume range
        self.volume_range = (volume_data.min(), volume_data.max())
        
        # Create differentiable TF (128-bin LUT, same as Adam mode)
        self.tf = DifferentiableTF(
            initial_nodes,
            volume_range=self.volume_range,
            num_bins=128,
            device=device
        )
        
        # Create differentiable renderer (reuse for ray generation & volume interpolation)
        self.renderer = DifferentiableVolumeRenderer(
            volume_data=volume_data,
            spacing=spacing,
            resolution=render_resolution,
            num_samples=num_samples_per_ray,
            device=device
        )
        self.renderer.set_camera(camera_info)
        
        # Prepare segmentation as a GPU volume for grid_sample lookup
        self._prepare_segmentation_volume(segmentation_mask)
        
        roi_count = int(segmentation_mask.sum())
        print(f"✅ VisibilityOptimizer initialized (Ray-based):")
        print(f"   Device: {device}")
        print(f"   Resolution: {render_resolution}×{render_resolution} ({render_resolution**2} rays)")
        print(f"   Samples per ray: {num_samples_per_ray}")
        print(f"   ROI voxels in segmentation: {roi_count:,}")
        print(f"   TF bins: {self.tf.num_bins}")
        
        # Adam optimizer
        self.optimizer = torch.optim.Adam(
            self.tf.parameters(),
            lr=0.01,
            betas=(0.9, 0.999)
        )
        
        # Store initial opacity LUT for regularization
        self.initial_opacity_lut = self.tf.opacity_lut.detach().clone()
        
        # Tracking
        self.iteration = 0
        self.best_loss = float('inf')
        self.best_nodes = initial_nodes
        self.converged = False
        
        # Initial visibility (computed on first step)
        self.initial_visibility = None
        self.target_visibility = None
        
        # Temp dir for debug images
        self.temp_dir = tempfile.mkdtemp(prefix='vis_opt_')
        
        # History for plotting
        self.visibility_history = []
        self.loss_history = []
    
    def _prepare_segmentation_volume(self, segmentation_mask):
        """
        Convert binary segmentation mask to a GPU tensor for grid_sample lookup.
        
        The segmentation volume is stored in the same format as the renderer's volume:
        (1, 1, D, H, W) for F.grid_sample compatibility.
        
        Args:
            segmentation_mask: (X, Y, Z) binary numpy array
        """
        seg_tensor = torch.tensor(
            segmentation_mask.astype(np.float32), 
            dtype=torch.float32, 
            device=self.device
        )
        # Shape: (1, 1, X, Y, Z) — same convention as renderer.volume
        self.seg_volume = seg_tensor.unsqueeze(0).unsqueeze(0)
        
        print(f"   Segmentation volume on GPU: {self.seg_volume.shape}")
    
    def _lookup_segmentation(self, sample_points):
        """
        Check if sample points belong to ROI using grid_sample on segmentation volume.
        
        Args:
            sample_points: (N, S, 3) world coordinates
            
        Returns:
            roi_mask: (N, S) float tensor, ~1.0 if inside ROI, ~0.0 if outside
        """
        N, S, _ = sample_points.shape
        
        # Convert world coords to normalized grid coords [-1, 1]
        # Same coordinate transform as renderer._interpolate_volume
        normalized_coords = sample_points / self.renderer.volume_bounds
        grid_coords = normalized_coords * 2.0 - 1.0
        
        # Swap axes: (x, y, z) → (z, y, x) for (D, H, W) volume
        grid_coords = torch.stack([
            grid_coords[..., 2],
            grid_coords[..., 1],
            grid_coords[..., 0]
        ], dim=-1)
        
        # Reshape for grid_sample: (1, N*S, 1, 1, 3)
        grid_coords_flat = grid_coords.view(1, N * S, 1, 1, 3)
        
        # grid_sample on segmentation volume
        sampled = F.grid_sample(
            self.seg_volume,
            grid_coords_flat,
            mode='nearest',  # Binary mask → nearest neighbor (no blurring)
            padding_mode='zeros',  # Outside volume = not ROI
            align_corners=True
        )
        
        # Reshape back: (N, S)
        roi_mask = sampled.view(N, S)
        
        return roi_mask
    
    def compute_visibility(self, tf_module=None):
        """
        Compute ROI visibility via vectorized 2D ray casting.
        
        V = Σ T(p) × O(p)  for all sample points p ∈ ROI
        
        Process:
        1. Generate all rays (same as render())
        2. Sample along rays → get 3D positions (N, S, 3)
        3. Interpolate volume → get intensities (N, S)
        4. Apply TF → get opacities (N, S)
        5. Compute transmittance T via cumprod (N, S)
        6. Lookup segmentation → get ROI mask (N, S)
        7. V = sum(T * O * roi_mask) over all (ray, sample) pairs
        
        Returns:
            visibility: scalar torch tensor (differentiable)
            visibility_map_2d: (H, W) tensor — per-pixel visibility contribution
        """
        if tf_module is None:
            tf_module = self.tf
        
        # 1. Generate all rays
        ray_origins, ray_directions = self.renderer._generate_rays()
        N = ray_origins.shape[0]  # resolution²
        
        # 2. Ray-volume intersection
        t_min, t_max, valid_mask = self.renderer._intersect_volume(ray_origins, ray_directions)
        
        # 3. Sample along rays: (N, S, 3) positions, (N, S) distances
        sample_points, sample_t = self.renderer._sample_along_rays(
            ray_origins, ray_directions, t_min, t_max
        )
        
        # 4. Interpolate volume at all sample points → intensities (N, S)
        intensities = self.renderer._interpolate_volume(sample_points)
        
        # 5. Apply TF → get RGBA (N, S, 4)
        rgba = tf_module(intensities)
        opacities = rgba[..., 3]  # (N, S) — O(p) for each sample
        
        # 6. Compute step sizes for opacity correction
        step_sizes = sample_t[:, 1:] - sample_t[:, :-1]
        step_sizes = torch.cat([step_sizes, step_sizes[:, -1:]], dim=1)
        
        # Opacity correction for step size: α_corrected = 1 - (1-α)^step_size
        corrected_opacities = 1.0 - torch.pow(1.0 - opacities + 1e-8, step_sizes)
        
        # 7. Compute transmittance T(p) via cumprod
        # transmittance[i] = Π(1 - α[j]) for j < i
        one_minus_alpha = 1.0 - corrected_opacities
        transmittance = torch.cumprod(
            torch.cat([
                torch.ones(N, 1, device=self.device),
                one_minus_alpha[:, :-1]
            ], dim=1),
            dim=1
        )  # (N, S)
        
        # 8. Lookup segmentation mask at each sample point
        roi_mask = self._lookup_segmentation(sample_points)  # (N, S), ~1.0 or ~0.0
        
        # 9. Compute per-sample visibility contribution: T(p) × O(p) × roi_mask(p)
        per_sample_visibility = transmittance * corrected_opacities * roi_mask  # (N, S)
        
        # 10. Sum over samples per ray → per-ray visibility (N,)
        per_ray_visibility = per_sample_visibility.sum(dim=1)
                
        # Zero out invalid rays
        per_ray_visibility = per_ray_visibility * valid_mask.float()

        # Identify rays that pass through ROI (any sample hits ROI)
        roi_ray_mask = (roi_mask.sum(dim=1) > 0) & valid_mask  # (N,) bool
        num_roi_rays = roi_ray_mask.float().sum().clamp(min=1.0)

        # Mean visibility over ROI-passing rays only → [0, 1] range
        visibility = per_ray_visibility.sum() / num_roi_rays

        # Store for logging
        self._last_num_roi_rays = int(num_roi_rays.item())

        # Reshape to 2D map for visualization
        res = self.render_resolution
        visibility_map_2d = per_ray_visibility.view(res, res)
        
        return visibility, visibility_map_2d
    
    def _calculate_loss(self, current_visibility, tf_module=None):
        """
        Loss function for visibility optimization.
        
        L = L_visibility + L_reg + L_smooth
        
        L_visibility: (V_target - V_current)² / V_target² — main objective
        L_reg: regularization to prevent drastic TF change
        L_smooth: smoothness penalty on TF
        """
        if tf_module is None:
            tf_module = self.tf
        
        # (1) Visibility loss: normalized MSE to target
        vis_loss = ((self.target_visibility - current_visibility) / (self.target_visibility + 1e-8)) ** 2
        
        # # (2) Regularization: penalize changes from initial TF
        # current_lut = tf_module.opacity_lut
        # initial_lut = self.initial_opacity_lut
        # delta = current_lut - initial_lut
        
        # # Asymmetric: penalize increase more than decrease
        # increase_penalty = torch.clamp(delta, min=0.0) ** 2
        # decrease_penalty = torch.clamp(-delta, min=0.0) ** 2
        # reg_loss = (increase_penalty * 1.0 + decrease_penalty * 0.3).mean()
        
        # # (3) Smoothness
        # smooth_loss = ((current_lut[1:] - current_lut[:-1]) ** 2).mean()
        
        # # Weights
        # w_vis = 10.0
        # w_reg = 5.0
        # w_smooth = 2.0
        
        loss_tensor = vis_loss
        loss_value = loss_tensor.item()
        
        return loss_value, loss_tensor, {
            'vis_loss': vis_loss.item()
        }
    
    def optimize_step(self):
        """
        Single Adam optimization step.
        
        Returns:
            dict with iteration results
        """
        self.optimizer.zero_grad()
        
        # Compute current visibility (fully vectorized)
        current_visibility, visibility_map_2d = self.compute_visibility(self.tf)
        
        # Save 2D visibility map for debugging
        self._save_visibility_map(visibility_map_2d)
        
        if self.initial_visibility is None:
            self.initial_visibility = current_visibility.detach().clone()
            # self.target_visibility = self.initial_visibility * 1.2
            self.target_visibility = torch.tensor(0.407, device=self.device)
            print(f"📊 Initial visibility (mean over ROI rays): {self.initial_visibility.item():.6f}")
            print(f"📊 ROI rays: {self._last_num_roi_rays} / {self.render_resolution**2} total rays")
            print(f"🎯 Target visibility (×1.2): {self.target_visibility.item():.6f}")
        
        # Calculate loss
        loss_value, loss_tensor, loss_components = self._calculate_loss(current_visibility)
        
        # Backward & step
        if loss_tensor.requires_grad:
            loss_tensor.backward()
            torch.nn.utils.clip_grad_norm_(self.tf.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.tf.set_opacity_constraint(min_opacity=0.0, max_opacity=1.0)
        
        self.iteration += 1
        vis_val = current_visibility.item()
        target_val = self.target_visibility.item()
        
        # Track history
        self.visibility_history.append(vis_val)
        self.loss_history.append(loss_value)
        
        # Check convergence: visibility reached target
        if vis_val >= target_val:
            self.converged = True
            print(f"   🎯 Target visibility reached! ({vis_val:.4f} >= {target_val:.4f})")
        
        # Update best
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.best_nodes = self.tf.to_nodes_direct()
        
        print(f"   [Vis-Opt] Iter {self.iteration}: V={vis_val:.4f} / target={target_val:.4f} | "
              f"Loss={loss_value:.4f} (vis={loss_components['vis_loss']:.4f})")
        
        return {
            'loss': loss_value,
            'visibility': vis_val,
            'target_visibility': target_val,
            'nodes': self.tf.to_nodes_direct(),
            'iteration': self.iteration,
            'converged': self.converged,
            'loss_components': loss_components,
        }
    
    def optimize(self, num_iterations=100, callback=None):
        """
        Run full optimization loop.
        
        Args:
            num_iterations: max iterations
            callback: optional callback(result_dict) per iteration
            
        Returns:
            final_nodes: optimized TF nodes
        """
        print(f"\n🚀 Starting Visibility-based optimization ({num_iterations} iterations)...")
        print(f"   Method: 2D ray-based (vectorized GPU)")
        
        self.iteration = 0
        self.best_loss = float('inf')
        self.converged = False
        self.initial_visibility = None
        self.target_visibility = None
        self.visibility_history = []
        self.loss_history = []
        
        consecutive_converged = 0
        
        for i in range(num_iterations):
            result = self.optimize_step()
            
            if callback:
                callback(result)
            
            if self.converged:
                consecutive_converged += 1
                if consecutive_converged >= 3:
                    print(f"✅ Visibility target reached — stopping at iteration {i + 1}")
                    break
            else:
                consecutive_converged = 0
        
        # Save final visualization
        self._save_convergence_plot()
        
        return self.best_nodes
    
    def _save_visibility_map(self, visibility_map_2d):
        """
        Save 2D visibility map as a heatmap image for debugging.
        
        Args:
            visibility_map_2d: (H, W) torch tensor — per-pixel visibility contribution
        """
        # Use results folder if set by worker, otherwise fallback
        if hasattr(self, 'vis_map_save_dir') and self.vis_map_save_dir:
            save_dir = self.vis_map_save_dir
        else:
            save_dir = "./resources/visibility_optimization/maps"
        os.makedirs(save_dir, exist_ok=True)
        
        vis_np = visibility_map_2d.detach().cpu().numpy()  # (H, W)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Heatmap (jet colormap, log scale for better contrast)
        ax1 = axes[0]
        vis_log = np.log1p(vis_np)
        im = ax1.imshow(vis_log, cmap='jet', interpolation='nearest')
        ax1.set_title(f'Visibility Map (log) — Iter {self.iteration}')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
        # 2. Binary ROI hit map
        ax2 = axes[1]
        roi_hit = (vis_np > 0).astype(np.float32)
        ax2.imshow(roi_hit, cmap='gray', interpolation='nearest')
        ax2.set_title(f'ROI Hit Map — {int(roi_hit.sum())} rays')
        
        for ax in axes:
            ax.axis('off')
        
        plt.suptitle(f'Iteration {self.iteration} | V={vis_np.sum():.4f}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f"vismap_iter_{self.iteration:03d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    def _save_convergence_plot(self):
        """Save visibility convergence plot"""
        if not self.visibility_history:
            return
        
        save_dir = "./resources/visibility_optimization"
        os.makedirs(save_dir, exist_ok=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Visibility plot
        iters = range(1, len(self.visibility_history) + 1)
        ax1.plot(iters, self.visibility_history, 'b-', linewidth=2, label='Current V')
        if self.target_visibility is not None:
            ax1.axhline(y=self.target_visibility.item(), color='r', linestyle='--', 
                       linewidth=1.5, label=f'Target V ({self.target_visibility.item():.2f})')
        if self.initial_visibility is not None:
            ax1.axhline(y=self.initial_visibility.item(), color='gray', linestyle=':', 
                       linewidth=1, label=f'Initial V ({self.initial_visibility.item():.2f})')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Visibility')
        ax1.set_title('Visibility Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss plot
        ax2.plot(iters, self.loss_history, 'r-', linewidth=2)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Loss')
        ax2.set_title('Loss Convergence')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(save_dir, f"convergence_{timestamp}.png"), dpi=150)
        plt.close()
        print(f"📊 Convergence plot saved to {save_dir}")
    
    def get_current_nodes(self):
        return self.tf.to_nodes_direct()
    
    def get_best_nodes(self):
        return self.best_nodes