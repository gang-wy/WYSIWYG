"""
Feature Analyzer for Volume Rendering Optimization

SAM 마스크 영역에서 3D Feature(Intensity) 및 공간 정보를 추출하는 분석 엔진.
"""

import numpy as np
import vtk
from scipy.ndimage import map_coordinates
from scipy.signal import find_peaks


class FeatureAnalyzer:
    """
    SAM 마스크 영역에서 3D Feature(Intensity) 및 공간 정보를 추출하는 분석 엔진.
    """
    
    def __init__(self, renderer, volume_data, voxel_spacing, volume_actor=None):
        self.renderer = renderer
        self.volume_data = volume_data  # Shape: (X, Y, Z)
        self.spacing = np.array(voxel_spacing)
        self.dims = np.array(volume_data.shape)
        self.actor = volume_actor

    def analyze_roi_profile(self, screen_coords, tf_lut, peeling_depth=1, sample_mode='surface', clipping_ranges=None):
        """
        [최적화 버전] 모든 Ray를 행렬 연산으로 한 번에 처리하여 3D 정보를 추출합니다.
        """
        screen_coords = np.array(screen_coords)
        num_rays = len(screen_coords)
        if num_rays == 0:
            return {}

        # 1. 모든 Ray의 Origin과 Direction을 한 번에 생성
        origins, directions = self._get_model_rays_vectorized(screen_coords)

        # 2. 모든 Ray에 대한 Box Intersection (N, 2)
        t_mins, t_maxs = self._intersect_box_vectorized(origins, directions)
        
        # 교차하지 않는 레이 제외
        valid_ray_mask = t_mins is not None
        if not np.any(valid_ray_mask):
            return {}
        
        # 3. 고정 길이 샘플링 (모든 레이를 동일한 개수로 샘플링하여 행렬화)
        num_samples = 256
        t_steps = np.linspace(t_mins, t_maxs, num_samples, axis=1)
        
        # 모든 샘플의 3D 좌표 계산 (N, M, 3)
        pts = origins[:, np.newaxis, :] + directions[:, np.newaxis, :] * t_steps[:, :, np.newaxis]
        
        # 4. 벌크 보간 (Bulk Interpolation)
        flat_pts = pts.reshape(-1, 3) / self.spacing
        all_intensities = map_coordinates(
            self.volume_data, 
            flat_pts.T, 
            order=1, 
            mode='constant', 
            cval=0
        ).reshape(num_rays, num_samples)

        # 5. Opacity & Peak 분석 (N, M)
        v_min, v_max = self.volume_data.min(), self.volume_data.max()
        norm_v = np.clip(((all_intensities - v_min) / (v_max - v_min + 1e-8) * 255), 0, 255).astype(int)
        alphas = tf_lut[norm_v]

        # Front-to-back 누적 계산
        transmittance = np.cumprod(1.0 - alphas, axis=1)
        prev_transmittance = np.roll(transmittance, 1, axis=1)
        prev_transmittance[:, 0] = 1.0
        contributions = alphas * prev_transmittance

        # 결과 추출용 리스트
        picked_intensities = []
        picked_depths = []
        picked_world_points = []
        target_indices_list = []
        initial_ray_directions = []
        success_indices = []
        
        camera_pos = np.array(self.renderer.GetActiveCamera().GetPosition())

        for i in range(num_rays):
            peaks, props = find_peaks(
                contributions[i], 
                height=0.05,
                distance=3,
                width=True, 
                rel_height=0.5
            )
            
            if len(peaks) == 0:
                continue
            
            peak_contributions = contributions[i][peaks]
            best_peak_idx = np.argmax(peak_contributions)
            
            left_side = props['left_ips'][best_peak_idx]
            right_side = props['right_ips'][best_peak_idx]
            idx = np.random.randint(int(left_side), int(right_side) + 1)
            # idx = int((left_side + right_side) / 2)
                
            target_idx = idx
            target_t = t_steps[i, idx]
            val = all_intensities[i, idx]
            
            model_pt = origins[i] + directions[i] * target_t

            # 클리핑 체크
            if self._is_point_clipped(model_pt, clipping_ranges):
                continue

            target_indices_list.append(target_idx)
            picked_intensities.append(val)
            picked_depths.append(target_t)

            world_pt = self._to_world(model_pt)
            picked_world_points.append(world_pt)

            ray_dir = (world_pt - camera_pos)
            initial_ray_directions.append(ray_dir / (np.linalg.norm(ray_dir) + 1e-8))
            success_indices.append(i)

        if not picked_intensities:
            return {}

        return {
            'profiles': all_intensities[np.array(success_indices)],
            'target_indices': np.array(target_indices_list),
            'target_range': (np.percentile(picked_intensities, 10), np.percentile(picked_intensities, 90)),
            'picked_intensities': np.array(picked_intensities),
            'picked_depths': np.array(picked_depths),
            'picked_points': np.array(picked_world_points),
            'initial_ray_directions': np.array(initial_ray_directions),
            'success_indices': np.array(success_indices)
        }

    def _get_model_rays_vectorized(self, screen_coords):
        """모든 화면 좌표를 한 번에 Model Ray로 변환"""
        num_rays = len(screen_coords)
        origins = np.zeros((num_rays, 3))
        directions = np.zeros((num_rays, 3))
        
        for i, (dx, dy) in enumerate(screen_coords):
            self.renderer.SetDisplayPoint(dx, dy, 0.0)
            self.renderer.DisplayToWorld()
            near = np.array(self.renderer.GetWorldPoint()[:3])
            
            self.renderer.SetDisplayPoint(dx, dy, 1.0)
            self.renderer.DisplayToWorld()
            far = np.array(self.renderer.GetWorldPoint()[:3])
            
            origins[i] = near
            directions[i] = (far - near) / (np.linalg.norm(far - near) + 1e-8)

        # Actor Matrix가 있다면 한 번에 변환
        if self.actor:
            mat = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Invert(self.actor.GetMatrix(), mat)
            
            orig_4d = np.hstack([origins, np.ones((num_rays, 1))])
            dir_4d = np.hstack([directions, np.zeros((num_rays, 1))])
            
            np_mat = np.array([mat.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)
            
            origins = (orig_4d @ np_mat.T)[:, :3]
            directions = (dir_4d @ np_mat.T)[:, :3]
            
        return origins, directions

    def _intersect_box_vectorized(self, o, d, bmin=None, bmax=None):
        """Ray-Box Intersection의 벡터화 버전"""
        if bmin is None:
            bmin = np.array([0, 0, 0])
        if bmax is None:
            bmax = self.dims * self.spacing
        
        inv_d = 1.0 / (d + 1e-8)
        t1 = (bmin - o) * inv_d
        t2 = (bmax - o) * inv_d
        
        tmin = np.max(np.minimum(t1, t2), axis=1)
        tmax = np.min(np.maximum(t1, t2), axis=1)
        
        valid = (tmax > tmin) & (tmax > 0)
        tmin_final = np.where(valid, np.maximum(0, tmin), 0)
        tmax_final = np.where(valid, tmax, 0)
        
        return tmin_final, tmax_final

    def _is_point_clipped(self, point_model_space, clipping_ranges):
        """
        포인트가 클리핑 범위 밖인지 체크
        """
        if clipping_ranges is None:
            return False
        
        bounds_max = self.dims * self.spacing
        normalized = point_model_space / bounds_max
        
        axes = ['x', 'y', 'z']
        for i, axis in enumerate(axes):
            if axis in clipping_ranges:
                min_val, max_val = clipping_ranges[axis]
                if normalized[i] < min_val or normalized[i] > max_val:
                    return True
        
        return False

    def _to_world(self, p):
        """Model space → World space 변환"""
        if not self.actor:
            return p
        p4 = self.actor.GetMatrix().MultiplyPoint([p[0], p[1], p[2], 1.0])
        return np.array(p4[:3]) / p4[3]
    
    def visualize_ray_profiles(self, results, tf_lut, max_rays=12, save_path=None):
        """
        analyze_roi_profile() 결과를 받아 ray별 프로파일을 2D 그래프로 시각화.

        각 subplot:
          - intensity (파란선)
          - alpha/opacity (주황선)
          - transmittance (초록선)
          - contribution (빨간선)
          - target point (세로 점선)

        Args:
            results: analyze_roi_profile()의 반환값
            tf_lut:  현재 opacity LUT (shape: [256])
            max_rays: 표시할 최대 ray 수
            save_path: 저장 경로 (None이면 plt.show())
        """
        import matplotlib
        matplotlib.use('Agg' if save_path else 'TkAgg')
        import matplotlib.pyplot as plt

        profiles = results.get('profiles')          # (N, 256)
        target_indices = results.get('target_indices')  # (N,)
        if profiles is None or len(profiles) == 0:
            print("[visualize_ray_profiles] 결과에 profiles 없음.")
            return

        num_rays = min(len(profiles), max_rays)
        cols = 3
        rows = (num_rays + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3.5))
        axes = np.array(axes).flatten()

        v_min = self.volume_data.min()
        v_max = self.volume_data.max()

        for i in range(num_rays):
            ax = axes[i]
            intensity = profiles[i]                          # (256,)
            x = np.arange(len(intensity))

            # opacity
            norm_v = np.clip(((intensity - v_min) / (v_max - v_min + 1e-8) * 255), 0, 255).astype(int)
            alpha = tf_lut[norm_v]

            # transmittance & contribution
            transmittance = np.cumprod(1.0 - alpha)
            prev_t = np.roll(transmittance, 1); prev_t[0] = 1.0
            contribution = alpha * prev_t

            ax2 = ax.twinx()

            ax.plot(x, intensity, color='steelblue', lw=1.2, label='Intensity')
            ax2.plot(x, alpha,         color='darkorange', lw=1.0, alpha=0.85, label='Alpha')
            ax2.plot(x, transmittance, color='seagreen',   lw=1.0, alpha=0.85, label='Transmit')
            ax2.plot(x, contribution,  color='crimson',    lw=1.2, alpha=0.9,  label='Contribution')

            # target point 표시
            if target_indices is not None and i < len(target_indices):
                tidx = target_indices[i]
                ax.axvline(x=tidx, color='black', lw=1.5, ls='--', alpha=0.8)
                ax.plot(tidx, intensity[tidx], 'k^', ms=7, zorder=5)

            ax.set_title(f'Ray {i}', fontsize=9)
            ax.set_xlabel('Sample index', fontsize=8)
            ax.set_ylabel('Intensity', fontsize=8, color='steelblue')
            ax2.set_ylabel('Opacity / Contrib', fontsize=8)
            ax2.set_ylim(0, 1)

            # 범례는 첫 번째 subplot에만
            if i == 0:
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')

        # 빈 subplot 숨기기
        for j in range(num_rays, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f'Ray Profiles (총 {len(profiles)}개 ray 중 {num_rays}개 표시)', fontsize=11)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[visualize_ray_profiles] 저장: {save_path}")
        else:
            plt.show()
            
    def visualize_sam_point_opacity(self, screen_coords, tf_lut, save_dir="ray_profiles_sam_points"):
        """
        SAM input point(2D screen 좌표)마다 해당 ray의 accumulated opacity 그래프를 저장.
        """
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)

        origins, directions = self._get_model_rays_vectorized(screen_coords)
        t_mins, t_maxs = self._intersect_box_vectorized(origins, directions)

        v_min = self.volume_data.min()
        v_max = self.volume_data.max()

        num_samples = 256
        t_steps = np.linspace(t_mins, t_maxs, num_samples, axis=1)
        pts = origins[:, np.newaxis, :] + directions[:, np.newaxis, :] * t_steps[:, :, np.newaxis]

        flat_pts = pts.reshape(-1, 3) / self.spacing
        all_intensities = map_coordinates(
            self.volume_data, flat_pts.T, order=1, mode='constant', cval=0
        ).reshape(len(screen_coords), num_samples)

        for i, (sx, sy) in enumerate(screen_coords):
            intensity = all_intensities[i]
            x = np.arange(num_samples)

            norm_v = np.clip(((intensity - v_min) / (v_max - v_min + 1e-8) * 255), 0, 255).astype(int)
            alpha = tf_lut[norm_v]
            accumulated_opacity = 1.0 - np.cumprod(1.0 - alpha)

            fig, ax = plt.subplots(figsize=(8, 4))

            ax.plot(x, accumulated_opacity, color='crimson', lw=2.0)
            ax.set_title(f'SAM Point {i}  (screen: {int(sx)}, {int(sy)})', fontsize=10)
            ax.set_xlabel('Sample index')
            ax.set_ylabel('Accumulated Opacity')
            ax.set_ylim(0, 1)

            plt.tight_layout()
            save_path = os.path.join(save_dir, f"sam_point_{i:02d}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[visualize_sam_point_opacity] 저장: {save_path}")