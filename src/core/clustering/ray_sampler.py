"""화면 좌표에서 볼륨으로 ray를 쏴 Stage A 입력 샘플(intensity/opacity/visibility/depth/INR)을 만든다."""

import numpy as np
import vtk
from scipy.ndimage import map_coordinates

from src.core.clustering.sampling_rules import (
    RAY_NUM_SAMPLES, correct_opacity, intensity_to_alpha, prev_transmittance)


class FeatureAnalyzer:
    """현재 카메라 기준 ray 생성, 박스 교차, 볼륨/INR 보간을 담당한다."""

    def __init__(self, renderer, volume_data, voxel_spacing, volume_actor=None, inr_feature_volume=None):
        self.renderer = renderer
        self.volume_data = volume_data                # (X, Y, Z)
        self.spacing = np.array(voxel_spacing)
        self.dims = np.array(volume_data.shape)
        self.actor = volume_actor
        self.inr_feature_volume = inr_feature_volume  # (X, Y, Z, C) 또는 None

    def compute_adaptive_stride(self, window_height_px, safety_factor=0.6):
        """가장 먼 볼륨 코너에서 voxel 하나가 화면에 투영되는 픽셀 크기 기준 ray 간격.

        Returns: int (>= 1) 픽셀 stride
        """
        camera = self.renderer.GetActiveCamera()
        cam_pos = np.array(camera.GetPosition())
        view_angle_rad = np.radians(camera.GetViewAngle())

        if self.actor:
            bounds = self.actor.GetBounds()
        else:
            bx, by, bz = self.dims * self.spacing
            bounds = (0, bx, 0, by, 0, bz)
        corners = np.array(np.meshgrid(bounds[0:2], bounds[2:4], bounds[4:6])).T.reshape(-1, 3)
        d_max = float(np.max(np.linalg.norm(corners - cam_pos, axis=1)))

        voxel_world_size = float(np.min(self.spacing))
        pixels_per_voxel = voxel_world_size * window_height_px / (2 * d_max * np.tan(view_angle_rad / 2))
        return max(1, int(np.floor(pixels_per_voxel * safety_factor)))

    def sample_alphas(self, origins, directions, t_steps, tf_lut):
        """ray 위 t_steps 지점의 볼륨 값을 보간하고 opacity로 변환한다.

        Returns: (alphas, intensities) — 둘 다 t_steps와 같은 shape
        """
        pts = origins[:, np.newaxis, :] + directions[:, np.newaxis, :] * t_steps[:, :, np.newaxis]
        flat = pts.reshape(-1, 3) / self.spacing
        vals = map_coordinates(self.volume_data, flat.T, order=1, mode='constant', cval=0)
        vals = vals.reshape(t_steps.shape)
        v_min, v_max = self.volume_data.min(), self.volume_data.max()
        return intensity_to_alpha(vals, tf_lut, v_min, v_max), vals

    def analyze_roi_profile(self, screen_coords, tf_lut, clipping_ranges=None):
        """화면 좌표마다 ray를 쏴 박스 교차 구간을 RAY_NUM_SAMPLES개로 균등 샘플링한다.

        Args:
            screen_coords: (N, 2) VTK display 좌표(좌하단 원점)
            tf_lut: (256,) opacity LUT
            clipping_ranges: {'x': (min, max), ...} 정규화 클리핑 범위 또는 None

        Returns: dict (유효 샘플이 없으면 빈 dict)
            sampled_points       (M, 3) world 좌표
            sampled_intensities  (M,)
            sampled_opacity      (M,)   TF opacity (step 보정 전)
            sampled_visibility   (M,)   ΔO = step 보정 opacity × 앞쪽 투과율
            sampled_transmittance (M,)  앞쪽 투과율 (step 보정 opacity로 누적)
            sampled_depths       (M,)   near plane 점부터의 거리
            sampled_inr_features (M, C) 또는 None
            sample_valid         (N, RAY_NUM_SAMPLES) bool — 박스 교차·clipping 통과 여부
        """
        screen_coords = np.array(screen_coords)
        num_rays = len(screen_coords)
        if num_rays == 0:
            return {}

        origins, directions = self.get_model_rays_vectorized(screen_coords)
        t_mins, t_maxs = self.intersect_box_vectorized(origins, directions)
        valid_ray_mask = t_maxs > t_mins
        if not np.any(valid_ray_mask):
            return {}

        num_samples = RAY_NUM_SAMPLES
        t_steps = np.linspace(t_mins, t_maxs, num_samples, axis=1)
        pts = origins[:, np.newaxis, :] + directions[:, np.newaxis, :] * t_steps[:, :, np.newaxis]

        all_intensities = map_coordinates(
            self.volume_data, (pts.reshape(-1, 3) / self.spacing).T,
            order=1, mode='constant', cval=0
        ).reshape(num_rays, num_samples)

        v_min, v_max = self.volume_data.min(), self.volume_data.max()
        alphas = intensity_to_alpha(all_intensities, tf_lut, v_min, v_max)
        # 합성(T, ΔO)은 렌더러처럼 ray step 길이로 보정한 opacity로 한다. feature 열 opacity는 TF 값 그대로.
        alphas_step = correct_opacity(alphas, ((t_maxs - t_mins) / (num_samples - 1))[:, None])
        transmittance = prev_transmittance(alphas_step)
        visibility = alphas_step * transmittance

        # 박스와 교차하지 않는 ray, 클리핑 범위 밖 샘플 제외
        sample_valid = np.broadcast_to(valid_ray_mask[:, None], (num_rays, num_samples)).copy()
        if clipping_ranges is not None:
            clipped = self.is_point_clipped_vectorized(pts.reshape(-1, 3), clipping_ranges)
            sample_valid &= ~clipped.reshape(num_rays, num_samples)

        flat_valid = sample_valid.reshape(-1)
        if not np.any(flat_valid):
            return {}

        sampled_model_points = pts.reshape(-1, 3)[flat_valid]
        return {
            'sampled_intensities': all_intensities.reshape(-1)[flat_valid],
            'sampled_depths': t_steps.reshape(-1)[flat_valid],
            'sampled_visibility': visibility.reshape(-1)[flat_valid],
            'sampled_transmittance': transmittance.reshape(-1)[flat_valid],
            'sampled_opacity': alphas.reshape(-1)[flat_valid],
            'sampled_points': self.to_world_vectorized(sampled_model_points),
            'sampled_inr_features': self.sample_inr_features(sampled_model_points),
            'sample_valid': sample_valid,   # (ray 수, 샘플 수) — 위 sampled_*는 이 값이 True인 자리를 ray 순서대로 편 것
        }

    def sample_inr_features(self, sampled_model_points, chunk_size=2_000_000):
        """model 좌표에서 INR feature volume을 전 채널 trilinear 보간한다.

        map_coordinates(order=1, mode='constant', cval=0)와 같은 결과(볼륨 밖은 0).
        Returns: (N, C) float32, feature volume이 없으면 None
        """
        if self.inr_feature_volume is None or len(sampled_model_points) == 0:
            return None

        vol = self.inr_feature_volume
        dims = np.array(vol.shape[:3])
        num_channels = vol.shape[-1]
        model_pts = np.asarray(sampled_model_points) / self.spacing

        inr_features = np.empty((len(model_pts), num_channels), dtype=np.float32)
        for start in range(0, len(model_pts), chunk_size):
            pts = model_pts[start:start + chunk_size]

            # 마지막 voxel이 frac=1.0으로 복원되도록 clip 이후에 frac을 계산
            idx = np.clip(np.floor(pts).astype(np.intp), 0, dims - 2)
            frac = (pts - idx).astype(np.float32)
            x0, y0, z0 = idx[:, 0], idx[:, 1], idx[:, 2]
            fx, fy, fz = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]

            acc = np.zeros((len(pts), num_channels), dtype=np.float32)
            for xi, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
                for yi, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
                    for zi, wz in ((z0, 1.0 - fz), (z0 + 1, fz)):
                        acc += (wx * wy * wz) * vol[xi, yi, zi]

            acc[np.any((pts < 0) | (pts > dims - 1), axis=1)] = 0.0
            inr_features[start:start + chunk_size] = acc

        return inr_features

    def get_model_rays_vectorized(self, screen_coords):
        """화면 좌표 → model 공간 ray. Returns: (origins (N,3) near plane 점, directions (N,3) 단위벡터)"""
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

        if self.actor:
            mat = vtk.vtkMatrix4x4()
            vtk.vtkMatrix4x4.Invert(self.actor.GetMatrix(), mat)
            np_mat = np.array([mat.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)
            origins = (np.hstack([origins, np.ones((num_rays, 1))]) @ np_mat.T)[:, :3]
            directions = (np.hstack([directions, np.zeros((num_rays, 1))]) @ np_mat.T)[:, :3]

        return origins, directions

    def intersect_box_vectorized(self, o, d, bmin=None, bmax=None):
        """ray와 볼륨 박스의 교차 구간. Returns: (t_min, t_max) — 교차하지 않으면 둘 다 0"""
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
        return np.where(valid, np.maximum(0, tmin), 0), np.where(valid, tmax, 0)

    def is_point_clipped_vectorized(self, points_model_space, clipping_ranges):
        """Returns: (N,) bool — True면 클리핑 범위 밖"""
        clipped = np.zeros(len(points_model_space), dtype=bool)
        if clipping_ranges is None:
            return clipped

        normalized = points_model_space / (self.dims * self.spacing)
        for i, axis in enumerate(('x', 'y', 'z')):
            if axis in clipping_ranges:
                min_val, max_val = clipping_ranges[axis]
                clipped |= (normalized[:, i] < min_val) | (normalized[:, i] > max_val)
        return clipped

    def to_world_vectorized(self, points_model_space):
        """model 좌표 (N,3) → world 좌표 (N,3)."""
        if not self.actor:
            return points_model_space

        mat = self.actor.GetMatrix()
        np_mat = np.array([mat.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)
        world_4d = np.hstack([points_model_space, np.ones((len(points_model_space), 1))]) @ np_mat.T

        w = world_4d[:, 3:4].copy()
        w[w == 0] = 1e-8
        return world_4d[:, :3] / w
