"""현재 카메라 + TF 기준 view-dependent feature(depth / visibility / opacity)를 voxel마다 계산한다.

feature 정의는 ray_sampler.FeatureAnalyzer의 샘플과 같다.
  depth      — near plane 점부터의 world 거리
  visibility — ΔO = step 보정 opacity(p) × T(p). T는 화면 격자 ray의 앞쪽 투과율 곡선을 쌍선형 + 깊이 선형
               보간한 값, step은 같은 이웃 ray들의 샘플 간격을 보간한 값(ray_sampler와 같은 보정)
"""
import numpy as np

from src.core.clustering import sampling_rules
from src.core.clustering.ray_sampler import FeatureAnalyzer
from src.core.clustering.sampling_rules import (
    RAY_NUM_SAMPLES, correct_opacity, intensity_to_alpha, prev_transmittance)


class ViewFeatureBaker:
    """화면 격자(grid_stride 픽셀 간격) 투과율 곡선을 생성 시 한 번 만들고 voxel feature 계산에 재사용한다.

    카메라·TF·볼륨·클리핑이 바뀌면 새로 만들어야 한다.
    """

    def __init__(self, renderer, volume_actor, window_size, volume_data, voxel_spacing,
                 tf_lut, grid_stride=1, clipping_ranges=None):
        self.renderer = renderer
        self.actor = volume_actor
        self.W, self.H = int(window_size[0]), int(window_size[1])
        self.volume = volume_data
        self.spacing = np.asarray(voxel_spacing, dtype=np.float64)
        self.tf_lut = np.asarray(tf_lut)
        self.g = max(1, int(grid_stride))
        self.clipping_ranges = clipping_ranges
        self.v_min, self.v_max = volume_data.min(), volume_data.max()

        m = volume_actor.GetMatrix()
        self.m2w = np.array([m.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)

        cam = renderer.GetActiveCamera()
        self.parallel = bool(cam.GetParallelProjection())
        self.C = np.array(cam.GetPosition(), dtype=np.float64)
        D = np.array(cam.GetFocalPoint(), dtype=np.float64) - self.C
        self.D = D / np.linalg.norm(D)
        renderer.SetDisplayPoint(self.W / 2.0, self.H / 2.0, 0.0)
        renderer.DisplayToWorld()
        near = np.array(renderer.GetWorldPoint()[:3], dtype=np.float64)
        self.n = float((near - self.C) @ self.D)   # near plane까지 거리
        P = cam.GetCompositeProjectionTransformMatrix(self.W / float(self.H), -1, 1)
        self.P = np.array([P.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)

        self.analyzer = FeatureAnalyzer(renderer, volume_data, voxel_spacing, volume_actor=volume_actor)
        self.build_transmittance_grid()

    def build_transmittance_grid(self):
        """격자 ray마다 RAY_NUM_SAMPLES 샘플의 앞쪽 투과율(T_prev)을 계산해 저장한다."""
        g = self.g
        gx, gy = np.meshgrid(np.arange(0, self.W, g), np.arange(0, self.H, g))
        self.ny, self.nx = gx.shape
        coords = np.column_stack((gx.ravel(), gy.ravel()))
        origins, dirs = self.analyzer.get_model_rays_vectorized(coords)
        tmin, tmax = self.analyzer.intersect_box_vectorized(origins, dirs)
        self.tmin, self.tmax = tmin, tmax
        self.ray_valid = tmax > tmin
        self.ray_step = (tmax - tmin) / (RAY_NUM_SAMPLES - 1)   # ray마다 샘플 간격(world)

        T_prev = np.ones((len(coords), RAY_NUM_SAMPLES), dtype=np.float32)
        step = 8192
        for s in range(0, len(coords), step):
            sl = slice(s, s + step)
            t_steps = np.linspace(tmin[sl], tmax[sl], RAY_NUM_SAMPLES, axis=1)
            alphas, _ = self.analyzer.sample_alphas(origins[sl], dirs[sl], t_steps, self.tf_lut)
            T_prev[sl] = prev_transmittance(correct_opacity(alphas, self.ray_step[sl, None]))
        self.T_prev = T_prev
        self.num_rays = len(coords)

    def to_world(self, p_model):
        h = np.hstack([p_model, np.ones((len(p_model), 1))])
        return (h @ self.m2w.T)[:, :3]

    def depth(self, p_world):
        """near plane 점부터의 거리 (ray_sampler의 sampled_depths와 같은 정의)."""
        v = p_world - self.C
        if self.parallel:
            return v @ self.D - self.n
        dist = np.linalg.norm(v, axis=1)
        cos = (v @ self.D) / np.maximum(dist, 1e-9)
        return dist - self.n / np.maximum(cos, 1e-9)

    def project(self, p_world):
        """world 점 → VTK display 좌표. Returns: (dx, dy, in_front bool)"""
        h = np.hstack([p_world, np.ones((len(p_world), 1))])
        clip = h @ self.P.T
        w = clip[:, 3]
        safe_w = np.where(np.abs(w) < 1e-12, 1e-12, w)
        dx = (clip[:, 0] / safe_w + 1.0) * 0.5 * self.W
        dy = (clip[:, 1] / safe_w + 1.0) * 0.5 * self.H
        return dx, dy, w > 0

    def neighbor_rays(self, dx, dy):
        """(dx, dy) 주변 격자 ray 4개. Returns: [(ray index (N,), 쌍선형 가중치 (N,) — 쓸 수 없는 ray는 0)] × 4"""
        fx, fy = dx / self.g, dy / self.g
        ix0, iy0 = np.floor(fx).astype(np.intp), np.floor(fy).astype(np.intp)
        wx, wy = fx - ix0, fy - iy0
        out = []
        for ox, oy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)),
                          (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
            ix, iy = ix0 + ox, iy0 + oy
            inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            r = np.where(inside, iy * self.nx + ix, 0)
            out.append((r, np.where(inside & self.ray_valid[r], w, 0.0)))
        return out

    def transmittance(self, dx, dy, t, neighbors=None):
        """(dx, dy) 주변 격자 ray 4개에서 깊이 t의 앞쪽 투과율을 보간한다.

        Returns: (N,) — 쓸 수 있는 이웃 ray가 없으면 nan
        """
        n = RAY_NUM_SAMPLES
        acc = np.zeros(len(t), dtype=np.float64)
        wsum = np.zeros(len(t), dtype=np.float64)
        for r, w in (neighbors or self.neighbor_rays(dx, dy)):
            span = np.maximum(self.tmax[r] - self.tmin[r], 1e-9)
            u = np.clip((t - self.tmin[r]) / span * (n - 1), 0.0, n - 1)
            i0 = np.minimum(np.floor(u).astype(np.intp), n - 2)
            f = u - i0
            Tv = self.T_prev[r, i0] * (1.0 - f) + self.T_prev[r, i0 + 1] * f
            acc += w * Tv
            wsum += w
        return np.where(wsum > 1e-6, acc / np.maximum(wsum, 1e-6), np.nan)

    def ray_step_at(self, dx, dy, neighbors=None):
        """(dx, dy) 주변 격자 ray들의 샘플 간격을 transmittance와 같은 가중치로 보간한다(이웃이 없으면 0)."""
        acc = wsum = 0.0
        for r, w in (neighbors or self.neighbor_rays(dx, dy)):
            acc = acc + w * self.ray_step[r]
            wsum = wsum + w
        return acc / np.maximum(wsum, 1e-6)

    def alpha(self, values):
        return intensity_to_alpha(values, self.tf_lut, self.v_min, self.v_max)

    def view_features(self, p_model, values):
        """voxel(model 좌표)의 view-dependent feature.

        Returns: (depth, visibility, opacity, classifiable)
            visibility — sampling_rules.VISIBILITY_FEATURE가 "T"면 앞쪽 투과율, 아니면 ΔO(step 보정 opacity × 투과율)
            opacity    — TF opacity (step 보정 전, Stage A feature 열과 같은 정의)
            classifiable=False — 카메라 뒤, 화면 밖, 클리핑된 voxel
        """
        p_world = self.to_world(p_model)
        dx, dy, front = self.project(p_world)
        t = self.depth(p_world)
        nb = self.neighbor_rays(dx, dy)
        T = self.transmittance(dx, dy, t, nb)
        a = self.alpha(values)
        ok = front & np.isfinite(T)
        if self.clipping_ranges:
            ok &= ~self.analyzer.is_point_clipped_vectorized(p_model, self.clipping_ranges)
        T = np.where(ok, np.nan_to_num(T), 0.0)
        vis = T if sampling_rules.VISIBILITY_FEATURE == "T" else correct_opacity(a, self.ray_step_at(dx, dy, nb)) * T
        return t, vis, a, ok
