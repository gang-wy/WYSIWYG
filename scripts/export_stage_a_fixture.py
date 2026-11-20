"""Unity(Galaxy XR) Stage A/B C# 구현 검증용 데이터를 기준 데이터(*_uint8_mobilesam.npz)에서 만든다.

기준 데이터의 Stage A 샘플은 ray마다 256개가 순서대로 들어 있고 model_to_world가 항등이므로,
각 ray의 첫/마지막 샘플 점과 depth로 ray(near plane 시작점, 방향)를 복원할 수 있다.

출력: <Unity>/Assets/TestData/StageA/<preset>/
  rays.bin           float64 (ray 수, 6) — origin xyz, dir xyz (model 공간, mm)
  check_samples.bin  float32 (검사 ray 수 × 256, 15) — intensity, opacity, visibility, depth, INR 11채널
  best_mask.bin      np.packbits(bitorder="little") — 전체 샘플의 best cluster 여부
  probe.bin          (probe 수, 8) 4바이트씩 — voxel i, j, k (int32), depth, visibility, opacity, ok, P(best) (float32)
  final_mask.bin     np.packbits(bitorder="little") — Stage B 최종 mask, NIfTI 순서(x가 가장 빠름)
  fixture.json       tf_lut, 검사 ray 번호, 기준 결과(roi 구간, best 샘플 수, 점수, feature mean/std),
                     카메라(model→clip 행렬 = projection × view, 창 크기, 위치, 시선, 화각, 적응형 stride)

INR 좌표 3채널은 기준 데이터에서 uint8로 양자화돼 있고(export_reference.quantize_inr), Unity는 정확한 값
(index / (N-1))을 쓰므로 좌표 채널은 허용 오차(≈1/510)로 비교해야 한다.

실행 예시 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe scripts\\export_stage_a_fixture.py
    .venv\\Scripts\\python.exe scripts\\export_stage_a_fixture.py --dataset tooth
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.clustering.sampling_rules import correct_opacity, prev_transmittance  # noqa: E402

REF_DIR = "resources/reference_data"
UNITY_OUT = r"D:\VOMLAB\WYSIWYG_GalaxyXR\Assets\TestData\StageA"
RAY_SAMPLES = 256
NUM_CHECK_RAYS = 64
NUM_PROBES = 20000


def adaptive_stride(d, safety_factor=0.6):
    """FeatureAnalyzer.compute_adaptive_stride와 같은 식 (bounds = [0, (N-1)·spacing]). Returns: int >= 1"""
    shape, spacing = d["volume_shape"], d["voxel_spacing"]
    hi = (shape - 1) * spacing
    corners = np.array(np.meshgrid(*[(0, h) for h in hi])).T.reshape(-1, 3)
    d_max = float(np.max(np.linalg.norm(corners - d["camera_position"], axis=1)))
    ppv = float(np.min(spacing)) * float(d["camera_window_size"][1]) / (
        2 * d_max * np.tan(np.radians(float(d["camera_view_angle"])) / 2))
    return max(1, int(np.floor(ppv * safety_factor)))


def recover_rays(points, depths):
    """샘플 점(ray 수 × 256, 3)과 depth로 ray를 복원한다. Returns: (origins, dirs) float64 (ray 수, 3)"""
    first, last = points[:, 0].astype(np.float64), points[:, -1].astype(np.float64)
    dirs = last - first
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    origins = first - dirs * depths[:, 0:1].astype(np.float64)
    return origins, dirs


def export(path):
    d = np.load(path, allow_pickle=True)
    preset = str(d["dataset"])
    if not np.allclose(d["model_to_world"], np.eye(4)):
        raise ValueError(f"{preset}: model_to_world가 항등이 아님")
    n = len(d["sample_intensity"])
    if n % RAY_SAMPLES:
        raise ValueError(f"{preset}: 샘플 수 {n}이 {RAY_SAMPLES}의 배수가 아님 (제외된 샘플 있음)")
    num_rays = n // RAY_SAMPLES

    points = d["sample_points"].reshape(num_rays, RAY_SAMPLES, 3)
    depths = d["sample_depth"].reshape(num_rays, RAY_SAMPLES)
    origins, dirs = recover_rays(points, depths)

    # 기준 데이터에 opacity_unit_distance가 없으면 step 보정 전 정의라, 샘플 visibility만 지금 정의(sampling_rules)로
    # 다시 계산한다(intensity·opacity·depth·INR은 보정과 무관).
    # shortcut: K-means 결과·Stage B probe·최종 mask는 보정 전 정의 그대로 — export_reference.py로 기준 데이터를 다시 만들면 불필요
    visibility = d["sample_visibility"]
    recomputed = "opacity_unit_distance" not in d.files
    if recomputed:
        step = (depths[:, -1] - depths[:, 0]).astype(np.float64) / (RAY_SAMPLES - 1)
        a_step = correct_opacity(d["sample_opacity"].reshape(num_rays, RAY_SAMPLES), step[:, None])
        visibility = (a_step * prev_transmittance(a_step)).reshape(-1)

    check_rays = np.unique(np.linspace(0, num_rays - 1, NUM_CHECK_RAYS).round().astype(int))
    per_sample = np.column_stack([
        d["sample_intensity"], d["sample_opacity"], visibility, d["sample_depth"], d["sample_inr"],
    ]).astype(np.float32).reshape(num_rays, RAY_SAMPLES, -1)[check_rays].reshape(-1, 15)

    best_mask = d["kmeans_labels"] == d["kmeans_best"]

    out = os.path.join(UNITY_OUT, preset)
    os.makedirs(out, exist_ok=True)
    np.column_stack([origins, dirs]).astype("<f8").tofile(os.path.join(out, "rays.bin"))
    per_sample.astype("<f4").tofile(os.path.join(out, "check_samples.bin"))
    np.packbits(best_mask, bitorder="little").tofile(os.path.join(out, "best_mask.bin"))

    # Stage B: probe voxel의 view feature / P(best), 최종 mask
    num_probes = min(NUM_PROBES, len(d["probe_voxel_index"]))
    probe = np.zeros((num_probes, 8), dtype="<f4")
    probe_view = probe.view("<i4")
    probe_view[:, 0:3] = d["probe_voxel_index"][:num_probes]
    for col, key in enumerate(("probe_depth", "probe_visibility", "probe_opacity", "probe_ok", "probe_p_best"), 3):
        probe[:, col] = d[key][:num_probes].astype(np.float32)
    probe.tofile(os.path.join(out, "probe.bin"))
    np.packbits(d["final_mask"].ravel(order="F"), bitorder="little").tofile(os.path.join(out, "final_mask.bin"))

    camera_P = d["camera_projection_matrix"] @ d["camera_view_matrix"]
    with open(os.path.join(out, "fixture.json"), "w", encoding="utf-8") as f:
        json.dump({
            "preset": preset,
            "volume_shape": d["volume_shape"].tolist(),
            "voxel_spacing": d["voxel_spacing"].tolist(),
            "num_rays": int(num_rays),
            "num_samples": int(n),
            "check_rays": check_rays.tolist(),
            "tf_lut": d["tf_lut"].astype(float).tolist(),
            "roi_range_norm": d["roi_range_norm"].tolist(),
            "roi_center_norm": float(d["roi_center_norm"]),
            "best_count": int(best_mask.sum()),
            "kmeans_scores": d["kmeans_scores"].astype(float).tolist(),
            "kmeans_best": int(d["kmeans_best"]),
            "feat_mean": d["kmeans_feat_mean"].tolist(),
            "feat_std": d["kmeans_feat_std"].tolist(),
            "camera_P": camera_P.ravel().tolist(),
            "camera_window_size": d["camera_window_size"].tolist(),
            "camera_position": d["camera_position"].tolist(),
            "camera_direction": (d["camera_focal_point"] - d["camera_position"]).tolist(),
            "camera_view_angle": float(d["camera_view_angle"]),
            "adaptive_stride": adaptive_stride(d),
            "num_probes": int(num_probes),
            "final_count": int(d["final_mask"].sum()),
            "note": "rays: model 공간(mm) near plane 시작점·단위 방향. 기준 = " + os.path.basename(path)
                    + (" (샘플 visibility만 step 보정 정의로 재계산, K-means·probe·mask는 보정 전)" if recomputed else ""),
        }, f, ensure_ascii=False, indent=1)

    # 복원 확인: 복원한 ray로 다시 계산한 샘플 점과 기준 점의 최대 차이(mm)
    t = depths.astype(np.float64)
    rebuilt = origins[:, None, :] + dirs[:, None, :] * t[:, :, None]
    err = np.abs(rebuilt - points).max()
    print(f"{preset:<11} ray {num_rays}, 샘플 {n}, 검사 ray {len(check_rays)}, best {best_mask.sum()}, "
          f"복원 오차 최대 {err:.2e}mm → {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", help="프리셋 이름 (없으면 전체)")
    args = parser.parse_args()
    pattern = f"{args.dataset}_uint8_mobilesam.npz" if args.dataset else "*_uint8_mobilesam.npz"
    for path in sorted(glob.glob(os.path.join(REF_DIR, pattern))):
        export(path)


if __name__ == "__main__":
    main()
