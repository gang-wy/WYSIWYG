"""on-device 이식 검증용 기준 데이터를 저장하고, 두 기준 데이터의 결과를 비교한다.

프리셋 하나를 앱과 똑같이 실행(SAM → Stage A → Stage B)한 뒤 단계별 입력/출력을 .npz 하나로 저장한다.
  camera_*        VTK 카메라(위치/초점/up/화각/clipping range/창 크기/view·projection 행렬), model_to_world
  sam_*           SAM 입력 이미지/포인트(VTK·이미지 좌표), 결과 mask(+ MobileSAM이면 logits), 소요 시간
  tf_lut          Stage A/B에 쓴 opacity LUT (256,)
  sample_*        Stage A ray 샘플 (points/intensity/inr/depth/visibility/opacity)
  kmeans_*        Stage A K-means 결과 (labels/centroids/feat_mean/feat_std/scores/best/layout)
  roi_*           best cluster intensity 구간 (range_norm, center_norm)
  probe_*         무작위 voxel 10만 개의 view feature(depth/visibility/opacity/ok)와 RF P(best)
  final_mask      Stage B 최종 local mask (X,Y,Z) bool
  final_labels    Final 라벨 볼륨 (X,Y,Z) int16

--inr-dtype으로 INR feature를 float16/uint8로 양자화(→ float32 복원)해서 실행할 수 있다.
--sam mobile이면 SAM3 대신 MobileSAM으로 mask를 만든다(같은 캡처 이미지·포인트).
--compare A.npz B.npz는 두 결과의 SAM mask IoU, Stage A best 샘플 일치도, 최종 mask Dice를 출력하고
SAM mask 겹침 이미지(빨강=A만, 초록=B만, 노랑=둘 다)를 저장한다.

실행 예시 (프로젝트 루트에서):
    .venv\\Scripts\\python.exe scripts\\export_reference.py --dataset tooth
    .venv\\Scripts\\python.exe scripts\\export_reference.py --dataset tooth --inr-dtype uint8
    .venv\\Scripts\\python.exe scripts\\export_reference.py --dataset tooth --inr-dtype uint8 --sam mobile
    .venv\\Scripts\\python.exe scripts\\export_reference.py --compare resources/reference_data/tooth_float32.npz resources/reference_data/tooth_uint8.npz
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_gt_evaluation import setup_window, wait_for_signal, SAM_TIMEOUT_MS  # noqa: E402  (VTK 환경변수 설정 포함)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.core.dataset_presets import DATASET_PRESETS  # noqa: E402
from src.core.clustering import sampling_rules  # noqa: E402
from src.core.sam.mobile_sam_wrapper import MobileSAMWrapper  # noqa: E402
from src.core.clustering.volume_classification import (  # noqa: E402
    stage_a_reference, build_full_feature_matrix, normalize_with, _rf_input)

PROBE_VOXELS = 100000


def quantize_inr(inr, dtype):
    """INR feature(채널값 0~1)를 dtype으로 저장했다가 float32로 되돌린 값. Returns: (X,Y,Z,C) float32"""
    if dtype == "float32":
        return inr
    if dtype == "float16":
        return inr.astype(np.float16).astype(np.float32)
    if dtype == "uint8":
        return (np.round(np.clip(inr, 0.0, 1.0) * 255.0).astype(np.uint8).astype(np.float32) / 255.0)
    raise ValueError(dtype)


def vtk_matrix(m):
    return np.array([m.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)


def camera_record(win):
    """VTK 카메라와 볼륨 변환 정보. Returns: dict(camera_*, model_to_world, voxel_spacing)"""
    r = win.rendering_panel.vtk_renderer
    cam = r.renderer.GetActiveCamera()
    w, h = r.vtk_widget.GetRenderWindow().GetSize()
    return {
        "camera_position": np.array(cam.GetPosition()),
        "camera_focal_point": np.array(cam.GetFocalPoint()),
        "camera_view_up": np.array(cam.GetViewUp()),
        "camera_view_angle": np.float64(cam.GetViewAngle()),
        "camera_clipping_range": np.array(cam.GetClippingRange()),
        "camera_window_size": np.array([w, h]),
        "camera_view_matrix": vtk_matrix(cam.GetViewTransformMatrix()),
        "camera_projection_matrix": vtk_matrix(cam.GetProjectionTransformMatrix(w / float(h), -1, 1)),
        "model_to_world": vtk_matrix(r.standard_volume.GetMatrix()),
        "voxel_spacing": np.asarray(win.voxel_spacing, dtype=np.float64),
    }


def probe_record(win, cluster_result, rng):
    """무작위 voxel의 view feature와 RF P(best). Returns: dict(probe_*)"""
    baker = win._stage_b_view_cache[1]
    forest = next(iter(cluster_result["_stage_b_forests"].values()))
    vol, inr = win.volume_data, win.inr_feature_volume
    shape = vol.shape[:3]

    flat = rng.choice(int(np.prod(shape)), min(PROBE_VOXELS, int(np.prod(shape))), replace=False)
    idx = np.column_stack(np.unravel_index(flat, shape))
    values = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
    depth, visibility, opacity, ok = baker.view_features(idx * baker.spacing, values)

    ref = stage_a_reference(cluster_result, win.current_roi_info)
    inr_vals = inr[idx[:, 0], idx[:, 1], idx[:, 2]]
    feats = build_full_feature_matrix(ref["layout"], inr_features=inr_vals, depths=depth,
                                      visibility=visibility, intensities=values, opacity=opacity)
    p_best = np.full(len(idx), np.nan, dtype=np.float32)
    if ok.any():
        p_best[ok] = forest.predict_best_proba(_rf_input(forest, normalize_with(ref, feats[ok]), inr_vals[ok]))
    return {"probe_voxel_index": idx.astype(np.int32), "probe_depth": depth.astype(np.float32),
            "probe_visibility": visibility.astype(np.float32), "probe_opacity": opacity.astype(np.float32),
            "probe_ok": ok, "probe_p_best": p_best}


def capture_sam_input(win):
    """main_window.on_check_sam과 같은 방식으로 화면을 캡처하고 포인트를 이미지 좌표로 바꾼다.

    Returns: (image_path, image (H,W,3) uint8, points_vtk (N,2), points_img (N,2), labels (N,))
    """
    image_path = win.rendering_panel.save_current_rendering(return_filename=True)
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    picked = win.optimization_panel.picked_points
    points_vtk = np.array([pt["pos"] for pt in picked], dtype=np.float64)
    points_img = np.array([[max(0, min(int(x), w - 1)), max(0, min(int(h - y), h - 1))] for x, y in points_vtk])
    labels = np.array([1 if pt["type"] == "positive" else 0 for pt in picked], dtype=np.int32)
    return image_path, image, points_vtk, points_img, labels


def run_sam(win, sam, image_path, points_img, labels):
    """sam("sam3" | "mobile")으로 mask를 만들어 main_window에 넣는다. Returns: (logits 또는 None, 소요 초)"""
    if sam == "sam3":
        t0 = time.time()
        win.sam_model.predict_async(image_path, points_img.tolist(), labels.tolist())
        wait_for_signal(win.sam_model.predicted, timeout_ms=SAM_TIMEOUT_MS)
        elapsed = time.time() - t0
        QApplication.processEvents()
        return None, elapsed   # 첫 호출이라 모델 로드 대기 시간이 포함될 수 있음

    wrapper = MobileSAMWrapper()
    wrapper.predict(image_path, points_img, labels)          # 워밍업
    t0 = time.time()
    mask, logits = wrapper.predict(image_path, points_img, labels)
    elapsed = time.time() - t0
    win.on_sam_mask_ready((mask, logits))
    return logits, elapsed


def export(args):
    """프리셋을 실행하고 단계별 결과를 .npz로 저장한다."""
    app, win, volume_data, voxel_spacing = setup_window(args.dataset)
    win.inr_feature_volume = quantize_inr(win.inr_feature_volume, args.inr_dtype)
    print(f"INR dtype: {args.inr_dtype}, SAM: {args.sam}")

    image_path, sam_image, sam_points, sam_points_img, sam_labels = capture_sam_input(win)
    sam_logits, sam_elapsed = run_sam(win, args.sam, image_path, sam_points_img, sam_labels)
    if win.current_sam_mask is None:
        raise RuntimeError("SAM 예측 실패")
    print(f"SAM mask {int(np.sum(win.current_sam_mask)):,}px, {sam_elapsed:.2f}초")

    t0 = time.time()
    win.on_clustering_clicked()
    QApplication.processEvents()
    elapsed = time.time() - t0

    roi = win.current_roi_info
    cr = roi["cluster_result"] if roi else None
    if cr is None or win.current_local_mask is None:
        raise RuntimeError("Stage A 또는 Stage B 실패")

    record = {
        "dataset": args.dataset, "inr_dtype": args.inr_dtype, "sam": args.sam, "elapsed_sec": elapsed,
        # 이 키가 있으면 T·ΔO가 step 보정 정의 (export_stage_a_fixture.py가 visibility 재계산 여부를 정한다)
        "opacity_unit_distance": np.float64(sampling_rules.OPACITY_UNIT_DISTANCE),
        "volume_shape": np.array(volume_data.shape),
        "sam_image": sam_image, "sam_points_vtk": sam_points, "sam_points_img": sam_points_img,
        "sam_labels": sam_labels, "sam_elapsed_sec": sam_elapsed,
        "sam_mask": np.asarray(win.current_sam_mask, dtype=bool),
        "tf_lut": win.tf_panel.tf_widget.get_opacity_lut_for_nodes(win.base_tf_nodes),
        "sample_points": roi["filtered_points"].astype(np.float32),
        "sample_intensity": roi["filtered_intensities"],
        "sample_inr": roi["filtered_inr_features"],
        "sample_depth": roi["filtered_depths"],
        "sample_visibility": roi["filtered_visibility"],
        "sample_opacity": roi["filtered_opacity"],
        "kmeans_labels": np.asarray(cr["labels"], dtype=np.int16),
        "kmeans_centroids": cr["centroids"], "kmeans_feat_mean": cr["feat_mean"],
        "kmeans_feat_std": cr["feat_std"], "kmeans_scores": cr["scores"],
        "kmeans_best": np.int32(cr["best_cluster_index"]),
        "kmeans_layout": np.array([f"{n}:{s}:{e}" for n, s, e in cr["feature_layout"]]),
        "roi_range_norm": np.array(roi["range_norm"]), "roi_center_norm": np.float64(roi["center_norm"]),
        "final_mask": np.asarray(win.current_local_mask[0]) > 0,
        "final_labels": win.current_cluster_label_volume,
    }
    if sam_logits is not None:
        record["sam_logits"] = sam_logits.astype(np.float16)
    record.update(camera_record(win))
    record.update(probe_record(win, cr, np.random.RandomState(0)))

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.sam == "sam3" else "_mobilesam"
    path = os.path.join(args.out_dir, f"{args.dataset}_{args.inr_dtype}{suffix}.npz")
    np.savez_compressed(path, **record)
    print(f"\n저장: {path} ({os.path.getsize(path) / 2**20:.1f}MB) — Stage A 샘플 {len(record['sample_intensity']):,}, "
          f"최종 mask {int(record['final_mask'].sum()):,} voxel, 소요 {elapsed:.1f}초")


def save_mask_overlay(image, mask_a, mask_b, path):
    """흑백으로 바꾼 캡처 이미지 위에 두 mask를 겹쳐 저장한다(빨강=A만, 초록=B만, 노랑=둘 다)."""
    out = np.repeat(image.astype(np.float32).mean(axis=2, keepdims=True), 3, axis=2)
    for region, color in ((mask_a & ~mask_b, (255, 40, 40)), (~mask_a & mask_b, (40, 220, 40)),
                          (mask_a & mask_b, (255, 220, 0))):
        out[region] = 0.45 * out[region] + 0.55 * np.array(color, dtype=np.float32)
    Image.fromarray(out.astype(np.uint8)).save(path)
    print(f"겹침 이미지: {path}")


def compare(path_a, path_b):
    """두 기준 데이터의 SAM mask, Stage A best 샘플, 최종 mask 일치도를 출력한다."""
    a, b = np.load(path_a, allow_pickle=True), np.load(path_b, allow_pickle=True)
    print(f"A: {path_a}\nB: {path_b}")

    sa, sb = a["sam_mask"], b["sam_mask"]
    if sa.shape == sb.shape:
        iou = int((sa & sb).sum()) / max(int((sa | sb).sum()), 1)
        print(f"SAM mask: A {int(sa.sum()):,} / B {int(sb.sum()):,}px, IoU {iou:.4f}")
        if "sam_image" in a.files:
            save_mask_overlay(a["sam_image"], sa, sb, os.path.splitext(path_b)[0] + "_vs_A_sam.png")
    else:
        print(f"SAM mask 크기가 다름: {sa.shape} vs {sb.shape}")

    n_a, n_b = len(a["sample_intensity"]), len(b["sample_intensity"])
    if n_a == n_b:
        best_a = a["kmeans_labels"] == a["kmeans_best"]
        best_b = b["kmeans_labels"] == b["kmeans_best"]
        inter, union = int((best_a & best_b).sum()), int((best_a | best_b).sum())
        print(f"Stage A best 샘플: A {int(best_a.sum()):,} / B {int(best_b.sum()):,}, IoU {inter / max(union, 1):.4f}")
        print(f"INR 샘플 최대 오차: {float(np.abs(a['sample_inr'] - b['sample_inr']).max()):.5f}")
    else:
        print(f"Stage A 샘플 수가 다름 (A {n_a:,}, B {n_b:,}) — 샘플 단위 비교 생략")

    print(f"ROI intensity 구간: A {np.round(a['roi_range_norm'], 4)} / B {np.round(b['roi_range_norm'], 4)}")

    ma, mb = a["final_mask"], b["final_mask"]
    inter = int((ma & mb).sum())
    dice = 2 * inter / max(int(ma.sum()) + int(mb.sum()), 1)
    print(f"최종 mask: A {int(ma.sum()):,} / B {int(mb.sum()):,} voxel, Dice {dice:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASET_PRESETS), default="tooth")
    parser.add_argument("--inr-dtype", choices=["float32", "float16", "uint8"], default="float32")
    parser.add_argument("--sam", choices=["sam3", "mobile"], default="sam3")
    parser.add_argument("--out-dir", default="resources/reference_data")
    parser.add_argument("--compare", nargs=2, metavar=("A_NPZ", "B_NPZ"))
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
    else:
        export(args)


if __name__ == "__main__":
    main()
