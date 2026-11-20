"""clustering 결과 하나를 GT segmentation과 정량 비교한다(Dice / IoU / 정밀도 / 재현율).

on_clustering_clicked() 한 번으로 Stage A + Stage B가 실행되고, Stage A seed voxel과
최종 local mask(current_local_mask)를 각각 GT와 비교해 출력한다.

GT 파일은 두 형태를 모두 받는다.
  - 3D 라벨 볼륨   (예: tooth_GT.nii — 0 배경 / 1 법랑질 / 2 상아질 / 3 신경)
  - 4D 채널 볼륨   (예: Engine_GT.nii — 채널마다 독립적인 이진 마스크)
--gt-label로 라벨(또는 채널)을 고르고, 생략하면 후보 목록만 출력하고 끝난다.

실행 예시 (반드시 WYSIWYG 프로젝트 루트에서):
    .venv\\Scripts\\python.exe scripts\\run_gt_evaluation.py --dataset tooth --gt-label 3
    .venv\\Scripts\\python.exe scripts\\run_gt_evaluation.py --dataset tooth --gt-label 3 --visibility --depth
    .venv\\Scripts\\python.exe scripts\\run_gt_evaluation.py --dataset tooth --gt-label 3 --no-visibility --no-depth
"""
import argparse
import glob
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# main.py와 같은 VTK 렌더링 백엔드(OSMesa). QApplication 생성 전에 설정해야 SAM 스크린샷이 수동 실행과 같다.
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
os.environ['VTK_USE_OSMESA'] = '1'

import json

import numpy as np
import nibabel as nib
from scipy import ndimage

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QEventLoop, QTimer, Qt, QPoint
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtTest import QTest

from src.main_window import VolumeRenderingMainWindow
from src.core.dataset_presets import DATASET_PRESETS as DATASETS
from src.core.clustering.volume_classification import stage_a_best_cluster_voxels

SAM_TIMEOUT_MS = 180000  # 3분
NEIGHBORHOOD = np.ones((3, 3, 3), dtype=np.float32)


# ----------------------------------------------------------------------------
# 앱 구동 헬퍼 (export_reference.py도 사용)
# ----------------------------------------------------------------------------
def wait_for_signal(signal, timeout_ms):
    """Qt 시그널이 한 번 발생할 때까지 블로킹 대기 (SAM 비동기 예측용)."""
    loop = QEventLoop()
    fired = {}

    def _on_fire(*args):
        fired["args"] = args
        loop.quit()

    conn = signal.connect(_on_fire)
    timer = QTimer()
    timer.setSingleShot(True)
    timer.timeout.connect(loop.quit)
    timer.start(timeout_ms)
    loop.exec()
    signal.disconnect(conn)


def set_global_tf(win, tf_nodes):
    """tf_panel 시그널을 거치지 않고 global TF를 main_window 상태와 렌더러에 직접 반영한다."""
    nodes_copy = [n[:] for n in tf_nodes]
    win.base_tf_nodes = nodes_copy
    win.preview_tf_nodes = None
    win.preview_active = False
    win.rendering_panel.update_transfer_function(nodes_copy)


def settle_first_render(win, camera_info):
    """뷰포트에 짧은 드래그 이벤트를 보내 첫 프레임을 정상화한 뒤 camera_info로 카메라를 복원한다."""
    vtk_widget = win.rendering_panel.vtk_renderer.vtk_widget
    start = vtk_widget.rect().center()
    moved = start + QPoint(8, 8)
    QTest.mousePress(vtk_widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(vtk_widget, moved)
    QTest.mouseRelease(vtk_widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, moved)
    QApplication.processEvents()
    win.rendering_panel.set_camera_info(camera_info)
    QApplication.processEvents()


def apply_main_app_style(app):
    """main.py와 같은 Fusion 스타일 + 다크 팔레트(렌더 창 크기를 수동 실행과 맞추기 위함)."""
    app.setStyle('Fusion')
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    app.setPalette(palette)


def setup_window(dataset):
    """프리셋의 볼륨/INR/TF/카메라/포인트를 올린 메인 윈도우를 만든다(SAM 실행은 안 함).

    Returns: (app, win, volume_data, voxel_spacing)
    """
    paths = DATASETS[dataset]
    app = QApplication.instance() or QApplication(sys.argv)
    apply_main_app_style(app)
    win = VolumeRenderingMainWindow()
    win.show()
    QApplication.processEvents()

    volume_data, voxel_spacing = win.file_panel.volume_loader.load(paths["volume"], paths.get("raw"))
    win.file_panel.volume_data = volume_data
    win.file_panel.voxel_spacing = voxel_spacing
    win.file_panel.volume_name = os.path.basename(paths["volume"])
    win.file_panel.volume_loaded.emit(volume_data)
    QApplication.processEvents()

    inr = np.asarray(nib.load(paths["inr"]).get_fdata(), dtype=np.float32)
    if inr.shape[:3] != volume_data.shape:
        raise ValueError(f"INR shape {inr.shape[:3]}이 volume {volume_data.shape}와 불일치")
    win.inr_feature_volume = inr

    with open(paths["tf"], "r") as f:
        tf_data = json.load(f)
    set_global_tf(win, tf_data["global"])
    if "shading" in tf_data:
        win.rendering_panel.set_shading(tf_data["shading"].get("enabled", False))

    with open(paths["camera"], "r") as f:
        cs = json.load(f)["camera_state"]
    camera_info = {"position": tuple(cs["position"]), "focal_point": tuple(cs["focal_point"]),
                   "view_up": tuple(cs["view_up"]), "view_angle": cs["view_angle"],
                   "distance": cs.get("distance", 0)}
    win.rendering_panel.set_camera_info(camera_info)
    settle_first_render(win, camera_info)

    with open(paths["points"], "r") as f:
        win.optimization_panel.picked_points = json.load(f)
    return app, win, volume_data, voxel_spacing


# ----------------------------------------------------------------------------
def find_gt_path(dataset, volume_path):
    """resources/Volume_Data에서 <dataset>_GT.* 규칙으로 GT 파일을 찾는다."""
    folder = os.path.dirname(volume_path)
    stem = os.path.basename(volume_path)
    for ext in (".nii.gz", ".nii", ".mha", ".raw", ".gz"):
        if stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    for pattern in (f"{stem}_GT.*", f"{dataset}_GT.*", f"{dataset.capitalize()}_GT.*"):
        hits = glob.glob(os.path.join(folder, pattern))
        if hits:
            return hits[0]
    return None


def load_gt(gt_path, volume_shape):
    """GT를 (라벨 후보 목록, 라벨->bool mask 함수, 종류)로 만든다."""
    arr = np.asarray(nib.load(gt_path).dataobj)
    vol3d = tuple(volume_shape[:3])

    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim == 4 and arr.shape[:3] == vol3d:
        cands = [(c, int((arr[..., c] > 0).sum())) for c in range(arr.shape[-1])]
        return cands, (lambda c: arr[..., int(c)] > 0), "채널"

    if arr.ndim == 3 and arr.shape == vol3d:
        vals, counts = np.unique(arr, return_counts=True)
        cands = [(int(v), int(n)) for v, n in zip(vals, counts)]
        return cands, (lambda v: arr == int(v)), "라벨"

    raise ValueError(f"GT shape {arr.shape}이 볼륨 {vol3d}와 맞지 않음")


def evaluate(pred, gt):
    """예측 mask를 GT와 비교한다. Returns: n/dice/iou/precision/recall/components/occupancy."""
    p, g = pred > 0, gt > 0
    inter = int((p & g).sum())
    ps, gs = int(p.sum()), int(g.sum())
    labeled, num = ndimage.label(p, structure=np.ones((3, 3, 3), dtype=bool))
    occ = (ndimage.convolve(p.astype(np.float32), NEIGHBORHOOD, mode="constant")[p] / 27.0
           if ps else np.array([0.0]))
    return {
        "n": ps,
        "dice": 2 * inter / (ps + gs) if (ps + gs) else 0.0,
        "iou": inter / max(int((p | g).sum()), 1),
        "precision": inter / ps if ps else 0.0,
        "recall": inter / gs if gs else 0.0,
        "components": int(num),
        "occupancy": float(occ.mean()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=list(DATASETS), default="tooth")
    parser.add_argument("--gt-path", default=None,
                        help="GT 파일 경로 (생략하면 <볼륨명>_GT.* 규칙으로 찾는다)")
    parser.add_argument("--tf-path", default=None,
                        help="TF json 경로 (생략하면 dataset preset의 기본 TF를 쓴다)")
    parser.add_argument("--gt-label", type=int, default=None,
                        help="비교할 GT 라벨/채널 번호 (생략하면 후보 목록만 출력)")
    parser.add_argument("--inr", action=argparse.BooleanOptionalAction, default=True,
                        help="INR embedding 8채널 사용 여부 (기본 켜짐)")
    parser.add_argument("--coords", action=argparse.BooleanOptionalAction, default=True,
                        help="정규화 좌표 3채널 사용 여부 (기본 켜짐)")
    parser.add_argument("--visibility", action=argparse.BooleanOptionalAction, default=True,
                        help="visibility 사용 여부 (기본 켜짐 — 지금 UI 기본값과 동일)")
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True,
                        help="depth 사용 여부 (기본 켜짐 — 지금 UI 기본값과 동일)")
    parser.add_argument("--intensity", action=argparse.BooleanOptionalAction, default=False,
                        help="local intensity 사용 여부 (기본 꺼짐)")
    parser.add_argument("--opacity", action=argparse.BooleanOptionalAction, default=True,
                        help="local opacity 사용 여부 (기본 켜짐 — 지금 UI 기본값과 동일)")
    parser.add_argument("--stride-floor", type=int, default=None,
                        help="Stage B ray stride 하한을 오버라이드 (생략하면 UI 기본값 2)")
    parser.add_argument("--k", type=int, default=None,
                        help="Stage A cluster 수 (생략하면 UI 기본값 6)")
    parser.add_argument("--seed", type=int, default=None,
                        help="K-means seed (생략하면 UI 기본값 3)")
    parser.add_argument("--visibility-feature", choices=["V", "T"], default="V",
                        help="visibility feature 열 값: V = alpha × 투과율(기본), T = 투과율. 점수는 항상 V")
    parser.add_argument("--inr-path", default=None,
                        help="INR feature .nii 경로 (생략하면 dataset preset의 INR)")
    parser.add_argument("--export-view-targets", default=None,
                        help="Stage B 뒤 voxel별 T(앞쪽 투과율)·alpha·ok를 이 .npz로 저장 (INR view head 학습 정답)")
    parser.add_argument("--save-mask", default=None,
                        help="최종 local mask를 이 .npy로 저장")
    parser.add_argument("--save-render", default=None,
                        help="ROI 등록 뒤 렌더 창을 이 .png로 저장 (GUI에서 보는 화면)")
    parser.add_argument("--save-render-side", default=None,
                        help="--save-render 뒤 카메라를 view up 축으로 90도 돌려 이 .png로 저장 (옆면)")
    parser.add_argument("--save-render-modes", default=None,
                        help="화면 3종을 <값>_{all,roi_off,roi_only}{,_side}.png로 저장 (ROI는 파랑, side는 Azimuth 90도)")
    return parser.parse_args()


def export_view_targets(baker, volume_data, voxel_spacing, path):
    """Stage B ViewFeatureBaker로 모든 voxel의 (T, alpha, ok)를 계산해 저장한다 (Z 슬랩 단위)."""
    X, Y, Z = volume_data.shape
    spacing = np.asarray(voxel_spacing, dtype=np.float64)
    T_all = np.zeros(volume_data.shape, dtype=np.float32)
    a_all = np.zeros(volume_data.shape, dtype=np.float32)
    ok_all = np.zeros(volume_data.shape, dtype=bool)
    ii, jj = np.meshgrid(np.arange(X), np.arange(Y), indexing="ij")
    for z0 in range(0, Z, 16):
        z1 = min(Z, z0 + 16)
        kk = np.arange(z0, z1)
        idx = np.stack(np.broadcast_arrays(ii[..., None], jj[..., None], kk[None, None, :]),
                       axis=-1).reshape(-1, 3)
        p_world = baker.to_world(idx * spacing)
        dx, dy, front = baker.project(p_world)
        T = baker.transmittance(dx, dy, baker.depth(p_world))
        ok = front & np.isfinite(T)
        if baker.clipping_ranges:
            ok &= ~baker.analyzer.is_point_clipped_vectorized(idx * spacing, baker.clipping_ranges)
        shape = (X, Y, z1 - z0)
        T_all[:, :, z0:z1] = np.where(ok, np.nan_to_num(T), 0.0).reshape(shape)
        a_all[:, :, z0:z1] = baker.alpha(volume_data[:, :, z0:z1].reshape(-1)).reshape(shape)
        ok_all[:, :, z0:z1] = ok.reshape(shape)
    np.savez_compressed(path, T=T_all, alpha=a_all, ok=ok_all)
    print(f"view target 저장: {path} — ok {int(ok_all.sum()):,}/{ok_all.size:,} voxel, "
          f"T 평균 {T_all[ok_all].mean():.3f}, alpha>0 {int((a_all > 0).sum()):,} voxel")


def save_render_modes(win, stem):
    """같은 mask로 전체 / ROI opacity 0 / ROI 밖 opacity 0 화면을 정면과 옆면으로 저장한다."""
    panel = win.rendering_panel
    vtk_renderer = panel.vtk_renderer
    camera = vtk_renderer.renderer.GetActiveCamera()

    def set_roi(keep_opacity):
        for lid, nodes in win.local_tf_nodes.items():
            win.local_tf_nodes[lid] = [[n[0], 0.0, 0.0, 1.0, n[4] if keep_opacity else 0.0] for n in nodes]
        win.push_local_probability_maps()
        QApplication.processEvents()

    def shoot(name):
        for suffix, angle in (("", 0), ("_side", 90)):
            camera.Azimuth(angle)
            vtk_renderer.renderer.ResetCameraClippingRange()
            vtk_renderer.render()
            QApplication.processEvents()
            shot = panel.save_current_rendering(return_filename=True)
            if shot:
                os.replace(shot, f"{stem}_{name}{suffix}.png")
            camera.Azimuth(-angle)
        vtk_renderer.renderer.ResetCameraClippingRange()

    original_nodes = {lid: [n[:] for n in nodes] for lid, nodes in win.local_tf_nodes.items()}
    set_roi(True)
    shoot("all")

    set_roi(False)
    shoot("roi_off")

    # ROI 밖: global TF는 두고 shader의 class 0(ROI 밖) 텍스처만 투명하게 바꾼다
    win.local_tf_nodes = {lid: [n[:] for n in nodes] for lid, nodes in original_nodes.items()}
    set_roi(True)
    state = vtk_renderer.shader_texture_state
    transparent = [[n[0], n[1], n[2], n[3], 0.0] for n in vtk_renderer.global_tf_nodes]
    vtk_renderer.shader_manager.update_class_color_opacity_texture(
        vtk_renderer.vtk_widget.GetRenderWindow(), state['color_textures'][0], state['opacity_textures'][0], transparent)
    shoot("roi_only")


def main():
    args = parse_args()
    paths = DATASETS[args.dataset]

    gt_path = args.gt_path or find_gt_path(args.dataset, paths["volume"])
    if not gt_path or not os.path.exists(gt_path):
        print(f"'{args.dataset}'의 GT 파일을 찾지 못했습니다. --gt-path로 지정해주세요.")
        return
    print(f"GT: {gt_path}")

    app = QApplication.instance() or QApplication(sys.argv)
    apply_main_app_style(app)
    win = VolumeRenderingMainWindow()
    win.show()
    QApplication.processEvents()

    volume_data, voxel_spacing = win.file_panel.volume_loader.load(paths["volume"], paths.get("raw"))
    win.file_panel.volume_data = volume_data
    win.file_panel.voxel_spacing = voxel_spacing
    win.file_panel.volume_name = os.path.basename(paths["volume"])
    win.file_panel.volume_loaded.emit(volume_data)
    QApplication.processEvents()

    candidates, get_mask, kind = load_gt(gt_path, volume_data.shape)
    if args.gt_label is None:
        print(f"\n사용 가능한 {kind} (--gt-label 로 하나를 고르세요)")
        for value, count in candidates:
            print(f"   {kind} {value}: {count:,} voxel")
        return
    gt = get_mask(args.gt_label)
    print(f"비교 대상: {kind} {args.gt_label} = {int(gt.sum()):,} voxel\n")

    inr_path = args.inr_path or paths["inr"]
    print(f"INR: {inr_path}")
    inr_data = np.asarray(nib.load(inr_path).get_fdata(), dtype=np.float32)
    if inr_data.shape[:3] != volume_data.shape:
        raise ValueError(f"INR shape {inr_data.shape[:3]}이 volume {volume_data.shape}와 불일치")
    win.inr_feature_volume = inr_data

    tf_path = args.tf_path or paths["tf"]
    print(f"TF: {tf_path}")
    with open(tf_path, "r") as f:
        tf_data = json.load(f)
    global_tf = tf_data["global"]
    set_global_tf(win, global_tf)
    if "shading" in tf_data:
        win.rendering_panel.set_shading(tf_data["shading"].get("enabled", False))

    with open(paths["camera"], "r") as f:
        cs = json.load(f)["camera_state"]
    camera_info = {
        "position": tuple(cs["position"]), "focal_point": tuple(cs["focal_point"]),
        "view_up": tuple(cs["view_up"]), "view_angle": cs["view_angle"],
        "distance": cs.get("distance", 0),
    }
    win.rendering_panel.set_camera_info(camera_info)
    settle_first_render(win, camera_info)

    with open(paths["points"], "r") as f:
        win.optimization_panel.picked_points = json.load(f)
    print(f"SAM 예측 중 (최대 {SAM_TIMEOUT_MS / 1000:.0f}초)...", flush=True)
    win.on_check_sam()
    wait_for_signal(win.sam_model.predicted, timeout_ms=SAM_TIMEOUT_MS)
    QApplication.processEvents()
    if win.current_sam_mask is None:
        raise RuntimeError("SAM 예측 실패")

    vtk_matrix = win.rendering_panel.vtk_renderer.standard_volume.GetMatrix()
    model_to_world = np.array([vtk_matrix.GetElement(i, j)
                               for i in range(4) for j in range(4)]).reshape(4, 4)

    win.wysiwyg_panel.feature_inr_check.setChecked(args.inr)
    win.wysiwyg_panel.feature_coords_check.setChecked(args.coords)
    win.wysiwyg_panel.local_feature_checks["visibility"].setChecked(args.visibility)
    win.wysiwyg_panel.local_feature_checks["depth"].setChecked(args.depth)
    win.wysiwyg_panel.local_feature_checks["intensity"].setChecked(args.intensity)
    win.wysiwyg_panel.local_feature_checks["opacity"].setChecked(args.opacity)
    combo = win.wysiwyg_panel.visibility_feature_combo
    combo.setCurrentIndex(combo.findData(args.visibility_feature))
    if args.k is not None:
        win.wysiwyg_panel.cluster_k_spin.setValue(args.k)
    if args.seed is not None:
        win.wysiwyg_panel.cluster_seed_spin.setValue(args.seed)
    if args.stride_floor is not None:
        win.wysiwyg_panel.stage_b_stride_spin.setValue(args.stride_floor)

    parts = [n for n, on in [("inr", args.inr), ("coords", args.coords),
                            ("visibility", args.visibility), ("depth", args.depth),
                            ("intensity", args.intensity), ("opacity", args.opacity)] if on]
    file_slug = "_".join(parts) if parts else "none"

    header = (f"\n{'단계':<11} {'voxel':>8} {'Dice':>6} {'IoU':>6} "
              f"{'정밀도':>7} {'재현율':>7} {'성분':>5} {'occ':>6}")
    print(f"feature: {file_slug}  k: {win.wysiwyg_panel.get_cluster_k()}  "
          f"visibility 열: {args.visibility_feature}")
    print(header)
    print("─" * len(header))

    t0 = time.time()
    try:
        win.on_clustering_clicked()
        QApplication.processEvents()
    except Exception as e:
        print(f"clustering 실패: {e}")
        return
    print(f"⏱️ on_clustering_clicked 총 소요 시간: {time.time() - t0:.2f}초")

    roi_info = win.current_roi_info
    cluster_result = roi_info.get("cluster_result") if roi_info else None
    if cluster_result is None:
        print("clustering 실패 (cluster_result 없음)")
        return

    stage_a = stage_a_best_cluster_voxels(
        cluster_result, roi_info, volume_data.shape, voxel_spacing,
        model_to_world=model_to_world)
    if stage_a is not None:
        r = evaluate(stage_a, gt)
        print(f"{'Stage A만':<11} {r['n']:>8,} {r['dice']:>6.3f} {r['iou']:>6.3f} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['components']:>5} {r['occupancy']:>6.3f}")

    if win.current_local_mask is None:
        print("Stage B 실패 (current_local_mask 없음)")
        return
    mask, debug_info = win.current_local_mask
    r = evaluate(mask, gt)
    print(f"{'+Stage B':<11} {r['n']:>8,} {r['dice']:>6.3f} {r['iou']:>6.3f} "
          f"{r['precision']:>7.3f} {r['recall']:>7.3f} "
          f"{r['components']:>5} {r['occupancy']:>6.3f}")
    if args.save_mask:
        np.save(args.save_mask, np.asarray(mask, dtype=bool))
        print(f"mask 저장: {args.save_mask}")
    if args.save_render_modes:
        save_render_modes(win, args.save_render_modes)
    if args.save_render:
        # 결과 ROI를 파랑으로 칠해 global TF 색과 구분 (opacity는 그대로)
        for lid, nodes in win.local_tf_nodes.items():
            win.local_tf_nodes[lid] = [[n[0], 0.0, 0.0, 1.0, n[4]] for n in nodes]
        win.push_local_probability_maps()
        QApplication.processEvents()
        shot = win.rendering_panel.save_current_rendering(return_filename=True)
        if shot:
            os.replace(shot, args.save_render)
        if args.save_render_side:
            vtk_renderer = win.rendering_panel.vtk_renderer
            vtk_renderer.renderer.GetActiveCamera().Azimuth(90)
            vtk_renderer.renderer.ResetCameraClippingRange()
            vtk_renderer.render()
            QApplication.processEvents()
            shot = win.rendering_panel.save_current_rendering(return_filename=True)
            if shot:
                os.replace(shot, args.save_render_side)
    if args.export_view_targets:
        export_view_targets(win._stage_b_view_cache[1], volume_data, voxel_spacing,
                            args.export_view_targets)

    print("\nDice/IoU는 높을수록 좋다. 정밀도가 1에 가까운데 재현율이 낮으면 "
          "'맞게 잡지만 많이 놓치는' 상태다.")


if __name__ == "__main__":
    main()
