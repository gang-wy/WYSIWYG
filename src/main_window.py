import json
import os
import time
from datetime import datetime
from PIL import Image
import numpy as np

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from src.gui.panel.file_panel import FilePanel
from src.gui.panel.tf_panel import TransferFunctionPanel
from src.gui.panel.rendering_panel import RenderingPanel
from src.gui.panel.optimization_panel import OptimizationPanel
from src.gui.panel.wysiwyg_panel import WysiwygPanel
from src.gui.widget.slice_debug_viewer import SliceDebugViewer

from src.core.sam.support_sam import SAMService
from src.core.clustering import sampling_rules
from src.core.clustering.ray_sampler import FeatureAnalyzer
from src.core.clustering.feature_clustering import ROIFeatureExtractor
from src.core.clustering.volume_classification import (
    classify_voxels_view_features, sample_label_points, voxel_indices_to_world,
    train_stage_b_forest, compose_final_labels, split_best_by_space)
from src.core.clustering.view_feature_baker import ViewFeatureBaker
from src.core.clustering.clustering_diagnostics import format_stage_a, format_stage_b, mask_shape_summary
from src.core.wysiwyg_tf_editor import WysiwygTFEditor
from src.core.dataset_presets import DATASET_PRESETS
from src.core.stroke_sampling import sample_stroke_points, STROKE_SAM_POINT_COUNT

from src.gui.wysiwyg.eraser_tool import preview_eraser
from src.gui.wysiwyg.brightness_tool import preview_brightness
from src.gui.wysiwyg.color_tool import preview_colorization
from src.gui.wysiwyg.contrast_tool import preview_contrast


def actor_matrix(actor):
    """vtkProp3D의 model→world 변환 행렬. Returns: (4,4) ndarray"""
    m = actor.GetMatrix()
    return np.array([m.GetElement(i, j) for i in range(4) for j in range(4)]).reshape(4, 4)


class VolumeRenderingMainWindow(QMainWindow):
    """패널 배치, 패널 간 시그널 연결, SAM → Clustering(Stage A/B) → Local TF 워크플로우를 담당한다."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Rendering & Optimization Tool")
        self.setGeometry(100, 100, 2200, 900)    # 가로는 패널 구성 후 fit_window_width가 정한다

        self.volume_data = None
        self.voxel_spacing = (1.0, 1.0, 1.0)
        self.inr_feature_volume = None       # (X, Y, Z, C)

        self.sam_model = SAMService()
        self.sam_model.loaded.connect(lambda: print("SAM loaded"))
        self.sam_model.predicted.connect(self.on_sam_mask_ready)
        self.sam_model.error.connect(self.on_sam_error)
        self.sam_model.load_async()
        self.current_sam_mask = None         # Check SAM 결과 (H, W)
        self._sam_request = None             # 진행 중인 Check SAM 요청의 point·캡처 이미지(덤프 기록용)
        self.current_sam_prompt = None       # current_sam_mask를 만든 point·캡처 이미지·mask 경로

        # Stage A 결과
        self.current_roi_info = None
        self.current_kmeans_world_points = None   # "K-means" 오버레이: ROI 샘플 world 좌표
        self.current_kmeans_labels = None
        self.current_best_cluster = None          # 오버레이에서 고정색으로 칠할 cluster
        self.current_rf_points = None             # "RF samples" 오버레이: (world 좌표, rgb)
        # Stage B 결과
        self.current_cluster_world_points = None  # "Final" 오버레이: 라벨 voxel 일부의 world 좌표
        self.current_cluster_labels = None
        self.current_cluster_label_volume = None  # (X, Y, Z) int16, slice viewer용
        self.current_p_best = None                # (X, Y, Z) float16 RF P(best), -1 = 판정 불가
        self._stage_b_view_cache = None           # (key, ViewFeatureBaker)

        # Global TF
        self.wysiwyg_tf_editor = WysiwygTFEditor()
        self.base_tf_nodes = None
        self.preview_tf_nodes = None

        # Local class(다중 ROI). 모든 dict의 key는 label_id, local_masks는 항상 서로 disjoint.
        self.local_masks = {}                # {label_id: (X,Y,Z) bool}
        self.local_mask_debug = {}           # {label_id: Stage B debug_info}
        self.local_roi_infos = {}            # {label_id: 생성 시점 roi_info}
        self.local_tf_nodes = {}             # {label_id: 확정된 local TF}
        self.local_preview_tf_nodes = {}     # {label_id: 미리보기 중인 local TF}
        self.local_tf_view_ranges = {}       # {label_id: (low, high)} TF 위젯 가로축 범위
        self.local_mask_active = False
        self.current_local_mask = None       # 가장 최근 class의 (mask_uint8, debug_info)

        self.slice_debug_viewer = None

        # Experiment 촬영 (TF 패널 Experiment 그룹)
        self.current_preset_name = None
        self.current_roi_name = None          # 프리셋 "rois"에서 고른 ROI
        self.current_camera_name = "main"     # "main" | "another"
        self.view_mode = "default"            # TransferFunctionPanel.VIEW_MODES 키
        self._last_stage_a_settings = None    # 직전 Stage A 실행의 UI 설정 (촬영 파일 이름용)

        self.init_ui()
        self.create_panels()
        self.connect_signals()
        self.fit_window_width()

        self.statusBar().showMessage("Ready - Load volume data to start")
        self.tf_panel.reset_background_color()

        self.is_updating_tf_programmatically = False
        self.preview_active = False
        self.base_tf_nodes = self.get_current_tf_nodes()

    # ------------------------------------------------------------------
    # UI 구성
    # ------------------------------------------------------------------
    def init_ui(self):
        """좌(파일/TF) · 중앙(렌더링) · 우(SAM + WYSIWYG) 3단 레이아웃을 만든다."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)

        left_container = QWidget()
        left_container.setFixedWidth(600)
        left_container.setStyleSheet("background-color: #2d2d2d;")

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #2d2d2d; }")

        scroll_content = QWidget()
        self.left_layout = QVBoxLayout(scroll_content)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(scroll_content)

        left_main_layout = QVBoxLayout(left_container)
        left_main_layout.addWidget(scroll_area)
        main_layout.addWidget(left_container)

        self.center_widget = QWidget()
        self.center_layout = QVBoxLayout(self.center_widget)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.center_widget, stretch=2)

        self.right_outer_container = QWidget()
        self.right_outer_layout = QHBoxLayout(self.right_outer_container)
        self.right_outer_layout.setContentsMargins(0, 0, 0, 0)
        self.right_outer_layout.setSpacing(10)

        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_outer_layout.addWidget(self.right_container, stretch=3)

        main_layout.addWidget(self.right_outer_container, stretch=3)

    def create_panels(self):
        self.file_panel = FilePanel()
        self.left_layout.addWidget(self.file_panel)

        self.tf_panel = TransferFunctionPanel()
        self.left_layout.addWidget(self.tf_panel)
        self.left_layout.addStretch(1)

        self.rendering_panel = RenderingPanel()
        self.center_layout.addWidget(self.rendering_panel)

        self.optimization_panel = OptimizationPanel()
        self.right_layout.addWidget(self.optimization_panel)

        self.wysiwyg_panel = WysiwygPanel(parent=self)
        self.wysiwyg_scroll_area = QScrollArea()
        self.wysiwyg_scroll_area.setWidgetResizable(True)
        self.wysiwyg_scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #2d2d2d; }")

        self.wysiwyg_scroll_content = QWidget()
        self.wysiwyg_scroll_layout = QVBoxLayout(self.wysiwyg_scroll_content)
        self.wysiwyg_scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.wysiwyg_scroll_layout.addWidget(self.wysiwyg_panel)
        self.wysiwyg_scroll_layout.addStretch(1)
        self.wysiwyg_scroll_area.setWidget(self.wysiwyg_scroll_content)
        self.right_outer_layout.addWidget(self.wysiwyg_scroll_area, stretch=4)

    RENDER_AREA_WIDTH = 628     # 렌더 영역 폭(VTK 창 610px). SAM 캡처 크기와 Stage A 샘플 수가 창 크기에 따라 바뀌지 않도록 고정
    WINDOW_HEIGHT = 900

    def fit_window_width(self):
        """오른쪽 패널이 줄어들지 않는 폭으로 창 가로 길이를 정한다(높이는 고정).

        - SAM 패널: Set Mode의 Positive/Negative/Line 버튼까지 보일 때의 폭으로 고정한다.
          그래서 Set Mode를 켜고 꺼도 옆 WYSIWYG 패널이 밀리지 않는다.
        - WYSIWYG 패널: 내용 최소 폭 + 세로 스크롤바 폭으로 고정하고 가로 스크롤을 끈다.
        - 렌더 영역: RENDER_AREA_WIDTH로 고정한다.
        """
        opt = self.optimization_panel
        mode_buttons = (opt.pos_btn, opt.neg_btn, opt.line_btn)
        for btn in mode_buttons:
            btn.setVisible(True)
        sam_width = self.right_container.minimumSizeHint().width()
        for btn in mode_buttons:
            btn.setVisible(opt.set_btn.isChecked())
        self.right_container.setFixedWidth(sam_width)

        area = self.wysiwyg_scroll_area
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        area.setFixedWidth(self.wysiwyg_scroll_content.minimumSizeHint().width()
                           + area.verticalScrollBar().sizeHint().width() + 2 * area.frameWidth())

        self.center_widget.setFixedWidth(self.RENDER_AREA_WIDTH)
        self.setFixedSize(self.centralWidget().minimumSizeHint().width(), self.WINDOW_HEIGHT)

    def connect_signals(self):
        self.file_panel.volume_loaded.connect(self.on_volume_loaded)
        self.file_panel.inr_features_loaded.connect(self.on_inr_features_loaded)

        self.tf_panel.tf_changed.connect(self.on_tf_changed)
        self.tf_panel.background_color_changed.connect(self.on_background_color_changed)
        self.tf_panel.shading_changed.connect(self.on_shading_changed)
        self.tf_panel.lighting_changed.connect(self.on_lighting_changed)
        self.tf_panel.light_direction_changed.connect(self.on_light_direction_changed)
        self.tf_panel.follow_camera_changed.connect(self.on_follow_camera_changed)
        self.tf_panel.ambient_color_changed.connect(self.on_ambient_color_changed)
        self.tf_panel.diffuse_color_changed.connect(self.on_diffuse_color_changed)
        self.tf_panel.specular_color_changed.connect(self.on_specular_color_changed)
        self.tf_panel.clipping_changed.connect(self.on_clipping_changed)
        self.tf_panel.clipping_enabled_changed.connect(self.on_clipping_enabled_changed)
        self.tf_panel.target_selected.connect(self.on_tf_target_selected)
        self.tf_panel.dataset_preset_requested.connect(self.load_dataset_preset)
        self.tf_panel.local_class_delete_requested.connect(self.delete_local_class)
        self.tf_panel.experiment_roi_requested.connect(self.load_experiment_roi)
        self.tf_panel.experiment_camera_requested.connect(self.load_experiment_camera)
        self.tf_panel.view_mode_requested.connect(self.set_view_mode)
        self.tf_panel.capture_requested.connect(self.capture_experiment_view)
        self.tf_panel.roi_color_changed.connect(self.apply_roi_color)

        self.optimization_panel.set_mode_changed.connect(self.on_set_mode_changed)
        self.optimization_panel.point_type_changed.connect(self.on_point_type_changed)
        self.optimization_panel.check_sam.connect(self.on_check_sam)
        self.optimization_panel.points_updated.connect(self.on_points_updated)
        self.optimization_panel.run_clustering_requested.connect(self.on_clustering_clicked)

        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.point_2d_picked.connect(self.on_point_2d_picked)
            self.rendering_panel.vtk_renderer.stroke_completed.connect(self.on_stroke_completed)

        self.wysiwyg_panel.preview_requested.connect(self.on_wysiwyg_preview_clicked)
        self.wysiwyg_panel.apply_requested.connect(self.apply_wysiwyg_preview)
        self.wysiwyg_panel.reset_requested.connect(self.reset_wysiwyg_preview)
        self.wysiwyg_panel.cluster_overlay_toggled.connect(self.on_cluster_overlay_toggled)
        self.wysiwyg_panel.kmeans_overlay_toggled.connect(self.on_kmeans_overlay_toggled)
        self.wysiwyg_panel.rf_overlay_toggled.connect(self.on_kmeans_overlay_toggled)   # 동작이 같음: 켜면 갱신, 끄면 지움
        self.wysiwyg_panel.cluster_view_changed.connect(self.on_cluster_view_changed)
        self.wysiwyg_panel.cluster_color_changed.connect(self.on_cluster_color_changed)
        self.wysiwyg_panel.slice_viewer_requested.connect(self.on_slice_viewer_requested)

    # ------------------------------------------------------------------
    # SAM 포인트 입력
    # ------------------------------------------------------------------
    def on_set_mode_changed(self, enabled):
        """Set 모드 전환: 찍은 포인트를 비우고 렌더러를 피킹(ON) / 카메라 조작(OFF) 모드로 바꾼다."""
        self.optimization_panel.picked_points = []
        self.optimization_panel.points_updated.emit([])

        renderer = self.rendering_panel.vtk_renderer
        if renderer:
            renderer.clear_overlay_points()
            self.rendering_panel.set_overlay_visible(enabled)
            renderer.set_interaction_enabled(not enabled)
            renderer.set_picking_enabled(enabled)

        self.statusBar().showMessage("SET MODE: ON (Pick Points)" if enabled else "SET MODE: OFF (Navigation)")

    def on_point_type_changed(self, p_type):
        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.current_pick_type = p_type

    def on_point_2d_picked(self, vtk_x, vtk_y):
        """렌더러에서 찍은 점(VTK display 좌표)을 포인트 목록과 화면 오버레이에 추가한다."""
        p_type = self.optimization_panel.current_point_type
        self.optimization_panel.picked_points.append({'pos': (vtk_x, vtk_y), 'type': p_type})
        self.rendering_panel.add_point_2d(vtk_x, vtk_y, p_type)
        print(f"Point Added: {vtk_x}, {vtk_y} ({p_type})")

    def on_stroke_completed(self, stroke):
        """Line 모드 스트로크 → Unity와 같은 규칙으로 positive 점 STROKE_SAM_POINT_COUNT개(이전 점 교체)."""
        samples = sample_stroke_points(stroke, STROKE_SAM_POINT_COUNT)
        if not samples:
            return
        points = [{'pos': (int(round(x)), int(round(y))), 'type': 'positive'} for x, y in samples]
        self.optimization_panel.picked_points = points
        renderer = self.rendering_panel.vtk_renderer
        renderer.point_overlay.clear_points()
        renderer.point_overlay.set_stroke(stroke)
        for pt in points:
            renderer.point_overlay.add_point(*pt['pos'], 'positive')
        renderer.render()
        print(f"Stroke: {len(stroke)}개 점 → SAM positive {len(points)}개 {[p['pos'] for p in points]}")

    def on_points_updated(self, points_list):
        """포인트 목록이 Load/Clear로 바뀌면 화면 오버레이를 다시 그린다.

        points_list: [{'pos': (x, y), 'type': 'positive' | 'negative'}, ...]
        """
        if not self.rendering_panel.vtk_renderer:
            return

        self.rendering_panel.clear_overlay()
        for pt in points_list:
            x, y = pt['pos'][0], pt['pos'][1]
            self.rendering_panel.add_point_2d(x, y, pt['type'])
        self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow().Render()
        print(f"🔄 View Updated: {len(points_list)} points loaded/cleared.")

    # ------------------------------------------------------------------
    # SAM
    # ------------------------------------------------------------------
    def on_check_sam(self):
        """현재 화면을 캡처하고 찍은 포인트로 SAM 예측을 비동기 요청한다."""
        points_data = self.optimization_panel.picked_points
        if not points_data:
            QMessageBox.warning(self, "Warning", "선택된 포인트가 없습니다. Set Mode에서 포인트를 먼저 찍어주세요.")
            return

        self.statusBar().showMessage("Capturing Scene & Running SAM... Please wait.")
        QApplication.processEvents()

        image_path = self.rendering_panel.save_current_rendering(return_filename=True)
        if not image_path:
            self.statusBar().showMessage("Error: Failed to capture screenshot.")
            return

        img_width, img_height = Image.open(image_path).size

        input_points, input_labels = [], []
        for pt in points_data:
            vtk_x, vtk_y = pt['pos']
            # VTK(좌하단 원점) → 이미지(좌상단 원점)
            img_x = max(0, min(int(vtk_x), img_width - 1))
            img_y = max(0, min(int(img_height - vtk_y), img_height - 1))
            input_points.append([img_x, img_y])
            input_labels.append(1 if pt['type'] == 'positive' else 0)

        self._sam_request = {
            "vtk": [list(pt['pos']) for pt in points_data], "image": input_points,
            "types": [pt['type'] for pt in points_data], "image_path": image_path,
        }
        self.sam_model.predict_async(image_path, input_points, input_labels)

    def on_sam_mask_ready(self, result):
        """SAM 결과 수신: mask를 저장·표시하고 이전 Stage A/B 결과를 초기화한다(local class는 유지)."""
        mask = result[0] if isinstance(result, tuple) else result
        if mask is None:
            self.statusBar().showMessage("SAM Result: No mask generated.")
            return

        save_dir = "./resources/masks"
        os.makedirs(save_dir, exist_ok=True)
        file_name = f"mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        mask_path = os.path.join(save_dir, file_name)
        try:
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            self.optimization_panel.set_image_by_path(mask_path)
        except Exception as e:
            print(f"❌ Failed to save mask: {e}")
            self.statusBar().showMessage("Error saving mask file.")

        self.current_sam_mask = mask
        self.current_sam_prompt = dict(self._sam_request or {}, mask_path=mask_path)
        self.optimization_panel.set_clustering_available(True)
        self.statusBar().showMessage(f"Mask saved: {file_name}. 'Run Clustering'을 눌러 ROI를 분석하세요.")

        self.current_roi_info = None
        self.preview_tf_nodes = None
        self.preview_active = False
        self.wysiwyg_panel.set_roi_selected(False)
        self.wysiwyg_panel.set_preview_active(False)
        self.tf_panel.tf_widget.clear_highlight_range()

        self.current_cluster_world_points = None
        self.current_cluster_labels = None
        self.current_cluster_label_volume = None
        self.current_p_best = None
        self.current_kmeans_world_points = None
        self.current_kmeans_labels = None
        self.current_best_cluster = None
        self.current_rf_points = None
        self.wysiwyg_panel.clear_cluster_view_options()
        self.rendering_panel.clear_roi_actor()

    def on_sam_error(self, msg):
        print("SAM error:", msg)

    # ------------------------------------------------------------------
    # Clustering (Stage A → Stage B)
    # ------------------------------------------------------------------
    def get_clipping_ranges(self):
        return self.tf_panel.get_clipping_ranges() if self.tf_panel else None

    def on_clustering_clicked(self):
        """Stage A: SAM mask 픽셀(stride 5)로 ray 샘플을 만들어 K-means로 ROI를 고른다.

        성공하면 roi_info/K-means 오버레이 데이터를 갱신하고 Stage B(auto_build_local_mask)를 이어서 실행한다.
        """
        mask = self.current_sam_mask
        if mask is None:
            self.statusBar().showMessage("먼저 'Check SAM'으로 마스크를 생성하세요.")
            return

        self.statusBar().showMessage("Running clustering...")
        QApplication.processEvents()
        # Stage A와 이어지는 Stage B가 같은 visibility 정의를 쓰도록 clustering 시작 때만 바꾼다
        sampling_rules.VISIBILITY_FEATURE = self.wysiwyg_panel.get_visibility_feature()
        print(f"[visibility feature] {sampling_rules.VISIBILITY_FEATURE}")

        try:
            renderer = self.rendering_panel.vtk_renderer
            h, w = mask.shape
            sw, sh = renderer.vtk_widget.GetRenderWindow().GetSize()
            sx, sy = sw / w, sh / h

            y_idx, x_idx = np.where(mask > 0)
            stride = 5
            print(f"[SAM mask] {mask.shape}  {len(y_idx)}px → stride{stride} {len(y_idx[::stride])} ray")
            coords = np.column_stack((x_idx[::stride] * sx, (h - y_idx[::stride]) * sy))

            analyzer = FeatureAnalyzer(
                renderer.renderer, self.volume_data, self.voxel_spacing,
                volume_actor=renderer.standard_volume,
                inr_feature_volume=self.inr_feature_volume)

            # opacity는 항상 global TF 기준 (선택창이 Local N이어도)
            tf_lut = self.tf_panel.tf_widget.get_opacity_lut_for_nodes(self.base_tf_nodes)
            clipping_ranges = self.get_clipping_ranges()
            if clipping_ranges:
                print(f"[clipping] {clipping_ranges}")

            results = analyzer.analyze_roi_profile(coords, tf_lut, clipping_ranges=clipping_ranges)
            if len(results.get('sampled_intensities', [])) == 0:
                self.statusBar().showMessage("Clustering: 유효한 ray 샘플이 없습니다.")
                return

            percentile_low, percentile_high = self.wysiwyg_panel.get_percentile_range()
            groups = self.wysiwyg_panel.get_feature_groups()
            extractor = ROIFeatureExtractor(
                volume_range=(self.volume_data.min(), self.volume_data.max()),
                k=self.wysiwyg_panel.get_cluster_k(),
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                use_inr=groups["inr"],
                use_coords=groups["coords"],
                use_visibility=groups["visibility"],
                use_depth=groups["depth"],
                use_intensity=groups["intensity"],
                use_opacity=groups["opacity"],
                seed=self.wysiwyg_panel.get_cluster_seed(),
                sparse_minmax=self.wysiwyg_panel.get_sparse_minmax(),
                n_init=self.wysiwyg_panel.get_cluster_n_init(),
            )
            roi_info = extractor.extract(results)
            self.current_roi_info = roi_info
            self.preview_tf_nodes = None
            self.preview_active = False
            self.wysiwyg_panel.set_roi_selected(roi_info is not None)
            self.wysiwyg_panel.set_preview_active(False)

            if roi_info is None:
                print("⚠️ roi_info is None")
                self.tf_panel.tf_widget.clear_highlight_range()
                return
            self.tf_panel.tf_widget.set_highlight_range(roi_info["range_norm"])

            # "K-means" 오버레이 데이터는 여기서, "Final"은 Stage B에서 채운다
            cluster_result = roi_info.get('cluster_result')
            self.current_cluster_world_points = None
            self.current_cluster_labels = None
            self.current_cluster_label_volume = None
            self.current_p_best = None
            self.current_rf_points = None
            if cluster_result is not None:
                self.current_kmeans_world_points = roi_info.get('filtered_points')
                self.current_kmeans_labels = cluster_result['labels']
                self.current_best_cluster = cluster_result['best_cluster_index']
                self.wysiwyg_panel.set_cluster_view_options(
                    cluster_result['k'], cluster_result['best_cluster_index'])
                self.on_cluster_view_changed()
            else:
                self.current_kmeans_world_points = None
                self.current_kmeans_labels = None
                self.current_best_cluster = None
                self.wysiwyg_panel.clear_cluster_view_options()
                self.rendering_panel.clear_roi_actor()

            cluster_intens = roi_info['cluster_intensities']
            print(f"\n[ROI] 샘플 {roi_info['num_samples']}개  "
                  f"intensity 범위(best cluster, p{percentile_low:.0f}~p{percentile_high:.0f}) "
                  f"{roi_info['range_norm'][0]:.4f}~{roi_info['range_norm'][1]:.4f}"
                  f"  (trim 전 {cluster_intens.min():.4f}~{cluster_intens.max():.4f})")

            if cluster_result is not None:
                print(format_stage_a(cluster_result))
                if self.wysiwyg_panel.get_best_spatial_split():
                    split = split_best_by_space(
                        cluster_result, roi_info, self.volume_data.shape, self.voxel_spacing,
                        actor_matrix(self.rendering_panel.vtk_renderer.standard_volume))
                    print(f"[Stage A 후처리] best 공간 분리(샘플 간격 {split['spacing']:.2f}mm): "
                          f"성분 {split['components']}개 → 주 성분 샘플 "
                          f"{split['main']:,}개만 best, 떨어진 {split['detached']:,}개는 제외")
                self.dump_stage_a(roi_info)
                self.auto_build_local_mask(cluster_result, tf_lut)
            else:
                print("[Stage A] clustering 결과 없음 (INR 미로드 등)")
                self.statusBar().showMessage(
                    "Clustering completed but no cluster_result (INR/Coords feature를 켜고 다시 시도하세요).")

            if self.slice_debug_viewer is not None and self.slice_debug_viewer.isVisible():
                self.refresh_slice_viewer_sources()

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Clustering failed: {e}")

    def _get_stage_b_view_baker(self, volume_actor, tf_lut, floor_stride):
        """현재 view용 ViewFeatureBaker를 반환한다.

        stride = max(적응형 stride, floor_stride). 카메라·actor·TF·clipping·창 크기·stride가
        이전과 같으면 캐시된 baker를 재사용한다. 캐시 key는 self._stage_b_view_cache[0]에 남는다.
        """
        renderer = self.rendering_panel.vtk_renderer
        camera = renderer.renderer.GetActiveCamera()
        sw, sh = renderer.vtk_widget.GetRenderWindow().GetSize()
        clipping_ranges = self.get_clipping_ranges()

        analyzer = FeatureAnalyzer(renderer.renderer, self.volume_data, self.voxel_spacing,
                                   volume_actor=volume_actor)
        adaptive_stride = analyzer.compute_adaptive_stride(sh)
        stride = max(adaptive_stride, floor_stride)

        key = (stride, sw, sh, camera.GetMTime(),
               volume_actor.GetMTime() if volume_actor else None,
               hash(np.asarray(tf_lut).tobytes()), repr(clipping_ranges))
        if self._stage_b_view_cache is not None and self._stage_b_view_cache[0] == key:
            print("[Stage B] 투과율 격자 캐시 재사용")
            return self._stage_b_view_cache[1]

        t0 = time.time()
        baker = ViewFeatureBaker(renderer.renderer, volume_actor, (sw, sh),
                                 self.volume_data, self.voxel_spacing, tf_lut,
                                 grid_stride=stride, clipping_ranges=clipping_ranges)
        print(f"⏱️ [Stage B] 투과율 격자: window {sw}x{sh}, stride={stride}"
              f"(적응형 {adaptive_stride}, floor {floor_stride}) → {baker.nx}x{baker.ny} ray, "
              f"{time.time() - t0:.2f}초")
        self._stage_b_view_cache = (key, baker)
        return baker

    STAGE_A_DUMP_DIR = "resources/Experiments/stage_a_dump"

    def stage_a_ui_settings(self):
        """Stage A 결과에 영향을 주는 UI 설정(덤프 기록용). 키 이름 = npz 키 이름."""
        panel = self.wysiwyg_panel
        p_low, p_high = panel.get_percentile_range()
        groups = panel.get_feature_groups()
        return {
            "k": int(panel.get_cluster_k()),
            "seed": int(panel.get_cluster_seed()),
            "kmeans_n_init": int(panel.get_cluster_n_init()),
            "percentile_low": float(p_low), "percentile_high": float(p_high),
            **{f"use_{name}": bool(on) for name, on in groups.items()},   # use_inr, use_coords, … use_opacity
            "sparse_minmax": bool(panel.get_sparse_minmax()),
            "visibility_feature": str(panel.get_visibility_feature()),   # "V" 또는 "T"
            "best_spatial_split": bool(panel.get_best_spatial_split()),
            "tf_nodes": self._tf_nodes_for_dump(self.base_tf_nodes),     # global TF 노드 [x, r, g, b, a]
            "clipping_ranges": json.dumps(self.get_clipping_ranges(), default=str),
        }

    @staticmethod
    def _tf_nodes_for_dump(nodes):
        """TF 노드를 float 배열로(형식이 달라 변환이 안 되면 JSON 문자열로 — 덤프 전체가 실패하지 않게)."""
        try:
            return np.asarray(nodes if nodes is not None else [], dtype=float)
        except (TypeError, ValueError):
            return json.dumps(nodes, default=str)

    def stage_b_ui_settings(self):
        """Stage B 결과에 영향을 주는 UI 설정(덤프 기록용)."""
        rf_trees, rf_margin, rf_min_prob, rf_inr_only = self.wysiwyg_panel.get_stage_b_rf()
        return {
            "stage_b_stride_floor": int(self.wysiwyg_panel.get_stage_b_ray_stride_floor()),
            "rf_trees": int(rf_trees), "rf_margin": int(rf_margin),
            "rf_min_prob": float(rf_min_prob), "rf_inr_only": bool(rf_inr_only),
        }

    def sam_prompt_for_dump(self):
        """이 Stage A의 SAM mask를 만든 point(VTK display 좌표 / 캡처 이미지 좌표, 종류)와 캡처·mask 파일 경로.

        point는 Check SAM 시점에 고정한 값이라 그 뒤에 point를 바꿔도 덤프에는 실제 SAM 입력이 남는다.
        VTK 좌표는 📥 Load Pts JSON의 'pos'와 같은 좌표계다.
        """
        p = self.current_sam_prompt or {}
        n = len(p.get("vtk", []))
        return {
            "sam_points_vtk": np.asarray(p.get("vtk", []), dtype=np.int32).reshape(n, 2),
            "sam_points_image": np.asarray(p.get("image", []), dtype=np.int32).reshape(n, 2),
            "sam_point_types": np.asarray(p.get("types", []), dtype="U8"),
            "sam_image_path": str(p.get("image_path", "") or ""),
            "sam_mask_path": str(p.get("mask_path", "") or ""),
        }

    def dump_stage_a(self, roi_info, extra=None):
        """진단용: Stage A 샘플(K-means 입력 정규화 feature, 라벨, world 좌표 등)과 UI 설정을 .npz로 저장한다.

        extra: 덤프에 더 넣을 키(예: Run 깊이 구간의 stage_a_method, 구간 설정).
        """
        try:
            cr = roi_info["cluster_result"]
            os.makedirs(self.STAGE_A_DUMP_DIR, exist_ok=True)
            path = os.path.join(self.STAGE_A_DUMP_DIR,
                                f"stage_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
            np.savez_compressed(
                path,
                normalized=cr["normalized"], labels=cr["labels"], centroids=cr["centroids"],
                feat_mean=cr["feat_mean"], feat_std=cr["feat_std"],
                layout=np.array([f"{n}:{s}:{e}" for n, s, e in cr["feature_layout"]]),
                scores=cr["scores"], best=cr["best_cluster_index"],
                best_cluster_mask=cr["best_cluster_mask"],
                detached_best_mask=cr.get("detached_best_mask", np.zeros(len(cr["labels"]), dtype=bool)),
                world_points=roi_info["filtered_points"], intensities=roi_info["filtered_intensities"],
                visibility=roi_info["filtered_visibility"], opacity=roi_info["filtered_opacity"],
                depths=roi_info["filtered_depths"], voxel_spacing=np.asarray(self.voxel_spacing, dtype=float),
                **self.stage_a_ui_settings(),   # seed, sparse_minmax 포함 UI 설정 전부
                # 분석 스크립트(scripts/compare_stage_a_slices.py 등)가 데이터셋을 스스로 알아내도록 기록
                volume_path=str(getattr(self.file_panel, "volume_path", "") or ""),
                raw_params=json.dumps(getattr(self.file_panel, "raw_params", None) or {}, default=str),
                inr_path=str(getattr(self.file_panel, "inr_path", "") or ""),
                volume_shape=np.asarray(self.volume_data.shape[:3]),
                experiment_roi=str(self.current_roi_name or ""),   # Experiment 그룹 ROI 이름 (짝 이름용)
                # 이 키가 있으면 ΔO·T가 step 보정 opacity로 합성된 덤프
                opacity_unit_distance=float(sampling_rules.OPACITY_UNIT_DISTANCE),
                **self.sam_prompt_for_dump(),
                **(extra or {}))
            print(f"💾 Stage A 샘플 저장: {path}")
            self._last_stage_a_dump = path
            self._last_stage_a_settings = self.stage_a_ui_settings()
        except Exception as e:
            self._last_stage_a_dump = None
            print(f"⚠️ Stage A 샘플 저장 실패: {e}")

    def dump_stage_b(self, mask, model_to_world):
        """진단용: Stage B 최종 mask와 UI 설정을 직전 Stage A 덤프와 같은 시각 이름(stage_b_*.npz)으로 저장한다."""
        try:
            a_path = getattr(self, "_last_stage_a_dump", None)
            name = (os.path.basename(a_path).replace("stage_a_", "stage_b_") if a_path
                    else f"stage_b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npz")
            path = os.path.join(self.STAGE_A_DUMP_DIR, name)
            np.savez_compressed(path, mask=np.asarray(mask, dtype=bool),
                                p_best=self.current_p_best,
                                model_to_world=np.asarray(model_to_world, dtype=float),
                                voxel_spacing=np.asarray(self.voxel_spacing, dtype=float),
                                **self.stage_b_ui_settings())
            print(f"💾 Stage B mask 저장: {path}")
        except Exception as e:
            print(f"⚠️ Stage B mask 저장 실패: {e}")

    def auto_build_local_mask(self, cluster_result, tf_lut):
        """Stage B: RF로 볼륨 전체 voxel을 판정해 새 local class를 등록하고 "Final" 오버레이를 만든다.

        tf_lut은 Stage A에 쓴 것과 같아야 한다. 같은 Stage A 결과·view·RF 설정이면 학습된 forest를
        cluster_result["_stage_b_forests"]에서 재사용한다.
        """
        if self.inr_feature_volume is not None and \
                tuple(self.inr_feature_volume.shape[:3]) != tuple(self.volume_data.shape[:3]):
            msg = (f"INR feature volume shape {self.inr_feature_volume.shape[:3]}이 "
                   f"volume_data shape {self.volume_data.shape[:3]}와 불일치합니다.")
            print(f"⚠️ {msg}")
            self.statusBar().showMessage(msg)
            return

        try:
            volume_actor = self.rendering_panel.vtk_renderer.standard_volume
            model_to_world = actor_matrix(volume_actor)
            num_voxels = float(np.prod(self.volume_data.shape[:3]))

            floor_stride = self.wysiwyg_panel.get_stage_b_ray_stride_floor()
            baker = self._get_stage_b_view_baker(volume_actor, tf_lut, floor_stride)

            rf_trees, rf_margin, rf_min_prob, rf_inr_only = self.wysiwyg_panel.get_stage_b_rf()
            forests = cluster_result.setdefault("_stage_b_forests", {})
            rf_key = (self._stage_b_view_cache[0], rf_trees, rf_margin, rf_inr_only)
            if rf_key not in forests:
                forests[rf_key] = train_stage_b_forest(
                    cluster_result, self.current_roi_info, self.volume_data,
                    self.inr_feature_volume, baker, model_to_world=model_to_world,
                    n_estimators=rf_trees, margin=rf_margin, inr_only=rf_inr_only)
            forest = forests[rf_key]
            best_pts = np.asarray(self.current_roi_info['filtered_points'])[forest.best_sample_idx]
            not_pts = voxel_indices_to_world(forest.not_best_voxels, self.voxel_spacing, model_to_world)
            self.current_rf_points = (
                np.concatenate([not_pts, best_pts]),
                np.concatenate([np.tile([60, 140, 255], (len(not_pts), 1)),
                                np.tile([255, 60, 60], (len(best_pts), 1))]))

            mask, voxel_labels, debug_info = classify_voxels_view_features(
                cluster_result, self.current_roi_info, self.volume_data,
                self.inr_feature_volume, baker, forests[rf_key], min_best_prob=rf_min_prob)

            self.current_p_best = debug_info.pop("p_best")   # local class debug에 32MB씩 쌓이지 않게 분리
            debug_info["voxel_count"] = int(mask.sum())
            debug_info["voxel_fraction"] = debug_info["voxel_count"] / num_voxels
            debug_info["mask_shape"] = mask_shape_summary(mask)
            print(format_stage_b(debug_info, mask))
            self.dump_stage_b(mask, model_to_world)

            # 초기 local TF = 지금 화면의 global TF 복사본 → ROI를 만들어도 모습이 그대로이고, 편집해야 달라진다
            range_norm = self.current_roi_info["range_norm"]
            label_id = self.register_local_class(
                mask, debug_info,
                view_range=self.padded_view_range(range_norm),
                roi_info=self.current_roi_info,
                tf_nodes=self.preview_tf_nodes if self.preview_active else None)
            if label_id is None:
                self.statusBar().showMessage("Clustering done, but local mask가 비어 있음.")
                return

            total = int(sum(int(m.sum()) for m in self.local_masks.values()))
            self.wysiwyg_panel.set_local_mask_status(
                total, total / num_voxels,
                debug_info['used_groups'] + [f"{len(self.local_masks)} class"],
                warning=debug_info.get('warning'))

            # "Final" 라벨: mask 안은 best, 밖은 best를 뺀 최근접 cluster
            try:
                best_idx = cluster_result["best_cluster_index"]
                voxel_labels = compose_final_labels(voxel_labels, self.local_masks[label_id], best_idx)
                idx, point_labels = sample_label_points(voxel_labels, 300000)
                world_points = voxel_indices_to_world(idx, self.voxel_spacing, model_to_world)

                has_points = len(world_points) > 0
                self.current_cluster_world_points = world_points if has_points else None
                self.current_cluster_labels = point_labels if has_points else None
                self.current_cluster_label_volume = voxel_labels
                self.on_cluster_view_changed()
            except Exception as e:
                print(f"⚠️ Cluster overlay 생성 중 예외: {e}")

            self.statusBar().showMessage(
                f"Clustering + local mask done: {debug_info['voxel_count']} voxels "
                f"({debug_info['voxel_fraction'] * 100:.1f}%)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.statusBar().showMessage(f"Local mask build failed (exception): {e}")
            self.sync_local_class_ui()

    # ------------------------------------------------------------------
    # 볼륨 / TF / 렌더링 설정
    # ------------------------------------------------------------------
    def on_inr_features_loaded(self, inr_data):
        self.inr_feature_volume = inr_data
        print(f"✅ INR feature volume stored: shape={inr_data.shape}")

    def on_volume_loaded(self, volume_data):
        """새 볼륨을 렌더러/TF 패널에 넘기고 SAM·ROI·local class 상태를 모두 초기화한다."""
        self.volume_data = volume_data
        self.voxel_spacing = self.file_panel.voxel_spacing
        self.rendering_panel.vtk_renderer.voxel_spacing = self.voxel_spacing
        self.rendering_panel.set_volume_data(volume_data)
        self.tf_panel.set_volume_data(volume_data)
        self.tf_panel.reset_clipping_safe()

        self.current_roi_info = None
        self.preview_tf_nodes = None
        self.base_tf_nodes = self.get_current_tf_nodes()
        self.tf_panel.tf_widget.clear_highlight_range()

        self.current_sam_mask = None
        self.current_sam_prompt = None
        self.optimization_panel.set_clustering_available(False)
        self.reset_local_classes()

    def on_tf_target_selected(self, kind, label_id):
        """TF 편집 대상(Global / Local N)이 바뀌면 그 대상의 TF와 가로축 범위를 위젯에 표시한다."""
        if kind == "global":
            nodes = self.preview_tf_nodes if self.preview_active else self.base_tf_nodes
        else:
            nodes = self.local_preview_tf_nodes.get(label_id) or self.local_tf_nodes.get(label_id)

        if nodes is not None:
            self.is_updating_tf_programmatically = True
            try:
                self.tf_panel.tf_widget.set_nodes(nodes)
            finally:
                self.is_updating_tf_programmatically = False

        view_range = self.local_tf_view_ranges.get(label_id) if kind == "local" else None
        if view_range:
            self.tf_panel.tf_widget.set_view_range(*view_range)
        else:
            self.tf_panel.tf_widget.set_view_range()

        self.wysiwyg_panel.set_preview_active(self.is_preview_active(kind, label_id))

    def on_tf_changed(self, tf_array):
        """TF 위젯 변경을 현재 대상(Global / Local N)의 렌더러에 반영한다.

        사용자가 직접 편집한 경우에만 그 대상의 base TF로 저장하고 미리보기를 해제한다.
        보기 모드(ROI 숨김 등) 중에 사용자가 편집하면 기본 보기로 되돌린다 — 한 대상만 바뀌어 섞이지 않게.
        코드가 위젯 노드를 바꾼 경우(편집 대상 전환 등)는 보기 모드를 유지하도록 다시 적용한다.
        """
        if self.view_mode != "default" and not self.is_updating_tf_programmatically:
            self.set_view_mode("default")
        kind, label_id = self.tf_panel.get_current_target()
        if kind == "global":
            self.rendering_panel.update_transfer_function(tf_array)
        else:
            self.rendering_panel.update_local_transfer_function(tf_array, label=label_id)

        if self.is_updating_tf_programmatically:
            if self.view_mode != "default":
                self.set_view_mode(self.view_mode)
            return

        if kind == "global":
            self.preview_tf_nodes = None
            self.preview_active = False
            self.base_tf_nodes = self.copy_tf_nodes(tf_array)
        else:
            self.local_preview_tf_nodes.pop(label_id, None)
            self.local_tf_nodes[label_id] = self.copy_tf_nodes(tf_array)
        self.wysiwyg_panel.set_preview_active(False)

    def on_background_color_changed(self, color1, color2):
        self.rendering_panel.set_background_color(color1, color2)

    def on_shading_changed(self, enabled):
        self.rendering_panel.set_shading(enabled)

    def on_lighting_changed(self, property_type, value):
        self.rendering_panel.set_lighting_property(property_type, value)

    def on_light_direction_changed(self, x, y, z):
        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.set_light_position('key', x, y, z)

    def on_follow_camera_changed(self, enabled):
        self.rendering_panel.vtk_renderer.set_follow_camera(enabled)

    def on_ambient_color_changed(self, r, g, b):
        self.rendering_panel.vtk_renderer.set_ambient_color(r, g, b)

    def on_diffuse_color_changed(self, r, g, b):
        self.rendering_panel.vtk_renderer.set_diffuse_color(r, g, b)

    def on_specular_color_changed(self, r, g, b):
        self.rendering_panel.vtk_renderer.set_specular_color(r, g, b)

    def on_clipping_changed(self, axis, min_val, max_val):
        self.rendering_panel.apply_clipping(axis, min_val, max_val)

    def on_clipping_enabled_changed(self, enabled):
        self.rendering_panel.set_clipping_enabled(enabled)

    def closeEvent(self, event):
        """종료 시 SAM 스레드와 VTK 리소스를 정리한다."""
        if hasattr(self, 'sam_model'):
            self.sam_model.shutdown()
        if hasattr(self, 'rendering_panel'):
            self.rendering_panel.cleanup()
        super().closeEvent(event)

    def load_dataset_preset(self, name):
        """프리셋의 볼륨 → INR → TF → 카메라 → clipping → SAM 포인트를 순서대로 로드한다(SAM 실행은 안 함)."""
        preset = DATASET_PRESETS.get(name)
        if preset is None:
            return

        try:
            self.file_panel.load_volume_from_path(preset["volume"], raw_params=preset.get("raw"))
            if preset.get("inr"):                     # INR 학습 전 데이터셋은 생략
                self.file_panel.load_inr_features_from_path(preset["inr"])
            self.tf_panel.load_tf_from_path(preset["tf"])
            if preset.get("camera"):                  # 카메라를 아직 저장하지 않은 데이터셋은 생략
                self.rendering_panel.load_camera_from_path(preset["camera"])
            if preset.get("clipping"):                # 볼륨 로드가 clipping을 초기화하므로 그 뒤에 적용
                self.tf_panel.set_clipping_ranges(preset["clipping"])
            if preset.get("points"):                  # 아직 SAM 포인트를 저장하지 않은 데이터셋은 생략
                self.optimization_panel.load_points_from_path(preset["points"])
            self.current_preset_name = name
            self.current_camera_name = "main"
            rois = preset.get("rois")
            self.tf_panel.set_experiment_rois(rois)
            self.current_roi_name = next(iter(rois)) if rois else None
            self.set_view_mode("default")
            self.statusBar().showMessage(f"Dataset preset '{preset['label']}' loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Dataset preset load failed: {e}")

    # ------------------------------------------------------------------
    # Experiment 촬영 (INR vs intensity 비교)
    # ------------------------------------------------------------------
    def current_experiment_roi(self):
        """Returns: 현재 프리셋의 현재 ROI 항목 {"points", "another"} 또는 None"""
        preset = DATASET_PRESETS.get(self.current_preset_name) or {}
        return (preset.get("rois") or {}).get(self.current_roi_name)

    def load_experiment_roi(self, roi_name):
        """프리셋 ROI의 SAM 포인트를 불러온다 (카메라는 기본 시점으로)."""
        preset = DATASET_PRESETS.get(self.current_preset_name) or {}
        roi = (preset.get("rois") or {}).get(roi_name)
        if roi is None:
            return
        self.current_roi_name = roi_name
        self.optimization_panel.load_points_from_path(roi["points"])
        self.load_experiment_camera("main")
        self.statusBar().showMessage(f"ROI '{roi_name}' 포인트 로드: {os.path.basename(roi['points'])}")

    def load_experiment_camera(self, which):
        """which: "main"(프리셋 기본 카메라) | "another"(현재 ROI의 측면 카메라)."""
        preset = DATASET_PRESETS.get(self.current_preset_name) or {}
        roi = self.current_experiment_roi() or {}
        path = preset.get("camera") if which == "main" else roi.get("another")
        if not path:
            self.statusBar().showMessage(f"ROI '{self.current_roi_name}'에는 {which} 카메라가 없습니다.")
            return
        self.rendering_panel.load_camera_from_path(path)
        self.current_camera_name = which
        self.statusBar().showMessage(f"카메라: {which} ({os.path.basename(path)})")

    def set_view_mode(self, mode):
        """보기 모드를 바꾸고 렌더링에 반영한다. 저장된 TF(base/local)는 바꾸지 않는다.

        default    저장된 TF 그대로
        roi_off    모든 local TF opacity 0 → ROI가 사라지고 나머지(global)만
        rest_off   global TF opacity 0 → ROI만
        roi_boost  local TF opacity = max(local, 강조값 × tent[best intensity 구간]) — Unity 슬라이더를 올린 것과 같은 식
        """
        if mode != "default" and not self.local_masks:
            self.statusBar().showMessage("ROI(local class)가 없어서 보기 모드를 바꿀 수 없습니다.")
            mode = "default"
        self.view_mode = mode
        self.tf_panel.set_view_mode(mode)

        global_nodes = self.preview_tf_nodes if self.preview_active else self.base_tf_nodes
        if global_nodes is not None:
            self.rendering_panel.update_transfer_function(
                self.scale_tf_alpha(global_nodes, 0.0) if mode == "rest_off" else global_nodes)
        strength = self.tf_panel.get_boost_strength()
        for lid in sorted(self.local_masks):
            nodes = self.local_preview_tf_nodes.get(lid) or self.local_tf_nodes.get(lid)
            if nodes is None:
                continue
            if mode == "roi_off":
                nodes = self.scale_tf_alpha(nodes, 0.0)
            elif mode == "roi_boost":
                roi = self.local_roi_infos.get(lid) or {}
                if "range_norm" in roi:
                    nodes = self.boosted_tf_nodes(nodes, roi["range_norm"], roi.get("center_norm"), strength)
            self.rendering_panel.update_local_transfer_function(nodes, label=lid, render=False)
        self.rendering_panel.render()

    @staticmethod
    def recolor_tf_nodes(nodes, rgb):
        """모든 노드 색을 rgb 단색으로 (opacity 그대로) — Unity 팔레트(ROI 전체 단색)와 같은 효과"""
        return [[n[0], float(rgb[0]), float(rgb[1]), float(rgb[2]), n[4]] for n in nodes]

    def apply_roi_color(self, rgb):
        """Experiment ROI 색 변경: 지금 있는 모든 local TF(확정본)를 그 색으로 바꾸고 다시 그린다."""
        for lid in list(self.local_tf_nodes):
            self.local_tf_nodes[lid] = self.recolor_tf_nodes(self.local_tf_nodes[lid], rgb)
            self.local_preview_tf_nodes.pop(lid, None)
        if self.local_masks:
            self.set_view_mode(self.view_mode)
            kind, label_id = self.tf_panel.get_current_target()
            if kind == "local":
                self.on_tf_target_selected(kind, label_id)   # 편집 위젯도 새 색으로

    @staticmethod
    def scale_tf_alpha(nodes, factor):
        return [[n[0], n[1], n[2], n[3], n[4] * factor] for n in nodes]

    @staticmethod
    def boosted_tf_nodes(nodes, range_norm, center, strength, n_samples=256):
        """local TF에 best intensity 구간 tent를 max로 더한 노드 (Unity LocalTFRenderer.BuildClass/RebuildLut와 같은 tent).

        tent: [low, high] 밖 0, low → center(꼭짓점 1) → high 선형. 구간 폭이 1e-4 미만이면 center ± 0.02.
        Returns: 256개로 샘플링한 [x, r, g, b, a] (색은 원래 local TF)
        """
        low, high = float(min(range_norm)), float(max(range_norm))
        low, high = min(max(low, 0.0), 1.0), min(max(high, 0.0), 1.0)
        c = float(center) if center is not None else 0.5 * (low + high)
        if high - low < 1e-4:
            low, high = max(0.0, c - 0.02), min(1.0, c + 0.02)
        c = min(max(c, low), high)

        arr = np.asarray(sorted(nodes, key=lambda n: n[0]), dtype=np.float64)
        xs = np.linspace(0.0, 1.0, n_samples)
        rgb = [np.interp(xs, arr[:, 0], arr[:, ch]) for ch in (1, 2, 3)]
        alpha = np.interp(xs, arr[:, 0], arr[:, 4])
        tent = np.zeros_like(xs)
        up = (xs > low) & (xs <= c)
        down = (xs > c) & (xs < high)
        tent[up] = (xs[up] - low) / max(c - low, 1e-6)
        tent[down] = (high - xs[down]) / max(high - c, 1e-6)
        alpha = np.maximum(alpha, strength * tent)
        return [[float(x), float(r), float(g), float(b), float(a)]
                for x, r, g, b, a in zip(xs, rgb[0], rgb[1], rgb[2], alpha)]

    CAPTURE_ROOT = os.path.join("resources", "Experiments", "captures")

    def capture_experiment_view(self, quiet=False):
        """현재 화면을 조건 이름으로 저장하고 captures.csv에 한 줄 남긴다.

        경로: resources/Experiments/captures/<데이터셋>/<ROI>/<데이터셋>_<ROI>_<INR|intensity>_s<seed>_<카메라>_<보기>.png
        feature·seed는 직전 Stage A 실행의 설정(덤프에 기록된 값)을 쓴다. 같은 이름이 있으면 _2, _3 …
        Returns: 저장 경로 또는 None
        """
        s = self._last_stage_a_settings
        if s is None:
            if not quiet:
                QMessageBox.warning(self, "촬영", "아직 Run Clustering(Stage A) 결과가 없습니다.")
            return None
        feature = "INR" if s.get("use_inr") else ("intensity" if s.get("use_intensity") else "other")
        dataset = self.current_preset_name or "custom"
        roi = (self.current_roi_name or "roi").replace(" ", "")
        mode = self.view_mode if self.view_mode != "roi_boost" else f"roi_boost{self.tf_panel.get_boost_strength():.2f}"
        folder = os.path.join(self.CAPTURE_ROOT, dataset, roi)
        stem = f"{dataset}_{roi}_{feature}_s{s.get('seed')}_{self.current_camera_name}_{mode}"
        path, i = os.path.join(folder, stem + ".png"), 2
        while os.path.exists(path):
            path, i = os.path.join(folder, f"{stem}_{i}.png"), i + 1

        saved = self.rendering_panel.save_rendering_to(path)
        if not saved:
            if not quiet:
                QMessageBox.critical(self, "촬영", "저장 실패")
            return None
        log = os.path.join(self.CAPTURE_ROOT, "captures.csv")
        new = not os.path.exists(log)
        dump = os.path.basename(getattr(self, "_last_stage_a_dump", None) or "")
        local_voxels = sum(int(m.sum()) for m in self.local_masks.values())
        with open(log, "a", encoding="utf-8-sig", newline="") as f:
            if new:
                f.write("file,time,dataset,roi,feature,seed,k,restarts,camera,view,stage_a_dump,local_voxels\n")
            f.write(f"{os.path.relpath(saved, self.CAPTURE_ROOT)},"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')},{dataset},{roi},{feature},{s.get('seed')},"
                    f"{s.get('k')},{s.get('kmeans_n_init')},{self.current_camera_name},{mode},{dump},{local_voxels}\n")
        self.tf_panel.set_capture_status(os.path.basename(saved))
        if not quiet:
            self.statusBar().showMessage(f"촬영 저장: {saved}")
        return saved

    # ------------------------------------------------------------------
    # Cluster 오버레이 / Slice viewer (디버그)
    # ------------------------------------------------------------------
    def on_cluster_overlay_toggled(self, checked):
        """"Final"(Stage B) 라벨 점 구름 오버레이 on/off."""
        if checked:
            self.on_cluster_view_changed()
        else:
            self.rendering_panel.clear_roi_actor()

    def on_kmeans_overlay_toggled(self, checked):
        """"K-means"(Stage A) ROI 샘플 라벨 오버레이 on/off."""
        if checked:
            self.on_cluster_view_changed()
        else:
            self.rendering_panel.clear_roi_actor()

    def on_cluster_view_changed(self):
        """켜져 있는 오버레이(Final / K-means)를 체크된 cluster만 보이도록 갱신한다."""
        panel = self.wysiwyg_panel
        if panel.show_rf_button.isChecked():
            if self.current_rf_points is None:
                self.statusBar().showMessage("RF 학습셋 없음: 이 ROI에서 Stage B를 먼저 실행하세요.")
                self.rendering_panel.clear_roi_actor()
            else:
                self.rendering_panel.vtk_renderer.set_point_cloud_actor(*self.current_rf_points)
            return
        if panel.show_clusters_button.isChecked():
            points, labels = self.current_cluster_world_points, self.current_cluster_labels
        elif panel.show_kmeans_button.isChecked():
            points, labels = self.current_kmeans_world_points, self.current_kmeans_labels
        else:
            return
        self.rendering_panel.show_cluster_points(
            points, labels, panel.get_visible_clusters(), self.current_best_cluster)

    def on_cluster_color_changed(self):
        """패널에서 바꾼 cluster 색을 켜져 있는 오버레이와 slice viewer에 반영한다."""
        self.on_cluster_view_changed()
        if self.slice_debug_viewer is not None:
            self.slice_debug_viewer.refresh_colors()

    def on_slice_viewer_requested(self):
        """볼륨 Z-slice 위에 Final / K-means 라벨을 겹쳐 보여주는 창을 연다(하나만 생성해 재사용)."""
        if self.volume_data is None:
            self.statusBar().showMessage("먼저 볼륨을 로드하세요.")
            return

        if self.slice_debug_viewer is None:
            self.slice_debug_viewer = SliceDebugViewer(parent=self)

        volume_actor = self.rendering_panel.vtk_renderer.standard_volume
        model_to_world = actor_matrix(volume_actor) if volume_actor is not None else None

        self.slice_debug_viewer.set_volume(self.volume_data, self.voxel_spacing, model_to_world)
        self.refresh_slice_viewer_sources()
        self.slice_debug_viewer.show()
        self.slice_debug_viewer.raise_()
        self.slice_debug_viewer.activateWindow()

    def refresh_slice_viewer_sources(self):
        """slice viewer의 Final(라벨 볼륨 우선, 없으면 점 구름)과 K-means 데이터를 현재 결과로 맞춘다."""
        viewer = self.slice_debug_viewer
        if viewer is None:
            return
        viewer.set_best_cluster(self.current_best_cluster)
        if self.current_cluster_label_volume is not None:
            viewer.set_points("Final", None, None)
            viewer.set_label_volume("Final", self.current_cluster_label_volume)
        else:
            viewer.set_label_volume("Final", None)
            viewer.set_points("Final", self.current_cluster_world_points, self.current_cluster_labels)
        viewer.set_points("K-means", self.current_kmeans_world_points, self.current_kmeans_labels)
        viewer.set_prob_volume("RF P(best)", self.current_p_best)

    # ------------------------------------------------------------------
    # Local class(다중 ROI) 관리
    # ------------------------------------------------------------------
    def reset_local_classes(self):
        """모든 local class 상태를 지우고 렌더러의 local 확률맵을 해제한다."""
        self.local_masks = {}
        self.local_mask_debug = {}
        self.local_roi_infos = {}
        self.local_tf_nodes = {}
        self.local_preview_tf_nodes = {}
        self.local_tf_view_ranges = {}
        self.local_mask_active = False
        self.current_local_mask = None
        self.rendering_panel.clear_local_probability_maps()
        self.tf_panel.set_local_tf_slots([])
        self.wysiwyg_panel.reset_local_mask_status()
        if self.view_mode != "default":
            self.set_view_mode("default")

    def sync_local_class_ui(self):
        """TF 패널의 Local 목록과 WYSIWYG 패널의 상태 라벨을 현재 local class에 맞춘다."""
        self.tf_panel.set_local_tf_slots(sorted(self.local_masks))
        if not self.local_masks:
            self.wysiwyg_panel.reset_local_mask_status()
            return
        total = int(sum(int(m.sum()) for m in self.local_masks.values()))
        self.wysiwyg_panel.set_local_mask_status(
            total, total / float(np.prod(self.volume_data.shape[:3])),
            [f"{len(self.local_masks)} local class"])

    def next_local_label_id(self):
        """새 class의 label_id(현재 최대 + 1). 삭제된 번호는 재사용하지 않는다."""
        return (max(self.local_masks) + 1) if self.local_masks else 1

    def drop_local_class_state(self, label_id):
        """label_id의 모든 상태를 dict에서 제거한다(렌더러 반영은 호출자 몫)."""
        for store in (self.local_masks, self.local_mask_debug, self.local_roi_infos,
                      self.local_tf_nodes, self.local_preview_tf_nodes, self.local_tf_view_ranges):
            store.pop(label_id, None)

    def push_local_probability_maps(self):
        """local_masks를 확률맵으로 렌더러에 올리고 class별 TF를 다시 적용한다."""
        if not self.local_masks:
            self.rendering_panel.clear_local_probability_maps()
            return
        prob_maps = {lid: mask.astype(np.float32) for lid, mask in self.local_masks.items()}
        self.rendering_panel.set_local_probability_maps(prob_maps)
        for lid in sorted(self.local_masks):
            nodes = self.local_preview_tf_nodes.get(lid) or self.local_tf_nodes.get(lid)
            if nodes is not None:
                self.rendering_panel.update_local_transfer_function(nodes, label=lid, render=False)
        self.rendering_panel.render()

    def register_local_class(self, mask, debug_info, view_range=None, roi_info=None, tf_nodes=None):
        """새 local class를 추가한다.

        겹치는 voxel은 기존 class에서 빼고(나중 class 우선), 그 결과 비게 된 class는 삭제한다.
        초기 TF는 tf_nodes, 없으면 global TF 복사본.
        Returns: 새 label_id, mask가 비어 있으면 None
        """
        new_mask = np.asarray(mask) > 0
        if not new_mask.any():
            print("⚠️ 빈 mask는 local class로 등록하지 않음")
            return None

        label_id = self.next_local_label_id()

        emptied = []
        for lid in list(self.local_masks):
            kept = self.local_masks[lid] & ~new_mask
            if kept.any():
                self.local_masks[lid] = kept
            else:
                emptied.append(lid)
        for lid in emptied:
            self.drop_local_class_state(lid)
            print(f"ℹ️ Local {lid}은 새 class와 완전히 겹쳐서 삭제됨")

        self.local_masks[label_id] = new_mask
        self.local_mask_debug[label_id] = debug_info
        if roi_info is not None:
            self.local_roi_infos[label_id] = roi_info
        self.local_tf_nodes[label_id] = self.copy_tf_nodes(
            tf_nodes if tf_nodes is not None else self.base_tf_nodes)
        roi_color = self.tf_panel.get_auto_roi_color()   # Experiment "새 ROI에 자동 적용" — opacity는 그대로, 색만 단색
        if roi_color is not None:
            self.local_tf_nodes[label_id] = self.recolor_tf_nodes(self.local_tf_nodes[label_id], roi_color)
        self.local_preview_tf_nodes.pop(label_id, None)
        if view_range:
            self.local_tf_view_ranges[label_id] = view_range

        self.local_mask_active = True
        self.current_local_mask = (new_mask.astype(np.uint8), debug_info)

        self.push_local_probability_maps()
        self.tf_panel.set_local_tf_slots(sorted(self.local_masks))
        self.tf_panel.set_current_target("local", label_id)
        if self.view_mode != "default":     # 새 ROI가 생기면 보기 모드를 기본으로 (global이 꺼진 채 남지 않게)
            self.set_view_mode("default")
        print(f"✅ Local {label_id} 등록 ({int(new_mask.sum()):,} voxel) — "
              f"현재 class {sorted(self.local_masks)}")
        return label_id

    def delete_local_class(self, label_id):
        """local class 하나(mask, TF, 상태)를 삭제한다. 마지막 class면 local 렌더링을 해제한다."""
        if label_id not in self.local_masks:
            return
        self.drop_local_class_state(label_id)
        if self.view_mode != "default":
            self.view_mode = "default"          # 아래 렌더링 갱신이 저장된 TF로 다시 그린다
            self.tf_panel.set_view_mode("default")
            global_nodes = self.preview_tf_nodes if self.preview_active else self.base_tf_nodes
            if global_nodes is not None:
                self.rendering_panel.update_transfer_function(global_nodes)

        if not self.local_masks:
            self.reset_local_classes()
            msg = f"Local {label_id} 삭제 — 남은 local class 없음 (global TF만)"
            print(f"🗑️ {msg}")
            self.statusBar().showMessage(msg)
            return

        newest = max(self.local_masks)
        self.current_local_mask = (self.local_masks[newest].astype(np.uint8),
                                   self.local_mask_debug.get(newest))
        self.push_local_probability_maps()
        self.sync_local_class_ui()
        msg = f"Local {label_id} 삭제 — 남은 class {sorted(self.local_masks)}"
        print(f"🗑️ {msg}")
        self.statusBar().showMessage(msg)

    # ------------------------------------------------------------------
    # TF 헬퍼
    # ------------------------------------------------------------------
    def copy_tf_nodes(self, nodes):
        if nodes is None:
            return None
        return [node[:] for node in nodes]

    def padded_view_range(self, range_norm, pad_fraction=0.5, min_pad=0.02):
        """range_norm 양옆에 폭의 pad_fraction(최소 min_pad)만큼 여백을 둔 TF 위젯 가로축 범위."""
        low, high = min(range_norm), max(range_norm)
        span = max(high - low, 1e-4)
        pad = max(span * pad_fraction, min_pad)
        return (max(0.0, low - pad), min(1.0, high + pad))

    def get_current_tf_nodes(self):
        return self.copy_tf_nodes(self.tf_panel.tf_widget.get_nodes())

    def current_target(self):
        """현재 TF 편집 대상. Returns: ("global", None) 또는 ("local", label_id)"""
        return self.tf_panel.get_current_target()

    def is_preview_active(self, kind, label_id):
        if kind == "global":
            return self.preview_active
        return self.local_preview_tf_nodes.get(label_id) is not None

    def apply_tf_to_view(self, nodes, kind="global", label_id=None):
        """nodes를 TF 위젯과 해당 대상(Global / Local)의 렌더러에 반영한다(base TF는 바꾸지 않음)."""
        if nodes is None:
            return
        self.is_updating_tf_programmatically = True
        try:
            self.tf_panel.tf_widget.set_nodes(nodes)
            if kind == "global":
                self.rendering_panel.update_transfer_function(nodes)
            else:
                self.rendering_panel.update_local_transfer_function(nodes, label=label_id)
        finally:
            self.is_updating_tf_programmatically = False

    # ------------------------------------------------------------------
    # WYSIWYG 도구
    # ------------------------------------------------------------------
    def on_wysiwyg_preview_clicked(self):
        """선택한 도구로 현재 대상 TF의 미리보기를 만든다."""
        tool = self.wysiwyg_panel.get_current_tool()
        params = self.wysiwyg_panel.get_current_tool_params()
        handlers = {
            "eraser": self.preview_wysiwyg_eraser,
            "brightness": self.preview_wysiwyg_brightness,
            "contrast": self.preview_wysiwyg_contrast,
            "colorization": self.preview_wysiwyg_colorization,
        }
        handler = handlers.get(tool)
        if handler is None:
            print(f"⚠️ 지원하지 않는 도구: {tool}")
            return
        handler(**params)

        kind, label_id = self.current_target()
        self.wysiwyg_panel.set_preview_active(self.is_preview_active(kind, label_id))

    def ensure_wysiwyg_ready(self, kind=None):
        """ROI 통계와 TF node가 있는지 확인한다. Local은 class 자신의 roi_info가 있으면 통과."""
        label_id = self.current_target()[1] if kind == "local" else None
        local_has_roi = kind == "local" and label_id in self.local_roi_infos
        if self.current_roi_info is None and not local_has_roi:
            print("⚠️ No ROI selected.")
            self.statusBar().showMessage("No ROI selected.")
            return False

        current_nodes = self.get_current_tf_nodes()
        if not current_nodes:
            print("⚠️ TF nodes are empty.")
            self.statusBar().showMessage("TF nodes are empty.")
            return False
        return True

    def run_wysiwyg_preview(self, tool_runner, **params):
        """도구 함수로 현재 대상(Global / Local N)의 미리보기 TF를 만들어 화면에 적용한다.

        tool_runner(tf_editor, base_tf_nodes, roi_info, **params) → (new_nodes, debug)
        Returns: 도구의 debug dict, 실패 시 None
        """
        kind, label_id = self.current_target()
        if not self.ensure_wysiwyg_ready(kind):
            return None

        if kind == "local" and not self.local_mask_active:
            msg = "Local mask가 아직 준비되지 않았습니다. 먼저 'Run Clustering'을 실행하세요."
            print(f"⚠️ {msg}")
            self.statusBar().showMessage(msg)
            return None

        try:
            if kind == "global" and not self.preview_active:
                self.base_tf_nodes = self.get_current_tf_nodes()

            if kind == "local":
                source_nodes = self.local_tf_nodes.get(label_id)
                roi_info = self.local_roi_infos.get(label_id, self.current_roi_info)
            else:
                source_nodes = self.base_tf_nodes
                roi_info = self.current_roi_info

            new_tf_nodes, tf_debug = tool_runner(
                tf_editor=self.wysiwyg_tf_editor,
                base_tf_nodes=source_nodes,
                roi_info=roi_info,
                **params,
            )

            new_tf_nodes = self.copy_tf_nodes(new_tf_nodes)
            if kind == "local":
                self.local_preview_tf_nodes[label_id] = new_tf_nodes
            else:
                self.preview_tf_nodes = new_tf_nodes
                self.preview_active = True

            self.apply_tf_to_view(new_tf_nodes, kind, label_id)
            self.wysiwyg_panel.set_preview_active(True)
            return tf_debug

        except Exception as e:
            print(f"⚠️ WYSIWYG preview failed: {e}")
            self.statusBar().showMessage(f"WYSIWYG preview failed: {e}")
            return None

    def apply_wysiwyg_preview(self):
        """현재 대상의 미리보기 TF를 base TF로 확정한다."""
        kind, label_id = self.current_target()

        if kind == "local":
            preview_nodes = self.local_preview_tf_nodes.pop(label_id, None)
            if preview_nodes is None:
                print("⚠️ No local preview to apply.")
                return
            self.local_tf_nodes[label_id] = self.copy_tf_nodes(preview_nodes)
            self.wysiwyg_panel.set_preview_active(False)
            self.apply_tf_to_view(self.local_tf_nodes[label_id], "local", label_id)
            print("✅ WYSIWYG local preview applied.")
            return

        if self.preview_tf_nodes is None:
            print("⚠️ No preview to apply.")
            return
        self.base_tf_nodes = self.copy_tf_nodes(self.preview_tf_nodes)
        self.preview_tf_nodes = None
        self.preview_active = False
        self.wysiwyg_panel.set_preview_active(False)
        self.apply_tf_to_view(self.base_tf_nodes, "global")
        print("✅ WYSIWYG preview applied.")

    def reset_wysiwyg_preview(self):
        """현재 대상의 미리보기를 버리고 base TF로 되돌린다."""
        kind, label_id = self.current_target()

        if kind == "local":
            local_nodes = self.local_tf_nodes.get(label_id)
            if local_nodes is None:
                print("⚠️ No local base TF stored.")
                return
            self.local_preview_tf_nodes.pop(label_id, None)
            self.wysiwyg_panel.set_preview_active(False)
            self.apply_tf_to_view(local_nodes, "local", label_id)
            print("↩️ WYSIWYG local preview reset.")
            return

        if self.base_tf_nodes is None:
            print("⚠️ No base TF stored.")
            return
        self.preview_tf_nodes = None
        self.preview_active = False
        self.wysiwyg_panel.set_preview_active(False)
        self.apply_tf_to_view(self.base_tf_nodes, "global")
        print("↩️ WYSIWYG preview reset.")

    def _print_tool_debug(self, title, tf_debug, keys):
        print(f"\n[WYSIWYG {title} PREVIEW]")
        for key in keys:
            value = tf_debug[key]
            print(f"   {key:<20}: {value:.5f}" if isinstance(value, float) else f"   {key:<20}: {value}")

    def preview_wysiwyg_eraser(self, strength=0.5, feather=0.0, mode="decrease"):
        tf_debug = self.run_wysiwyg_preview(preview_eraser, strength=strength, feather=feather, mode=mode)
        if tf_debug is not None:
            self._print_tool_debug("ERASER", tf_debug, (
                "mode", "range_norm", "feather", "mean_weight",
                "opacity_before_mean", "opacity_after_mean"))

    def preview_wysiwyg_brightness(self, strength=0.15, feather=0.0, mode="increase"):
        tf_debug = self.run_wysiwyg_preview(preview_brightness, strength=strength, mode=mode, feather=0.0)
        if tf_debug is not None:
            self._print_tool_debug("BRIGHTNESS", tf_debug, (
                "mode", "range_norm", "strength", "affected_count", "l_before_mean", "l_after_mean"))

    def preview_wysiwyg_colorization(self, strength=0.25, feather=0.0, mode="apply", color=None):
        tf_debug = self.run_wysiwyg_preview(
            preview_colorization, strength=strength, feather=0.0, mode=mode, color=color)
        if tf_debug is not None:
            self._print_tool_debug("COLORIZATION", tf_debug, (
                "mode", "range_norm", "strength", "affected_count", "target_rgb",
                "l_before_mean", "l_after_mean"))

    def preview_wysiwyg_contrast(self, strength=0.2, feather=0.0, mode="increase"):
        tf_debug = self.run_wysiwyg_preview(preview_contrast, strength=strength, feather=0.0, mode=mode)
        if tf_debug is not None:
            self._print_tool_debug("CONTRAST", tf_debug, (
                "mode", "range_norm", "strength", "original_roi_count", "affected_count",
                "expanded", "scale", "l_std_before", "l_std_after"))
