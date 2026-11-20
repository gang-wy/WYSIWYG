"""우측 WYSIWYG 패널: 상태 표시, Clustering/Stage B 설정, 도구 선택·옵션, Preview/Apply/Reset, 디버그 오버레이."""

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QStackedWidget, QSlider, QSpinBox, QDoubleSpinBox,
    QComboBox, QColorDialog, QFrame, QCheckBox
)

from src.gui.widget.renderer_widget import cluster_colors, set_cluster_color

# 옵션 페이지 인덱스
PAGE_EMPTY, PAGE_BASIC, PAGE_COLOR = 0, 1, 2
BASIC_TOOLS = ("eraser", "contrast", "brightness")
BASIC_TOOL_TITLES = {"eraser": "Visibility", "contrast": "Contrast", "brightness": "Brightness"}
IMPLEMENTED_TOOLS = BASIC_TOOLS + ("colorization",)
# 논문 도구 중 아직 구현되지 않은 것(버튼만 비활성 상태로 표시)
TOOL_NAMES = ("eraser", "colorization", "rainbow", "contrast",
              "brightness", "silhouette", "fuzziness", "peeling")


class WysiwygPanel(QWidget):
    """설정값은 get_* 메서드로 읽히고, 버튼 동작은 시그널로 main_window에 전달된다."""

    preview_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    cluster_overlay_toggled = pyqtSignal(bool)   # "Final"(Stage B) 오버레이 on/off
    kmeans_overlay_toggled = pyqtSignal(bool)    # "K-means"(Stage A) 오버레이 on/off
    rf_overlay_toggled = pyqtSignal(bool)        # "RF samples"(Stage B 학습셋) 오버레이 on/off
    cluster_view_changed = pyqtSignal()          # 표시할 cluster 체크 변경 (get_visible_clusters로 조회)
    cluster_color_changed = pyqtSignal()
    slice_viewer_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tool = None
        self.selected_color = "#ff0000"
        self.best_cluster = None
        self.cluster_swatches = []     # cluster 번호 순서의 색 버튼
        self.cluster_view_checks = []  # cluster 번호 순서의 표시 체크박스
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)

        # 상태
        status_group = QGroupBox("WYSIWYG Status")
        status_layout = QVBoxLayout()
        self.roi_status_label = QLabel("ROI: Not Selected")
        self.tool_status_label = QLabel("Tool: None")
        self.preview_status_label = QLabel("Preview: Inactive")
        for label in (self.roi_status_label, self.tool_status_label, self.preview_status_label):
            status_layout.addWidget(label)
        status_group.setLayout(status_layout)
        main_layout.addWidget(status_group)

        main_layout.addWidget(self.build_clustering_group())

        # 도구 선택
        tool_group = QGroupBox("Tools")
        tool_layout = QGridLayout()
        self.tool_buttons = {}
        for idx, tool_name in enumerate(TOOL_NAMES):
            button = QPushButton(tool_name.capitalize())
            button.setCheckable(True)
            if tool_name in IMPLEMENTED_TOOLS:
                button.clicked.connect(lambda checked, name=tool_name: self.select_tool(name))
            else:
                button.setEnabled(False)
                button.setToolTip("아직 구현되지 않은 도구")
            tool_layout.addWidget(button, idx // 2, idx % 2)
            self.tool_buttons[tool_name] = button
        tool_group.setLayout(tool_layout)
        main_layout.addWidget(tool_group)

        # 도구 옵션
        option_group = QGroupBox("Tool Options")
        option_layout = QVBoxLayout()
        self.option_stack = QStackedWidget()
        self.option_stack.addWidget(self.create_empty_page())         # PAGE_EMPTY
        self.option_stack.addWidget(self.create_basic_page())         # PAGE_BASIC
        self.option_stack.addWidget(self.create_colorization_page())  # PAGE_COLOR
        option_layout.addWidget(self.option_stack)
        option_group.setLayout(option_layout)
        main_layout.addWidget(option_group)

        # Local TF 상태
        local_group = QGroupBox("Local TF (Multi-DVR)")
        local_layout = QVBoxLayout()
        self.local_mask_status_label = QLabel("Local mask: not built")
        self.local_mask_status_label.setWordWrap(True)
        local_layout.addWidget(self.local_mask_status_label)
        local_group.setLayout(local_layout)
        main_layout.addWidget(local_group)

        # Preview / Apply / Reset
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()
        for text, signal in (("Preview", self.preview_requested),
                             ("Apply", self.apply_requested),
                             ("Reset", self.reset_requested)):
            button = QPushButton(text)
            button.clicked.connect(signal.emit)
            action_layout.addWidget(button)
        action_group.setLayout(action_layout)
        main_layout.addWidget(action_group)

        main_layout.addWidget(self.build_debug_group())
        main_layout.addStretch()

    def build_clustering_group(self):
        """K / K-means seed·재시작 / Stage B stride 하한, Stage B RF 설정, feature 그룹 체크박스."""
        cluster_group = QGroupBox("Clustering")
        cluster_layout = QVBoxLayout()

        params_row = QHBoxLayout()
        params_row.addWidget(QLabel("k"))
        self.cluster_k_spin = QSpinBox()
        self.cluster_k_spin.setRange(2, 30)
        self.cluster_k_spin.setValue(6)
        params_row.addWidget(self.cluster_k_spin)

        params_row.addWidget(QLabel("seed"))
        self.cluster_seed_spin = QSpinBox()
        self.cluster_seed_spin.setRange(0, 999999)
        self.cluster_seed_spin.setValue(3)
        self.cluster_seed_spin.setToolTip("K-means 초기화 seed. Run Clustering 때 반영된다.")
        params_row.addWidget(self.cluster_seed_spin)

        params_row.addWidget(QLabel("restarts"))
        self.cluster_n_init_spin = QSpinBox()
        self.cluster_n_init_spin.setRange(1, 50)
        self.cluster_n_init_spin.setValue(10)
        self.cluster_n_init_spin.setToolTip(
            "K-means 재시작 횟수. 샘플 5만 개로 이만큼 다시 돌려 SSE가 가장 낮은 해에서 전체를 돈다. "
            "1이면 초기화 1번.")
        params_row.addWidget(self.cluster_n_init_spin)

        params_row.addWidget(QLabel("Stage B stride floor"))
        self.stage_b_stride_spin = QSpinBox()
        self.stage_b_stride_spin.setRange(1, 32)
        self.stage_b_stride_spin.setValue(2)
        self.stage_b_stride_spin.setToolTip(
            "Stage B 투과율 격자 ray 간격(픽셀)의 하한. 실제 간격은 voxel 화면 투영 크기로 자동 계산된다.")
        params_row.addWidget(self.stage_b_stride_spin)
        cluster_layout.addLayout(params_row)

        rf_row = QHBoxLayout()
        rf_row.addWidget(QLabel("Stage B RF:"))
        self.stage_b_rf_inr_only_check = QCheckBox("INR만")
        self.stage_b_rf_inr_only_check.setChecked(False)
        self.stage_b_rf_inr_only_check.setToolTip("RF를 INR embedding + coords만으로 학습·판정한다.")
        rf_row.addWidget(self.stage_b_rf_inr_only_check)

        rf_row.addWidget(QLabel("trees"))
        self.stage_b_rf_trees_spin = QSpinBox()
        self.stage_b_rf_trees_spin.setRange(10, 1000)
        self.stage_b_rf_trees_spin.setSingleStep(50)
        self.stage_b_rf_trees_spin.setValue(100)
        self.stage_b_rf_trees_spin.setToolTip("tree 수. 볼륨 전체 예측 시간이 tree 수에 비례한다.")
        rf_row.addWidget(self.stage_b_rf_trees_spin)

        rf_row.addWidget(QLabel("마진"))
        self.stage_b_rf_margin_spin = QSpinBox()
        self.stage_b_rf_margin_spin.setRange(0, 20)
        self.stage_b_rf_margin_spin.setValue(2)
        self.stage_b_rf_margin_spin.setToolTip("best seed를 이만큼(voxel) 팽창한 영역에서는 not-best 샘플을 뽑지 않는다.")
        rf_row.addWidget(self.stage_b_rf_margin_spin)

        rf_row.addWidget(QLabel("P(best) ≥"))
        self.stage_b_rf_prob_spin = QDoubleSpinBox()
        self.stage_b_rf_prob_spin.setRange(0.0, 0.99)
        self.stage_b_rf_prob_spin.setSingleStep(0.05)
        self.stage_b_rf_prob_spin.setDecimals(2)
        self.stage_b_rf_prob_spin.setValue(0.0)
        self.stage_b_rf_prob_spin.setToolTip("best 판정은 P(best) > 0.5 이고 이 값 이상.")
        rf_row.addWidget(self.stage_b_rf_prob_spin)
        rf_row.addStretch()
        cluster_layout.addLayout(rf_row)

        feature_group_row = QHBoxLayout()
        feature_group_row.addWidget(QLabel("Features:"))
        self.feature_inr_check = QCheckBox("INR (8D)")
        self.feature_inr_check.setChecked(True)
        feature_group_row.addWidget(self.feature_inr_check)
        self.feature_coords_check = QCheckBox("Coords (3D)")
        self.feature_coords_check.setChecked(True)
        feature_group_row.addWidget(self.feature_coords_check)
        cluster_layout.addLayout(feature_group_row)

        local_row = QHBoxLayout()
        local_row.addWidget(QLabel("Local:"))
        self.local_feature_checks = {
            "visibility": QCheckBox("Visibility"),
            "depth": QCheckBox("Depth"),
            "intensity": QCheckBox("Intensity"),
            "opacity": QCheckBox("Opacity"),
        }
        self.local_feature_checks["visibility"].setToolTip("현재 카메라 시점 기준 값")
        self.local_feature_checks["depth"].setToolTip("현재 카메라 시점 기준 값")
        default_checked = {"visibility": True, "depth": True, "intensity": False, "opacity": True}
        for name, check in self.local_feature_checks.items():
            check.setChecked(default_checked[name])
            local_row.addWidget(check)
        local_row.addStretch()
        cluster_layout.addLayout(local_row)

        vis_row = QHBoxLayout()
        vis_row.addWidget(QLabel("Visibility 값:"))
        self.visibility_feature_combo = QComboBox()
        self.visibility_feature_combo.addItem("V = α × 투과율", "V")
        self.visibility_feature_combo.addItem("T = 투과율", "T")
        self.visibility_feature_combo.setToolTip(
            "Visibility feature 열에 넣을 값 (Run Clustering 때 반영). best 점수는 항상 V로 계산한다.")
        vis_row.addWidget(self.visibility_feature_combo)
        self.sparse_minmax_check = QCheckBox("V/O min-max")
        self.sparse_minmax_check.setChecked(False)
        self.sparse_minmax_check.setToolTip(
            "Visibility·Opacity 열을 z-score 대신 min-max(0 ~ 0이 아닌 값의 p99)로 정규화한다. "
            "끄면 모든 열을 z-score (Run Clustering 때 반영).")
        vis_row.addWidget(self.sparse_minmax_check)
        self.best_spatial_split_check = QCheckBox("best 공간 분리")
        self.best_spatial_split_check.setChecked(True)
        self.best_spatial_split_check.setToolTip(
            "Stage A 끝: best 샘플을 공간 연결성분(26-연결)으로 나눠 샘플이 가장 많은 성분만 best로 남기고, "
            "떨어진 나머지는 best 학습 샘플에서 뺀다(RF not-best로는 쓰지 않음). "
            "한 스트로크 = 대상 하나 전제 (Run Clustering 때 반영).")
        vis_row.addWidget(self.best_spatial_split_check)
        vis_row.addStretch()
        cluster_layout.addLayout(vis_row)

        cluster_group.setLayout(cluster_layout)
        return cluster_group

    def build_debug_group(self):
        """Final / K-means 오버레이 토글(서로 배타), cluster 표시 체크·색 버튼, Z-Slice Viewer 버튼."""
        debug_group = QGroupBox("Debug")
        debug_layout = QVBoxLayout()

        self.show_clusters_button = QPushButton("Show Final Clusters (on main view)")
        self.show_clusters_button.setCheckable(True)
        self.show_clusters_button.toggled.connect(self._on_show_final_toggled)
        debug_layout.addWidget(self.show_clusters_button)

        self.show_kmeans_button = QPushButton("Show K-means (ROI samples, on main view)")
        self.show_kmeans_button.setCheckable(True)
        self.show_kmeans_button.toggled.connect(self._on_show_kmeans_toggled)
        debug_layout.addWidget(self.show_kmeans_button)

        self.show_rf_button = QPushButton("Show RF samples (best red / not-best blue)")
        self.show_rf_button.setCheckable(True)
        self.show_rf_button.setToolTip("마지막 Stage B RF 학습셋 위치. 빨강 = best(Stage A 샘플), "
                                       "파랑 = not-best(볼륨 voxel). cluster 표시 체크와 무관.")
        self.show_rf_button.toggled.connect(self._on_show_rf_toggled)
        debug_layout.addWidget(self.show_rf_button)

        cluster_view_row = QHBoxLayout()
        cluster_view_row.addWidget(QLabel("Cluster view"))
        for text, checked in (("All", True), ("None", False)):
            button = QPushButton(text)
            button.clicked.connect(lambda _checked, on=checked: self.set_all_clusters_visible(on))
            cluster_view_row.addWidget(button)
        debug_layout.addLayout(cluster_view_row)

        self.cluster_swatch_grid = QGridLayout()
        self.cluster_swatch_grid.setAlignment(Qt.AlignmentFlag.AlignLeft)
        debug_layout.addLayout(self.cluster_swatch_grid)

        self.slice_viewer_button = QPushButton("Z-Slice Viewer...")
        self.slice_viewer_button.setToolTip("볼륨 Z-slice 위에 cluster 라벨을 겹쳐 보여준다.")
        self.slice_viewer_button.clicked.connect(self.slice_viewer_requested.emit)
        debug_layout.addWidget(self.slice_viewer_button)

        debug_group.setLayout(debug_layout)
        return debug_group

    # ------------------------------------------------------------------
    # 도구 옵션 페이지
    # ------------------------------------------------------------------
    def create_empty_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Select a WYSIWYG tool."))
        layout.addWidget(QLabel("Then choose ROI and click Preview."))
        layout.addStretch()
        return page

    def create_basic_page(self):
        """Strength / Feather 슬라이더 + Mode 콤보 (eraser, contrast, brightness 공용)."""
        page = QWidget()
        layout = QVBoxLayout(page)

        self.basic_strength_label = QLabel("Strength")
        self.basic_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.basic_strength_slider.setRange(0, 100)
        self.basic_strength_slider.setValue(50)

        self.basic_feather_label = QLabel("Feather")
        self.basic_feather_slider = QSlider(Qt.Orientation.Horizontal)
        self.basic_feather_slider.setRange(0, 100)
        self.basic_feather_slider.setValue(0)

        self.basic_direction_label = QLabel("Mode")
        self.basic_direction_combo = QComboBox()
        self.basic_direction_combo.addItems(["Increase", "Decrease"])

        for widget in (self.basic_strength_label, self.basic_strength_slider,
                       self.basic_feather_label, self.basic_feather_slider,
                       self.basic_direction_label, self.basic_direction_combo):
            layout.addWidget(widget)
        layout.addStretch()
        return page

    def create_colorization_page(self):
        """Blend Strength 슬라이더 + 색 선택."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Colorization Options"))

        self.color_strength_label = QLabel("Blend Strength")
        self.color_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.color_strength_slider.setRange(0, 100)
        self.color_strength_slider.setValue(25)

        self.color_preview_label = QLabel("Selected Color")
        self.color_preview_box = QFrame()
        self.color_preview_box.setFixedHeight(28)
        self.color_preview_box.setStyleSheet(f"background-color: {self.selected_color}; border: 1px solid #888;")

        self.color_pick_button = QPushButton("Choose Color")
        self.color_pick_button.clicked.connect(self.choose_colorization_color)

        for widget in (self.color_strength_label, self.color_strength_slider,
                       self.color_preview_label, self.color_preview_box, self.color_pick_button):
            layout.addWidget(widget)
        layout.addStretch()
        return page

    def choose_colorization_color(self):
        color = QColorDialog.getColor()
        if not color.isValid():
            return
        self.selected_color = color.name()  # "#RRGGBB"
        self.color_preview_box.setStyleSheet(f"background-color: {self.selected_color}; border: 1px solid #888;")

    def select_tool(self, tool_name):
        """도구 버튼 체크 상태와 옵션 페이지를 선택한 도구에 맞춘다."""
        self.current_tool = tool_name
        for name, button in self.tool_buttons.items():
            button.setChecked(name == tool_name)
        self.tool_status_label.setText(f"Tool: {tool_name}")

        if tool_name in BASIC_TOOLS:
            self.option_stack.setCurrentIndex(PAGE_BASIC)
            title = BASIC_TOOL_TITLES[tool_name]
            self.basic_strength_label.setText(f"{title} Strength")
            self.basic_direction_label.setText(f"{title} Mode")
            self.basic_direction_combo.clear()
            self.basic_direction_combo.addItems(["Increase", "Decrease"])
        elif tool_name == "colorization":
            self.option_stack.setCurrentIndex(PAGE_COLOR)
        else:
            self.option_stack.setCurrentIndex(PAGE_EMPTY)

    def get_current_tool(self):
        return self.current_tool

    def get_current_tool_params(self):
        """현재 도구의 옵션값. Returns: 도구 함수에 넘길 kwargs dict"""
        if self.current_tool in BASIC_TOOLS:
            return {
                "strength": self.basic_strength_slider.value() / 100.0,
                "feather": self.basic_feather_slider.value() / 100.0,
                "mode": self.basic_direction_combo.currentText().lower(),
            }
        if self.current_tool == "colorization":
            return {
                "strength": self.color_strength_slider.value() / 100.0,
                "feather": 0.0,
                "mode": "apply",
                "color": self.selected_color,
            }
        return {}

    # ------------------------------------------------------------------
    # 디버그 오버레이
    # ------------------------------------------------------------------
    def _on_show_final_toggled(self, checked):
        if checked:
            self.show_kmeans_button.setChecked(False)
            self.show_rf_button.setChecked(False)
        self.cluster_overlay_toggled.emit(checked)

    def _on_show_kmeans_toggled(self, checked):
        if checked:
            self.show_clusters_button.setChecked(False)
            self.show_rf_button.setChecked(False)
        self.kmeans_overlay_toggled.emit(checked)

    def _on_show_rf_toggled(self, checked):
        if checked:
            self.show_clusters_button.setChecked(False)
            self.show_kmeans_button.setChecked(False)
        self.rf_overlay_toggled.emit(checked)

    def set_cluster_view_options(self, n_clusters: int, best_cluster: int):
        """cluster마다 색 버튼(best에 ★)과 그 아래 표시 체크박스를 만든다. 전부 체크된 상태로 시작(시그널 없음)."""
        self.best_cluster = best_cluster
        self.clear_cluster_view_options()
        for c in range(n_clusters):
            row, col = 2 * (c // 10), c % 10
            button = QPushButton("★" if c == best_cluster else "")
            button.setFixedSize(24, 24)
            button.setToolTip(f"Cluster {c} 색 바꾸기")
            button.clicked.connect(lambda _checked, cluster=c: self.on_cluster_swatch_clicked(cluster))
            self.cluster_swatch_grid.addWidget(button, row, col)
            self.cluster_swatches.append(button)

            check = QCheckBox(str(c))
            check.setChecked(True)
            check.setToolTip(f"Cluster {c} 표시")
            check.toggled.connect(lambda _checked: self.cluster_view_changed.emit())
            self.cluster_swatch_grid.addWidget(check, row + 1, col)
            self.cluster_view_checks.append(check)
        self._paint_cluster_colors()

    def clear_cluster_view_options(self):
        for widget in self.cluster_swatches + self.cluster_view_checks:
            self.cluster_swatch_grid.removeWidget(widget)
            widget.deleteLater()
        self.cluster_swatches = []
        self.cluster_view_checks = []

    def get_visible_clusters(self):
        """체크된 cluster 번호 목록."""
        return [c for c, check in enumerate(self.cluster_view_checks) if check.isChecked()]

    def set_all_clusters_visible(self, visible):
        for check in self.cluster_view_checks:
            check.blockSignals(True)
            check.setChecked(visible)
            check.blockSignals(False)
        self.cluster_view_changed.emit()

    def _paint_cluster_colors(self):
        """색 버튼을 현재 cluster 색으로 칠한다."""
        rgbs = cluster_colors(range(len(self.cluster_swatches)), self.best_cluster)
        for button, rgb in zip(self.cluster_swatches, rgbs):
            color = QColor(*(int(v) for v in rgb))
            star = "white" if color.lightness() < 128 else "black"
            button.setStyleSheet(f"background-color: {color.name()}; color: {star}; border: 1px solid gray;")

    def on_cluster_swatch_clicked(self, cluster):
        rgb = cluster_colors([cluster], self.best_cluster)[0]
        color = QColorDialog.getColor(QColor(*(int(v) for v in rgb)), self, f"Cluster {cluster} 색상 선택")
        if not color.isValid():
            return
        set_cluster_color(cluster, self.best_cluster, (color.red(), color.green(), color.blue()))
        self._paint_cluster_colors()
        self.cluster_color_changed.emit()

    # ------------------------------------------------------------------
    # 설정값 조회
    # ------------------------------------------------------------------
    def get_cluster_k(self):
        return self.cluster_k_spin.value()

    def get_cluster_seed(self):
        return self.cluster_seed_spin.value()

    def get_cluster_n_init(self):
        return self.cluster_n_init_spin.value()

    def get_percentile_range(self):
        """best cluster intensity 구간의 (low, high) percentile. UI 조절은 없앴고 항상 min~max(0, 100)."""
        return 0.0, 100.0

    def get_stage_b_ray_stride_floor(self):
        return self.stage_b_stride_spin.value()

    def get_stage_b_rf(self):
        """Returns: (tree 수, 마진(voxel), P(best) 하한, INR만 여부)"""
        return (self.stage_b_rf_trees_spin.value(), self.stage_b_rf_margin_spin.value(),
                self.stage_b_rf_prob_spin.value(), self.stage_b_rf_inr_only_check.isChecked())

    def get_best_spatial_split(self):
        return self.best_spatial_split_check.isChecked()

    def get_feature_groups(self):
        """Returns: {"inr", "coords", "visibility", "depth", "intensity", "opacity": bool}"""
        groups = {
            "inr": self.feature_inr_check.isChecked(),
            "coords": self.feature_coords_check.isChecked(),
        }
        for name, check in self.local_feature_checks.items():
            groups[name] = check.isChecked()
        return groups

    def get_sparse_minmax(self):
        return self.sparse_minmax_check.isChecked()

    def get_visibility_feature(self):
        """Returns: "V" 또는 "T" (sampling_rules.VISIBILITY_FEATURE 값)"""
        return self.visibility_feature_combo.currentData()

    # ------------------------------------------------------------------
    # 상태 표시
    # ------------------------------------------------------------------
    def reset_local_mask_status(self):
        self.local_mask_status_label.setText("Local mask: not built")

    def set_local_mask_status(self, voxel_count, voxel_fraction, used_groups, warning=None):
        groups_str = ", ".join(used_groups) if used_groups else "-"
        text = f"Local mask: {voxel_count} voxels ({voxel_fraction * 100:.1f}%) via [{groups_str}]"
        if warning:
            text += f"\n⚠ {warning}"
        self.local_mask_status_label.setText(text)

    def set_roi_selected(self, selected: bool):
        self.roi_status_label.setText("ROI: Selected" if selected else "ROI: Not Selected")

    def set_preview_active(self, active: bool):
        self.preview_status_label.setText("Preview: Active" if active else "Preview: Inactive")
