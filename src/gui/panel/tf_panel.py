"""좌측 TF 패널: 편집 대상(Global / Local N) 선택, TF 곡선 위젯, 배경색, 데이터셋 프리셋, Shading, Clipping."""

import json
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QCheckBox, QLabel,
    QPushButton, QColorDialog, QSlider,
    QFileDialog, QMessageBox, QWidget,
    QSizePolicy, QComboBox, QGridLayout, QButtonGroup, QDoubleSpinBox
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

from src.core.dataset_presets import DATASET_PRESETS
from src.gui.panel.base_panel import BasePanel
from src.gui.widget.transfer_function_widget import TransferFunctionWidget
from src.gui.widget.light_sphere_widget import LightSphereWidget
from src.gui.panel.clipping_panel import ClippingPanel


class TransferFunctionPanel(BasePanel):
    """TF·렌더링 설정 UI. 값 변경은 시그널로만 내보내고 실제 적용은 main_window가 한다."""

    tf_changed = pyqtSignal(object)                  # TF node list
    background_color_changed = pyqtSignal(tuple, tuple)
    shading_changed = pyqtSignal(bool)
    ambient_color_changed = pyqtSignal(float, float, float)
    diffuse_color_changed = pyqtSignal(float, float, float)
    specular_color_changed = pyqtSignal(float, float, float)
    lighting_changed = pyqtSignal(str, float)        # ('ambient'|'diffuse'|'specular', 값)
    light_direction_changed = pyqtSignal(float, float, float)
    follow_camera_changed = pyqtSignal(bool)
    clipping_changed = pyqtSignal(str, float, float)
    clipping_enabled_changed = pyqtSignal(bool)
    target_selected = pyqtSignal(str, object)        # ("global", None) | ("local", label_id)
    dataset_preset_requested = pyqtSignal(str)       # DATASET_PRESETS key
    local_class_delete_requested = pyqtSignal(int)   # label_id
    # Experiment 그룹 (INR vs intensity 비교 촬영)
    experiment_roi_requested = pyqtSignal(str)       # 프리셋 "rois"의 ROI 이름 → SAM 포인트 로드
    experiment_camera_requested = pyqtSignal(str)    # "main" | "another"
    view_mode_requested = pyqtSignal(str)            # VIEW_MODES 키
    capture_requested = pyqtSignal()
    roi_color_changed = pyqtSignal(tuple)            # (r, g, b) 0~1 — 모든 ROI(local TF) 색을 이 단색으로
    # 보기 모드: 저장된 TF는 그대로 두고 렌더링만 바꾼다 (main_window.apply_view_mode)
    VIEW_MODES = (("default", "기본"), ("roi_off", "ROI 숨김"), ("rest_off", "ROI만"), ("roi_boost", "ROI 강조"))

    def __init__(self):
        self.clipping_panel = None
        self.volume_data = None
        super().__init__("Transfer Function", collapsible=False)

    def setup_content(self):
        self.content_layout.setSpacing(5)
        header_layout = QHBoxLayout()
        self.title_label = QLabel("🎨 Transfer Function")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px 5px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.content_layout.addLayout(header_layout)

        # 편집 대상 선택 + 선택한 Local 삭제
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Editing:"))
        self.tf_target_combo = QComboBox()
        self.tf_target_combo.addItem("Global", "global")
        self.tf_target_combo.currentIndexChanged.connect(self.on_target_combo_changed)
        target_row.addWidget(self.tf_target_combo, 1)
        self.delete_local_btn = QPushButton("Delete")
        self.delete_local_btn.setToolTip("선택한 Local class의 mask와 TF를 삭제합니다")
        self.delete_local_btn.setEnabled(False)
        self.delete_local_btn.clicked.connect(self.on_delete_local_clicked)
        target_row.addWidget(self.delete_local_btn)
        self.content_layout.addLayout(target_row)

        self.tf_widget = TransferFunctionWidget()
        self.tf_widget.tf_changed.connect(self.on_tf_widget_changed)
        self.content_layout.addWidget(self.tf_widget)

        tf_control_layout, _ = self.create_button_horizontal([
            {"text": "Reset", "callback": self.reset_tf, "height": 30},
            {"text": "Save", "callback": self.save_tf, "height": 30},
            {"text": "Load", "callback": self.load_tf, "height": 30}
        ])
        self.content_layout.addLayout(tf_control_layout)

        bg_group, bg_layout = self.create_group_box("Background", "horizontal")
        self.bg_color_btn = QPushButton()
        self.bg_color_btn.setStyleSheet("background-color: #FFFFFF;")
        self.bg_color_btn.setMinimumHeight(25)
        self.bg_color_btn.clicked.connect(self.select_background_color)
        bg_layout.addWidget(self.bg_color_btn)
        reset_bg_btn = QPushButton("Reset")
        reset_bg_btn.clicked.connect(self.reset_background_color)
        reset_bg_btn.setMaximumWidth(60)
        bg_layout.addWidget(reset_bg_btn)
        self.content_layout.addWidget(bg_group)

        self.setup_dataset_presets()
        self.setup_experiment_controls()
        self.setup_shading_controls()
        self.setup_clipping_controls()
        self.content_layout.addStretch(1)

    def setup_dataset_presets(self):
        """DATASET_PRESETS 항목마다 버튼을 3줄 그리드로 만든다."""
        preset_group, preset_layout = self.create_group_box("Dataset Presets", "vertical")
        grid = QGridLayout()
        grid.setSpacing(5)
        per_row = (len(DATASET_PRESETS) + 2) // 3
        for i, (key, preset) in enumerate(DATASET_PRESETS.items()):
            btn = QPushButton(preset["label"])
            btn.clicked.connect(lambda checked=False, k=key: self.dataset_preset_requested.emit(k))
            grid.addWidget(btn, i // per_row, i % per_row)
        preset_layout.addLayout(grid)
        self.content_layout.addWidget(preset_group)

    def setup_experiment_controls(self):
        """INR vs intensity 비교 촬영: ROI(SAM 포인트) 선택, 기본/측면 카메라, 보기 모드, 조건 이름으로 촬영 저장."""
        group, layout = self.create_group_box("Experiment", "vertical")

        roi_row = QHBoxLayout()
        roi_row.addWidget(QLabel("ROI:"))
        self.exp_roi_combo = QComboBox()
        self.exp_roi_combo.setToolTip("프리셋의 ROI(SAM 포인트 + 측면 카메라). 바꾸면 포인트를 불러온다")
        self.exp_roi_combo.activated.connect(
            lambda _i: self.experiment_roi_requested.emit(self.exp_roi_combo.currentText()))
        roi_row.addWidget(self.exp_roi_combo, 1)
        self.exp_cam_main_btn = QPushButton("main cam")
        self.exp_cam_main_btn.clicked.connect(lambda: self.experiment_camera_requested.emit("main"))
        roi_row.addWidget(self.exp_cam_main_btn)
        self.exp_cam_another_btn = QPushButton("another cam")
        self.exp_cam_another_btn.setToolTip("이 ROI를 측면에서 보는 카메라")
        self.exp_cam_another_btn.clicked.connect(lambda: self.experiment_camera_requested.emit("another"))
        roi_row.addWidget(self.exp_cam_another_btn)
        layout.addLayout(roi_row)

        mode_row = QHBoxLayout()
        self.view_mode_group = QButtonGroup(self)
        self.view_mode_group.setExclusive(True)
        self.view_mode_buttons = {}
        for key, text in self.VIEW_MODES:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.clicked.connect(lambda _c=False, k=key: self.view_mode_requested.emit(k))
            self.view_mode_group.addButton(btn)
            self.view_mode_buttons[key] = btn
            mode_row.addWidget(btn)
        self.view_mode_buttons["default"].setChecked(True)
        self.view_mode_buttons["roi_off"].setToolTip("ROI(local) opacity 0 — 남은 부분으로 놓친 구조 확인")
        self.view_mode_buttons["rest_off"].setToolTip("ROI 밖(global) opacity 0 — 분리된 구조만")
        self.view_mode_buttons["roi_boost"].setToolTip(
            "ROI의 best intensity 구간 tent(꼭짓점 = 강조값)를 local TF에 max로 더한다 (Unity 슬라이더를 올린 것과 같음)"
            " — 투명 구간에 숨은 과분할 확인")
        self.boost_spin = QDoubleSpinBox()
        self.boost_spin.setRange(0.05, 1.0)
        self.boost_spin.setSingleStep(0.05)
        self.boost_spin.setValue(0.5)
        self.boost_spin.setToolTip("ROI 강조 tent 꼭짓점 opacity (실험 내내 같은 값으로)")
        self.boost_spin.valueChanged.connect(self.on_boost_changed)
        mode_row.addWidget(self.boost_spin)
        layout.addLayout(mode_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("ROI 색:"))
        self.roi_color = (0.0, 0.0, 1.0)   # 기본 파랑
        self.roi_color_btn = QPushButton()
        self.roi_color_btn.setMinimumHeight(22)
        self.roi_color_btn.setToolTip("ROI(local TF) 전체를 칠할 단색 — 바꾸면 지금 있는 ROI에도 적용")
        self.roi_color_btn.clicked.connect(self.select_roi_color)
        color_row.addWidget(self.roi_color_btn, 1)
        self.auto_roi_color_check = QCheckBox("새 ROI에 자동 적용")
        self.auto_roi_color_check.setChecked(True)
        color_row.addWidget(self.auto_roi_color_check)
        layout.addLayout(color_row)
        self._update_roi_color_button()

        cap_row = QHBoxLayout()
        cap_btn = QPushButton("📸 촬영 저장")
        cap_btn.setToolTip("resources/Experiments/captures/<데이터셋>/<ROI>/ 에 조건 이름으로 저장 (captures.csv에 기록)")
        cap_btn.clicked.connect(self.capture_requested.emit)
        cap_row.addWidget(cap_btn)
        self.capture_status_label = QLabel("")
        self.capture_status_label.setStyleSheet("color: #aaa;")
        cap_row.addWidget(self.capture_status_label, 1)
        layout.addLayout(cap_row)

        self.content_layout.addWidget(group)
        self.set_experiment_rois(None)

    def set_experiment_rois(self, rois):
        """프리셋 "rois"(없으면 None)로 ROI 콤보를 채운다. 첫 ROI가 선택된다."""
        self.exp_rois = rois or {}
        self.exp_roi_combo.clear()
        self.exp_roi_combo.addItems(list(self.exp_rois))
        has = bool(self.exp_rois)
        for w in (self.exp_roi_combo, self.exp_cam_main_btn, self.exp_cam_another_btn):
            w.setEnabled(has)

    def set_view_mode(self, mode):
        """보기 모드 버튼 표시만 바꾼다 (시그널 없음)."""
        btn = self.view_mode_buttons.get(mode)
        if btn is not None:
            btn.setChecked(True)

    def get_boost_strength(self):
        return float(self.boost_spin.value())

    def on_boost_changed(self, _value):
        if self.view_mode_buttons["roi_boost"].isChecked():
            self.view_mode_requested.emit("roi_boost")

    def set_capture_status(self, text):
        self.capture_status_label.setText(text)

    def select_roi_color(self):
        r, g, b = (int(round(c * 255)) for c in self.roi_color)
        color = QColorDialog.getColor(QColor(r, g, b), self, "ROI 색")
        if not color.isValid():
            return
        self.roi_color = (color.redF(), color.greenF(), color.blueF())
        self._update_roi_color_button()
        self.roi_color_changed.emit(self.roi_color)

    def _update_roi_color_button(self):
        r, g, b = (int(round(c * 255)) for c in self.roi_color)
        self.roi_color_btn.setStyleSheet(f"background-color: rgb({r}, {g}, {b});")

    def get_auto_roi_color(self):
        """Returns: 새 ROI에 칠할 (r, g, b), 자동 적용이 꺼져 있으면 None"""
        return self.roi_color if self.auto_roi_color_check.isChecked() else None

    def setup_shading_controls(self):
        """접이식 Shading 섹션: on/off, 광원 방향 구, ambient/diffuse/specular 슬라이더·색, Follow Camera."""
        shading_widget = QWidget()
        shading_widget.setStyleSheet("""
            QWidget {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: rgba(64, 64, 64, 50);
            }
        """)
        shading_layout = QVBoxLayout(shading_widget)
        shading_layout.setContentsMargins(0, 0, 0, 0)
        shading_layout.setSpacing(0)

        self.shade_header_btn = QPushButton("▶ Shading Controls")
        self.shade_header_btn.setCheckable(True)
        self.shade_header_btn.setChecked(False)
        self.shade_header_btn.clicked.connect(self.toggle_shading_section)
        self.shade_header_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                border: none;
                background-color: #404040;
                color: white;
                padding: 10px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #505050; }
            QPushButton:checked { background-color: #606060; }
        """)
        shading_layout.addWidget(self.shade_header_btn)

        self.shade_content_widget = QWidget()
        self.shade_content_widget.setVisible(False)
        self.shade_content_widget.setStyleSheet("QWidget { border: none; background-color: transparent; }")
        shade_content_layout = QVBoxLayout(self.shade_content_widget)
        shade_content_layout.setContentsMargins(10, 10, 10, 10)
        shade_content_layout.setSpacing(8)

        self.shade_toggle = QCheckBox("Enable Shading")
        self.shade_toggle.setChecked(False)
        self.shade_toggle.stateChanged.connect(self.on_shading_changed)
        self.shade_toggle.setStyleSheet("""
            QCheckBox { spacing: 5px; color: white; }
            QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #777; background-color: #353535; border-radius: 3px; }
            QCheckBox::indicator:checked { background-color: #0078d7; }
        """)
        shade_content_layout.addWidget(self.shade_toggle)

        lighting_widget = QWidget()
        lighting_main_layout = QHBoxLayout(lighting_widget)
        lighting_main_layout.setContentsMargins(20, 0, 0, 0)
        lighting_main_layout.setSpacing(15)

        self.light_sphere = LightSphereWidget()
        self.light_sphere.setEnabled(False)
        self.light_sphere.setFixedSize(80, 80)
        self.light_sphere.light_changed.connect(self.light_direction_changed.emit)
        lighting_main_layout.addWidget(self.light_sphere)

        sliders_widget = QWidget()
        sliders_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)
        sliders_layout.setSpacing(5)

        self.ambient_slider, self.ambient_label, self.ambient_color_btn = self.create_lighting_row(
            sliders_layout, "Ambient:", 40, self.on_ambient_changed, self.on_ambient_color_clicked)
        self.diffuse_slider, self.diffuse_label, self.diffuse_color_btn = self.create_lighting_row(
            sliders_layout, "Diffuse:", 60, self.on_diffuse_changed, self.on_diffuse_color_clicked, max_val=500)
        self.specular_slider, self.specular_label, self.specular_color_btn = self.create_lighting_row(
            sliders_layout, "Specular:", 20, self.on_specular_changed, self.on_specular_color_clicked)

        self.follow_camera_checkbox = QCheckBox("Follow Camera")
        self.follow_camera_checkbox.setChecked(False)
        self.follow_camera_checkbox.setEnabled(False)
        self.follow_camera_checkbox.toggled.connect(self.follow_camera_changed.emit)
        self.follow_camera_checkbox.setStyleSheet(self.shade_toggle.styleSheet())
        sliders_layout.addWidget(self.follow_camera_checkbox)

        lighting_main_layout.addWidget(sliders_widget, 1)
        shade_content_layout.addWidget(lighting_widget)
        shading_layout.addWidget(self.shade_content_widget)
        self.content_layout.addWidget(shading_widget)

    def create_lighting_row(self, parent_layout, label_text, default_val, slider_callback, color_callback, max_val=100):
        """라벨 + 슬라이더(값/100) + 값 표시 + 색 버튼 한 줄. Returns: (slider, value_label, color_btn)"""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        label_fixed = QLabel(label_text)
        label_fixed.setMinimumWidth(60)
        layout.addWidget(label_fixed)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, max_val)
        slider.setValue(default_val)
        slider.valueChanged.connect(slider_callback)
        slider.setEnabled(False)
        layout.addWidget(slider, 1)

        value_label = QLabel(f"{default_val/100:.1f}")
        value_label.setMinimumWidth(30)
        layout.addWidget(value_label)

        color_btn = QPushButton()
        color_btn.setFixedSize(20, 20)
        color_btn.setStyleSheet("background-color: white; border: 1px solid gray;")
        color_btn.clicked.connect(color_callback)
        color_btn.setEnabled(False)
        layout.addWidget(color_btn)

        parent_layout.addWidget(widget)
        return slider, value_label, color_btn

    def setup_clipping_controls(self):
        """접이식 Clipping 섹션(ClippingPanel 포함)."""
        clipping_widget = QWidget()
        clipping_widget.setStyleSheet("""
            QWidget {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: rgba(64, 64, 64, 50);
            }
        """)
        clipping_layout = QVBoxLayout(clipping_widget)
        clipping_layout.setContentsMargins(0, 0, 0, 0)
        clipping_layout.setSpacing(0)

        self.clip_header_btn = QPushButton("▶ Clipping Controls")
        self.clip_header_btn.setCheckable(True)
        self.clip_header_btn.setChecked(False)
        self.clip_header_btn.clicked.connect(self.toggle_clipping_section)
        self.clip_header_btn.setStyleSheet(self.shade_header_btn.styleSheet())
        clipping_layout.addWidget(self.clip_header_btn)

        self.clip_content_widget = QWidget()
        self.clip_content_widget.setVisible(False)
        self.clip_content_widget.setStyleSheet("QWidget { border: none; background-color: transparent; }")
        clip_content_layout = QVBoxLayout(self.clip_content_widget)
        clip_content_layout.setContentsMargins(10, 10, 10, 10)
        clip_content_layout.setSpacing(10)

        self.clipping_panel = ClippingPanel()
        self.clipping_panel.setTitle("")
        self.clipping_panel.setStyleSheet("QGroupBox {border: none; margin-top: 0; padding-top: 0;}")
        self.clipping_panel.clipping_changed.connect(self.clipping_changed.emit)
        self.clipping_panel.clipping_enabled_changed.connect(self.clipping_enabled_changed.emit)

        clip_content_layout.addWidget(self.clipping_panel)
        clipping_layout.addWidget(self.clip_content_widget)
        self.content_layout.addWidget(clipping_widget)

    def toggle_shading_section(self):
        """Shading 섹션 펼치기/접기. 펼치면 Clipping 섹션은 접는다."""
        is_expanded = self.shade_header_btn.isChecked()
        if is_expanded and self.clip_header_btn.isChecked():
            self.clip_header_btn.setChecked(False)
            self.clip_content_widget.setVisible(False)
            self.clip_header_btn.setText("▶ Clipping Controls")
        self.shade_header_btn.setText(f"{'▼' if is_expanded else '▶'} Shading Controls")
        self.shade_content_widget.setVisible(is_expanded)

    def toggle_clipping_section(self):
        """Clipping 섹션 펼치기/접기. 펼치면 Shading 섹션은 접는다."""
        is_expanded = self.clip_header_btn.isChecked()
        if is_expanded and self.shade_header_btn.isChecked():
            self.shade_header_btn.setChecked(False)
            self.shade_content_widget.setVisible(False)
            self.shade_header_btn.setText("▶ Shading Controls")
        self.clip_header_btn.setText(f"{'▼' if is_expanded else '▶'} Clipping Controls")
        self.clip_content_widget.setVisible(is_expanded)

    def set_volume_data(self, volume_data):
        """TF 위젯 히스토그램과 클리핑 패널에 새 볼륨을 반영한다."""
        self.volume_data = volume_data
        if volume_data is not None:
            self.tf_widget.set_volume_data(volume_data)
            if self.clipping_panel:
                self.clipping_panel.set_volume_shape(volume_data.shape)

    # ------------------------------------------------------------------
    # 편집 대상(Global / Local N)
    # ------------------------------------------------------------------
    def get_current_target(self):
        """Returns: ("global", None) 또는 ("local", label_id)"""
        data = self.tf_target_combo.currentData()
        if not data or data == "global":
            return ("global", None)
        _, label_str = data.split(":", 1)
        return ("local", int(label_str))

    def sync_delete_button(self):
        """Delete 버튼은 Local 선택 중에만 활성화."""
        self.delete_local_btn.setEnabled(self.get_current_target()[0] == "local")

    def on_delete_local_clicked(self):
        kind, label_id = self.get_current_target()
        if kind == "local" and label_id is not None:
            self.local_class_delete_requested.emit(int(label_id))

    def set_local_tf_slots(self, label_ids):
        """선택창을 Global + Local {label_ids}로 다시 구성한다.

        이전 선택이 남아 있으면 유지하고 없으면 Global로 돌아간 뒤, target_selected를 한 번 보낸다.
        """
        current_kind, current_label = self.get_current_target()
        self.tf_target_combo.blockSignals(True)
        self.tf_target_combo.clear()
        self.tf_target_combo.addItem("Global", "global")
        for lid in label_ids:
            self.tf_target_combo.addItem(f"Local {lid}", f"local:{lid}")
        target_key = "global" if current_kind == "global" else f"local:{current_label}"
        idx = self.tf_target_combo.findData(target_key)
        self.tf_target_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.tf_target_combo.blockSignals(False)
        self.sync_delete_button()
        self.target_selected.emit(*self.get_current_target())

    def set_current_target(self, kind, label_id=None):
        """선택창을 지정한 대상으로 바꾼다(목록에 있을 때만)."""
        key = "global" if kind == "global" else f"local:{label_id}"
        idx = self.tf_target_combo.findData(key)
        if idx >= 0:
            self.tf_target_combo.setCurrentIndex(idx)
        self.sync_delete_button()

    def on_target_combo_changed(self, idx):
        if idx < 0:
            return
        self.sync_delete_button()
        self.target_selected.emit(*self.get_current_target())

    # ------------------------------------------------------------------
    # TF 편집 / 저장 / 로드
    # ------------------------------------------------------------------
    def on_tf_widget_changed(self):
        self.tf_changed.emit(self.tf_widget.get_nodes())

    def reset_tf(self):
        self.tf_widget.reset_to_default()
        self.tf_changed.emit(self.tf_widget.get_nodes())

    def save_tf(self):
        """현재 TF 곡선과 shading 설정을 JSON으로 저장한다."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transfer Function", "./resources/TFs/TF_global.json",
            "JSON Files (*.json);;All Files (*)")
        if not file_path:
            return
        if not file_path.lower().endswith('.json'):
            file_path += '.json'

        try:
            export_data = {
                'global': self.tf_widget.get_nodes(),
                'shading': {
                    'enabled': self.shade_toggle.isChecked(),
                    'ambient': self.ambient_slider.value() / 100.0,
                    'diffuse': self.diffuse_slider.value() / 100.0,
                    'specular': self.specular_slider.value() / 100.0
                },
                'metadata': {'version': '3.0'}
            }
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            self.emit_status(f"Transfer function saved: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {str(e)}")

    def load_tf(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Transfer Function", "./resources/TFs", "JSON Files (*.json);;All Files (*)")
        if file_path:
            self.load_tf_from_path(file_path)

    def load_tf_from_path(self, file_path):
        """TF JSON의 shading 설정과 'global' TF 곡선을 적용한다."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)

            if 'shading' in import_data:
                shading_data = import_data['shading']
                shading_enabled = shading_data.get('enabled', False)

                self.shade_toggle.blockSignals(True)
                self.shade_toggle.setChecked(shading_enabled)
                self.shade_toggle.blockSignals(False)
                self.set_lighting_sliders_enabled(shading_enabled)

                self.ambient_slider.setValue(int(shading_data.get('ambient', 0.4) * 100))
                self.diffuse_slider.setValue(int(shading_data.get('diffuse', 0.6) * 100))
                self.specular_slider.setValue(int(shading_data.get('specular', 0.2) * 100))
                self.shading_changed.emit(shading_enabled)

            if 'global' in import_data:
                self.tf_widget.set_transfer_function_from_array(import_data['global'])
                self.tf_changed.emit(import_data['global'])

            self.emit_status(f"Transfer function loaded: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    # ------------------------------------------------------------------
    # 배경 / 조명
    # ------------------------------------------------------------------
    def select_background_color(self):
        color = QColorDialog.getColor(QColor(255, 255, 255), self, "배경색 선택")
        if color.isValid():
            bg_color = (color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0)
            self.bg_color_btn.setStyleSheet(f"background-color: {color.name()};")
            self.background_color_changed.emit(bg_color, bg_color)

    def reset_background_color(self):
        bg = (1.0, 1.0, 1.0)
        self.bg_color_btn.setStyleSheet("background-color: #FFFFFF;")
        self.background_color_changed.emit(bg, bg)

    def on_shading_changed(self, state):
        enabled = bool(state)
        self.set_lighting_sliders_enabled(enabled)
        self.shading_changed.emit(enabled)

    def set_lighting_sliders_enabled(self, enabled):
        for widget in (self.ambient_slider, self.diffuse_slider, self.specular_slider,
                       self.light_sphere, self.follow_camera_checkbox,
                       self.ambient_color_btn, self.diffuse_color_btn, self.specular_color_btn):
            widget.setEnabled(enabled)

    def on_ambient_changed(self, value):
        self.ambient_label.setText(f"{value / 100.0:.1f}")
        self.lighting_changed.emit("ambient", value / 100.0)

    def on_diffuse_changed(self, value):
        self.diffuse_label.setText(f"{value / 100.0:.1f}")
        self.lighting_changed.emit("diffuse", value / 100.0)

    def on_specular_changed(self, value):
        self.specular_label.setText(f"{value / 100.0:.1f}")
        self.lighting_changed.emit("specular", value / 100.0)

    def _pick_light_color(self, title, button, signal):
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, title)
        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()}; border: 1px solid gray;")
            signal.emit(color.redF(), color.greenF(), color.blueF())

    def on_ambient_color_clicked(self):
        self._pick_light_color("Select Ambient Color", self.ambient_color_btn, self.ambient_color_changed)

    def on_diffuse_color_clicked(self):
        self._pick_light_color("Select Diffuse Color", self.diffuse_color_btn, self.diffuse_color_changed)

    def on_specular_color_clicked(self):
        self._pick_light_color("Select Specular Color", self.specular_color_btn, self.specular_color_changed)

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------
    def reset_clipping_safe(self):
        if self.clipping_panel is not None:
            self.clipping_panel.reset_clipping()

    def set_clipping_ranges(self, ranges):
        if self.clipping_panel is not None:
            self.clipping_panel.set_clipping_ranges(ranges)

    def get_clipping_ranges(self):
        """Returns: ClippingPanel.get_clipping_ranges() 결과 또는 None"""
        if self.clipping_panel:
            return self.clipping_panel.get_clipping_ranges()
        return None
