"""X/Y/Z 축별 min/max 볼륨 클리핑 슬라이더 패널."""

from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, QSlider, QGroupBox, QFrame
from PyQt6.QtCore import pyqtSignal, Qt

AXES = ('x', 'y', 'z')


class ClippingPanel(QGroupBox):
    """clipping_ranges: {'x': [min, max], 'y': ..., 'z': ...} (0~1 정규화 값)"""

    clipping_changed = pyqtSignal(str, float, float)   # (axis, min, max)
    clipping_enabled_changed = pyqtSignal(bool)

    def __init__(self):
        super().__init__("Volume Clipping")
        self.clipping_ranges = {axis: [0.0, 1.0] for axis in AXES}
        self.sliders = {}
        self.value_labels = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)

        self.enable_checkbox = QCheckBox("Enable Clipping")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.toggled.connect(self.on_clipping_enabled_changed)
        self.enable_checkbox.setStyleSheet("""
            QCheckBox { spacing: 5px; color: white; }
            QCheckBox::indicator {
                width: 13px; height: 13px;
                border: 1px solid #777;
                background-color: #353535;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked { background-color: #0078d7; }
        """)
        layout.addWidget(self.enable_checkbox)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(line)

        for axis_label, axis_key, direction_label in (('X', 'x', 'Left/Right'),
                                                      ('Y', 'y', 'Front/Back'),
                                                      ('Z', 'z', 'Top/Bottom')):
            layout.addWidget(QLabel(f"{axis_label}-Axis ({direction_label})"))
            for is_min, title, init_value, init_text in ((True, "Min:", 0, "0"),
                                                         (False, "Max:", 1000, "100")):
                row = QHBoxLayout()
                label = QLabel(title)
                label.setFixedWidth(35)
                row.addWidget(label)

                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 1000)
                slider.setValue(init_value)
                slider.setEnabled(False)
                slider.valueChanged.connect(
                    lambda v, ax=axis_key, m=is_min: self.on_slider_changed(ax, m, v))
                key = f"{axis_key}_{'min' if is_min else 'max'}"
                self.sliders[key] = slider
                row.addWidget(slider)

                value_label = QLabel(init_text)
                value_label.setFixedWidth(40)
                value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
                self.value_labels[key] = value_label
                row.addWidget(value_label)
                layout.addLayout(row)

            if axis_key != 'z':
                spacer_line = QFrame()
                spacer_line.setFrameShape(QFrame.Shape.HLine)
                spacer_line.setFrameShadow(QFrame.Shadow.Sunken)
                spacer_line.setStyleSheet("QFrame { margin: 5px 0px; }")
                layout.addWidget(spacer_line)

        layout.addStretch()

    def on_clipping_enabled_changed(self, enabled):
        """슬라이더 활성화를 바꾸고, 켜질 때는 현재 범위를 다시 내보낸다."""
        for slider in self.sliders.values():
            slider.setEnabled(enabled)
        self.clipping_enabled_changed.emit(enabled)
        if enabled:
            for axis in AXES:
                self.clipping_changed.emit(axis, *self.clipping_ranges[axis])

    def on_slider_changed(self, axis, is_min, value):
        """min ≤ max를 유지하며 범위를 갱신하고 clipping_changed를 보낸다."""
        normalized_value = value / 1000.0
        key = f"{axis}_{'min' if is_min else 'max'}"
        other_value = self.sliders[f"{axis}_{'max' if is_min else 'min'}"].value() / 1000.0

        if (is_min and normalized_value > other_value) or (not is_min and normalized_value < other_value):
            normalized_value = other_value
            self.sliders[key].setValue(int(normalized_value * 1000))

        self.clipping_ranges[axis][0 if is_min else 1] = normalized_value
        self.update_value_label(axis, is_min)
        self.clipping_changed.emit(axis, *self.clipping_ranges[axis])

    def update_value_label(self, axis, is_min):
        key = f"{axis}_{'min' if is_min else 'max'}"
        self.value_labels[key].setText(f"{int(self.sliders[key].value() / 10)}%")

    def reset_axis(self, axis):
        self.sliders[f"{axis}_min"].setValue(0)
        self.sliders[f"{axis}_max"].setValue(1000)
        self.clipping_ranges[axis] = [0.0, 1.0]
        self.clipping_changed.emit(axis, 0.0, 1.0)

    def reset_clipping(self):
        """모든 축을 전체 범위로 되돌리고 클리핑을 끈다."""
        for axis in AXES:
            self.reset_axis(axis)
        self.enable_checkbox.setChecked(False)

    def set_volume_shape(self, shape):
        """새 볼륨 로드 시 값 라벨을 갱신한다."""
        for axis in AXES:
            self.update_value_label(axis, True)
            self.update_value_label(axis, False)

    def set_clipping_ranges(self, ranges):
        """프리셋의 {'z': [min, max], ...}(0~1)를 슬라이더에 넣고 클리핑을 켠다. 없는 축은 전체 범위."""
        for axis in AXES:
            lo, hi = ranges.get(axis, (0.0, 1.0))
            self.clipping_ranges[axis] = [float(lo), float(hi)]
            for is_min, value in ((True, lo), (False, hi)):
                slider = self.sliders[f"{axis}_{'min' if is_min else 'max'}"]
                slider.blockSignals(True)
                slider.setValue(int(round(value * 1000)))
                slider.blockSignals(False)
                self.update_value_label(axis, is_min)
        if self.enable_checkbox.isChecked():
            self.on_clipping_enabled_changed(True)     # 이미 켜져 있으면 toggled가 안 오므로 직접 반영
        else:
            self.enable_checkbox.setChecked(True)

    def get_clipping_ranges(self):
        """Returns: 클리핑이 켜져 있으면 {'x': [min, max], ...}, 꺼져 있으면 None"""
        if not self.enable_checkbox.isChecked():
            return None
        return {axis: list(r) for axis, r in self.clipping_ranges.items()}
