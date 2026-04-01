from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox,
    QLabel, QPushButton, QStackedWidget, QSlider, QDoubleSpinBox,
    QComboBox, QColorDialog, QFrame
)
from PyQt6.QtCore import Qt


class WysiwygPanel(QWidget):
    preview_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    tool_changed = pyqtSignal(str)

    def __init__(self, parent=None, tf_histogram_widget=None):
        super().__init__(parent)

        self.current_tool = None
        self.tf_histogram_widget = tf_histogram_widget
        self.selected_color = "#ff0000"

        self._build_ui()

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # -------------------------
        # 1. Status Group
        # -------------------------
        status_group = QGroupBox("WYSIWYG Status")
        status_layout = QVBoxLayout()

        self.roi_status_label = QLabel("ROI: Not Selected")
        self.tool_status_label = QLabel("Tool: None")
        self.preview_status_label = QLabel("Preview: Inactive")

        status_layout.addWidget(self.roi_status_label)
        status_layout.addWidget(self.tool_status_label)
        status_layout.addWidget(self.preview_status_label)
        status_group.setLayout(status_layout)

        main_layout.addWidget(status_group)

        # -------------------------
        # 2. Tool Select Group
        # -------------------------
        tool_group = QGroupBox("Tools")
        tool_layout = QGridLayout()

        self.tool_buttons = {}

        tool_names = [
            "eraser", "colorization", "rainbow", "contrast",
            "brightness", "silhouette", "fuzziness", "peeling"
        ]

        for idx, tool_name in enumerate(tool_names):
            button = QPushButton(tool_name.capitalize())
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, name=tool_name: self.select_tool(name)
            )
            row = idx // 2
            col = idx % 2
            tool_layout.addWidget(button, row, col)
            self.tool_buttons[tool_name] = button

        tool_group.setLayout(tool_layout)
        main_layout.addWidget(tool_group)

        # -------------------------
        # 3. Option Stack
        # -------------------------
        option_group = QGroupBox("Tool Options")
        option_layout = QVBoxLayout()

        self.option_stack = QStackedWidget()

        self.page_empty = self._create_empty_page()
        self.page_basic = self._create_basic_page()
        self.page_colorization = self._create_colorization_page()
        self.page_rainbow = self._create_rainbow_page()
        self.page_peeling = self._create_peeling_page()

        self.option_stack.addWidget(self.page_empty)         # index 0
        self.option_stack.addWidget(self.page_basic)         # index 1
        self.option_stack.addWidget(self.page_colorization)  # index 2
        self.option_stack.addWidget(self.page_rainbow)       # index 3
        self.option_stack.addWidget(self.page_peeling)       # index 4

        option_layout.addWidget(self.option_stack)
        option_group.setLayout(option_layout)

        main_layout.addWidget(option_group)

        # -------------------------
        # 4. Action Buttons
        # -------------------------
        action_group = QGroupBox("Actions")
        action_layout = QHBoxLayout()

        self.preview_button = QPushButton("Preview")
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset")

        self.preview_button.clicked.connect(self.preview_requested.emit)
        self.apply_button.clicked.connect(self.apply_requested.emit)
        self.reset_button.clicked.connect(self.reset_requested.emit)

        action_layout.addWidget(self.preview_button)
        action_layout.addWidget(self.apply_button)
        action_layout.addWidget(self.reset_button)

        action_group.setLayout(action_layout)
        main_layout.addWidget(action_group)

        main_layout.addStretch()

    def _create_empty_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(QLabel("Select a WYSIWYG tool."))
        layout.addWidget(QLabel("Then choose ROI and click Preview."))
        layout.addStretch()
        return page

    def _create_basic_page(self):
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

        layout.addWidget(self.basic_strength_label)
        layout.addWidget(self.basic_strength_slider)
        layout.addWidget(self.basic_feather_label)
        layout.addWidget(self.basic_feather_slider)
        layout.addWidget(self.basic_direction_label)
        layout.addWidget(self.basic_direction_combo)

        layout.addStretch()
        return page

    def _create_colorization_page(self):
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
        self.color_preview_box.setStyleSheet(
            f"background-color: {self.selected_color}; border: 1px solid #888;"
        )

        self.color_pick_button = QPushButton("Choose Color")
        self.color_pick_button.clicked.connect(self._choose_colorization_color)

        layout.addWidget(self.color_strength_label)
        layout.addWidget(self.color_strength_slider)
        layout.addWidget(self.color_preview_label)
        layout.addWidget(self.color_preview_box)
        layout.addWidget(self.color_pick_button)

        layout.addStretch()
        return page

    def _create_rainbow_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Rainbow Options"))

        self.rainbow_preset_combo = QComboBox()
        self.rainbow_preset_combo.addItems(["Rainbow", "Heat", "Cool", "Custom"])

        self.rainbow_strength_label = QLabel("Strength")
        self.rainbow_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.rainbow_strength_slider.setRange(0, 100)
        self.rainbow_strength_slider.setValue(50)

        layout.addWidget(QLabel("Preset"))
        layout.addWidget(self.rainbow_preset_combo)
        layout.addWidget(self.rainbow_strength_label)
        layout.addWidget(self.rainbow_strength_slider)
        layout.addStretch()
        return page

    def _create_peeling_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(QLabel("Peeling Options"))

        self.peeling_depth_label = QLabel("Layer Depth")
        self.peeling_depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.peeling_depth_slider.setRange(0, 100)
        self.peeling_depth_slider.setValue(30)

        layout.addWidget(self.peeling_depth_label)
        layout.addWidget(self.peeling_depth_slider)
        layout.addStretch()
        return page
    
    def _choose_colorization_color(self):
        color = QColorDialog.getColor()

        if not color.isValid():
            return

        self.selected_color = color.name()  # "#RRGGBB"
        self.color_preview_box.setStyleSheet(
            f"background-color: {self.selected_color}; border: 1px solid #888;"
        )

    def select_tool(self, tool_name):
        self.current_tool = tool_name

        for name, button in self.tool_buttons.items():
            button.setChecked(name == tool_name)

        self.tool_status_label.setText(f"Tool: {tool_name}")

        if tool_name in ["eraser", "contrast", "brightness", "silhouette", "fuzziness"]:
            self.option_stack.setCurrentIndex(1)
            self._update_basic_page_for_tool(tool_name)
        elif tool_name == "colorization":
            self.option_stack.setCurrentIndex(2)
        elif tool_name == "rainbow":
            self.option_stack.setCurrentIndex(3)
        elif tool_name == "peeling":
            self.option_stack.setCurrentIndex(4)
        else:
            self.option_stack.setCurrentIndex(0)

        self.tool_changed.emit(tool_name)

    def _update_basic_page_for_tool(self, tool_name):
        if tool_name == "eraser":
            self.basic_strength_label.setText("Visibility Strength")
            self.basic_direction_label.setText("Visibility Mode")
            self.basic_direction_combo.clear()
            self.basic_direction_combo.addItems(["Increase", "Decrease"])

        elif tool_name == "brightness":
            self.basic_strength_label.setText("Brightness Strength")
            self.basic_direction_label.setText("Brightness Mode")
            self.basic_direction_combo.clear()
            self.basic_direction_combo.addItems(["Increase", "Decrease"])

        elif tool_name == "contrast":
            self.basic_strength_label.setText("Contrast Strength")
            self.basic_direction_label.setText("Contrast Mode")
            self.basic_direction_combo.clear()
            self.basic_direction_combo.addItems(["Increase", "Decrease"])

        elif tool_name == "silhouette":
            self.basic_strength_label.setText("Silhouette Strength")
            self.basic_direction_label.setText("Silhouette Mode")
            self.basic_direction_combo.clear()
            self.basic_direction_combo.addItems(["Enhance", "Reduce"])

        elif tool_name == "fuzziness":
            self.basic_strength_label.setText("Fuzziness Strength")
            self.basic_direction_label.setText("Fuzziness Mode")
            self.basic_direction_combo.clear()
            self.basic_direction_combo.addItems(["More", "Less"])

    def get_current_tool(self):
        return self.current_tool

    def get_current_tool_params(self):
        tool = self.current_tool

        if tool in ["eraser", "contrast", "brightness", "silhouette", "fuzziness"]:
            return {
                "strength": self.basic_strength_slider.value() / 100.0,
                "feather": self.basic_feather_slider.value() / 100.0,
                "mode": self.basic_direction_combo.currentText().lower(),
            }

        elif tool == "colorization":
            return {
            "strength": self.color_strength_slider.value() / 100.0,
            "feather": 0.0,
            "mode": "apply",
            "color": self.selected_color,
            }

        elif tool == "rainbow":
            return {
                "preset": self.rainbow_preset_combo.currentText().lower(),
                "strength": self.rainbow_strength_slider.value() / 100.0,
            }

        elif tool == "peeling":
            return {
                "depth": self.peeling_depth_slider.value() / 100.0,
            }

        return {}

    def set_roi_selected(self, selected: bool):
        if selected:
            self.roi_status_label.setText("ROI: Selected")
        else:
            self.roi_status_label.setText("ROI: Not Selected")

    def set_preview_active(self, active: bool):
        if active:
            self.preview_status_label.setText("Preview: Active")
        else:
            self.preview_status_label.setText("Preview: Inactive")