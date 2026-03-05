"""
간결화된 Transfer Function 패널 
- Global TF만 지원 (Class-specific 모드 제거)
- Shading, Clipping 컨트롤 유지
"""

import json
import numpy as np
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QCheckBox, QLabel, 
    QPushButton, QColorDialog, QSlider,
    QFileDialog, QMessageBox, QWidget, QGroupBox,
    QSizePolicy
)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor

from src.gui.panel.base_panel import BasePanel
from src.gui.widget.transfer_function_widget import TransferFunctionWidget
from src.gui.widget.light_sphere_widget import LightSphereWidget
from src.gui.panel.clipping_panel import ClippingPanel


class TransferFunctionPanel(BasePanel):
    """간결화된 Transfer Function 패널 - Global TF만 지원"""
    
    # 시그널 정의
    tf_changed = pyqtSignal(object)
    background_color_changed = pyqtSignal(tuple, tuple)
    shading_changed = pyqtSignal(bool)
    ambient_color_changed = pyqtSignal(float, float, float)
    diffuse_color_changed = pyqtSignal(float, float, float)
    specular_color_changed = pyqtSignal(float, float, float)
    lighting_changed = pyqtSignal(str, float)
    light_direction_changed = pyqtSignal(float, float, float)
    follow_camera_changed = pyqtSignal(bool)
    clipping_changed = pyqtSignal(str, float, float)
    clipping_enabled_changed = pyqtSignal(bool)

    def __init__(self):        
        self.rendering_panel = None
        self.clipping_panel = None
        self.global_tf = None
        self.volume_data = None
        super().__init__("Transfer Function", collapsible=False)

        
    def setup_content(self):
        """내용 설정"""
        self.content_layout.setSpacing(5)
        header_layout = QHBoxLayout()
        self.title_label = QLabel("🎨 Transfer Function")
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 10px 5px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch() # 오른쪽 정렬 효과
        self.content_layout.addLayout(header_layout)
        # TF 위젯
        self.tf_widget = TransferFunctionWidget()
        self.tf_widget.tf_changed.connect(self.on_tf_widget_changed)
        self.content_layout.addWidget(self.tf_widget)
        
        # TF 컨트롤 버튼들
        tf_control_buttons = [
            {"text": "Reset", "callback": self.reset_tf, "height": 30},
            {"text": "Save", "callback": self.save_tf, "height": 30},
            {"text": "Load", "callback": self.load_tf, "height": 30}
        ]
        tf_control_layout, _ = self.create_button_horizontal(tf_control_buttons)
        self.content_layout.addLayout(tf_control_layout)
        
        # 배경색 컨트롤
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

        # Shading 컨트롤
        self._setup_shading_controls()
        
        # Clipping 컨트롤
        self._setup_clipping_controls()
        
        # 하단 여백
        self.content_layout.addStretch(1)

    def _setup_shading_controls(self):
        """Shading 컨트롤 설정"""
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
        
        # Shading 헤더 버튼
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
        
        # Shading 내용 컨테이너
        self.shade_content_widget = QWidget()
        self.shade_content_widget.setVisible(False)
        self.shade_content_widget.setStyleSheet("QWidget { border: none; background-color: transparent; }")
        
        shade_content_layout = QVBoxLayout(self.shade_content_widget)
        shade_content_layout.setContentsMargins(10, 10, 10, 10)
        shade_content_layout.setSpacing(8)
        
        # Enable Shading 체크박스
        self.shade_toggle = QCheckBox("Enable Shading")
        self.shade_toggle.setChecked(False)
        self.shade_toggle.stateChanged.connect(self.on_shading_changed)
        self.shade_toggle.setStyleSheet("""
            QCheckBox { spacing: 5px; color: white; }
            QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #777; background-color: #353535; border-radius: 3px; }
            QCheckBox::indicator:checked { background-color: #0078d7; }
        """)
        shade_content_layout.addWidget(self.shade_toggle)
        
        # Lighting 위젯
        lighting_widget = QWidget()
        lighting_main_layout = QHBoxLayout(lighting_widget)
        lighting_main_layout.setContentsMargins(20, 0, 0, 0)
        lighting_main_layout.setSpacing(15)

        # Light Sphere
        self.light_sphere = LightSphereWidget()
        self.light_sphere.setEnabled(False)
        self.light_sphere.setFixedSize(80, 80)
        self.light_sphere.light_changed.connect(self.on_light_direction_changed)
        lighting_main_layout.addWidget(self.light_sphere)

        # 슬라이더들
        sliders_widget = QWidget()
        sliders_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        sliders_layout = QVBoxLayout(sliders_widget)
        sliders_layout.setContentsMargins(0, 0, 0, 0)
        sliders_layout.setSpacing(5)

        # Ambient
        self.ambient_slider, self.ambient_label, self.ambient_color_btn = self._create_lighting_row(
            sliders_layout, "Ambient:", 40, self.on_ambient_changed, self.on_ambient_color_clicked
        )

        # Diffuse
        self.diffuse_slider, self.diffuse_label, self.diffuse_color_btn = self._create_lighting_row(
            sliders_layout, "Diffuse:", 60, self.on_diffuse_changed, self.on_diffuse_color_clicked, max_val=500
        )

        # Specular
        self.specular_slider, self.specular_label, self.specular_color_btn = self._create_lighting_row(
            sliders_layout, "Specular:", 20, self.on_specular_changed, self.on_specular_color_clicked
        )

        # Follow Camera 체크박스
        self.follow_camera_checkbox = QCheckBox("Follow Camera")
        self.follow_camera_checkbox.setChecked(False)
        self.follow_camera_checkbox.setEnabled(False)
        self.follow_camera_checkbox.toggled.connect(self.on_follow_camera_changed)
        self.follow_camera_checkbox.setStyleSheet(self.shade_toggle.styleSheet())
        sliders_layout.addWidget(self.follow_camera_checkbox)

        lighting_main_layout.addWidget(sliders_widget, 1)
        shade_content_layout.addWidget(lighting_widget)
        
        shading_layout.addWidget(self.shade_content_widget)
        self.content_layout.addWidget(shading_widget)

    def _create_lighting_row(self, parent_layout, label_text, default_val, slider_callback, color_callback, max_val=100):
        """조명 슬라이더 행 생성"""
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

    def _setup_clipping_controls(self):
        """Clipping 컨트롤 설정"""
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
        
        # Clipping 헤더 버튼
        self.clip_header_btn = QPushButton("▶ Clipping Controls")
        self.clip_header_btn.setCheckable(True)
        self.clip_header_btn.setChecked(False)
        self.clip_header_btn.clicked.connect(self.toggle_clipping_section)
        self.clip_header_btn.setStyleSheet(self.shade_header_btn.styleSheet())
        clipping_layout.addWidget(self.clip_header_btn)
        
        # Clipping 내용 컨테이너
        self.clip_content_widget = QWidget()
        self.clip_content_widget.setVisible(False)
        self.clip_content_widget.setStyleSheet("QWidget { border: none; background-color: transparent; }")
        
        clip_content_layout = QVBoxLayout(self.clip_content_widget)
        clip_content_layout.setContentsMargins(10, 10, 10, 10)
        clip_content_layout.setSpacing(10)
        
        # ClippingPanel 인스턴스 생성
        self.clipping_panel = ClippingPanel()
        self.clipping_panel.setTitle("")
        self.clipping_panel.setStyleSheet("QGroupBox {border: none; margin-top: 0; padding-top: 0;}")
        self.clipping_panel.clipping_changed.connect(self.clipping_changed.emit)
        self.clipping_panel.clipping_enabled_changed.connect(self.clipping_enabled_changed.emit)
        
        clip_content_layout.addWidget(self.clipping_panel)
        clipping_layout.addWidget(self.clip_content_widget)
        self.content_layout.addWidget(clipping_widget)

    def toggle_shading_section(self):
        """Shading 섹션 접기/펼치기"""
        is_expanded = self.shade_header_btn.isChecked()
        
        if is_expanded and self.clip_header_btn.isChecked():
            self.clip_header_btn.setChecked(False)
            self.clip_content_widget.setVisible(False)
            self.clip_header_btn.setText("▶ Clipping Controls")
            
        arrow = "▼" if is_expanded else "▶"
        self.shade_header_btn.setText(f"{arrow} Shading Controls")
        self.shade_content_widget.setVisible(is_expanded)

    def toggle_clipping_section(self):
        """Clipping 섹션 접기/펼치기"""
        is_expanded = self.clip_header_btn.isChecked()
        
        if is_expanded and self.shade_header_btn.isChecked():
            self.shade_header_btn.setChecked(False)
            self.shade_content_widget.setVisible(False)
            self.shade_header_btn.setText("▶ Shading Controls")

        arrow = "▼" if is_expanded else "▶"
        self.clip_header_btn.setText(f"{arrow} Clipping Controls")
        self.clip_content_widget.setVisible(is_expanded)

    def set_volume_data(self, volume_data):
        """볼륨 데이터 설정"""
        self.volume_data = volume_data
        if volume_data is not None:
            self.tf_widget.set_volume_data(volume_data)
            if self.clipping_panel and hasattr(volume_data, 'shape'):
                self.clipping_panel.set_volume_shape(volume_data.shape)

    def on_tf_widget_changed(self):
        """TF 위젯 변경 처리"""
        self.global_tf = self.tf_widget.get_nodes()
        self.tf_changed.emit(self.global_tf)

    def reset_tf(self):
        """TF 초기화"""
        self.tf_widget.reset_to_default()
        self.global_tf = self.tf_widget.get_nodes()
        self.tf_changed.emit(self.global_tf)

    def save_tf(self):
        """TF 저장"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Transfer Function", "./resources/TFs/TF_global.json", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
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
        """TF 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Transfer Function", "./resources/TFs", 
            "JSON Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    import_data = json.load(f)
                
                # Shading 설정 로드
                if 'shading' in import_data:
                    shading_data = import_data['shading']
                    shading_enabled = shading_data.get('enabled', False)
                    
                    self.shade_toggle.blockSignals(True)
                    self.shade_toggle.setChecked(shading_enabled)
                    self.shade_toggle.blockSignals(False)
                    
                    self._set_lighting_sliders_enabled(shading_enabled)
                    
                    self.ambient_slider.setValue(int(shading_data.get('ambient', 0.4) * 100))
                    self.diffuse_slider.setValue(int(shading_data.get('diffuse', 0.6) * 100))
                    self.specular_slider.setValue(int(shading_data.get('specular', 0.2) * 100))
                    
                    self.shading_changed.emit(shading_enabled)
                
                # TF 로드
                if 'global' in import_data:
                    self.tf_widget.set_transfer_function_from_array(import_data['global'])
                    self.global_tf = import_data['global']
                    self.tf_changed.emit(self.global_tf)
                
                self.emit_status(f"Transfer function loaded: {file_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load: {str(e)}")

    def select_background_color(self):
        """배경색 선택"""
        color = QColorDialog.getColor(QColor(255, 255, 255), self, "배경색 선택")
        if color.isValid():
            bg_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
            self.bg_color_btn.setStyleSheet(f"background-color: {color.name()};")
            self.background_color_changed.emit(bg_color, bg_color)
    
    def reset_background_color(self):
        """배경색 리셋"""
        bg = (1.0, 1.0, 1.0)
        self.bg_color_btn.setStyleSheet("background-color: #FFFFFF;")
        self.background_color_changed.emit(bg, bg)

    def on_shading_changed(self, state):
        """쉐이딩 변경 처리"""
        enabled = bool(state)
        self._set_lighting_sliders_enabled(enabled)
        self.shading_changed.emit(enabled)
    
    def _set_lighting_sliders_enabled(self, enabled):
        """조명 슬라이더들 활성화/비활성화"""
        self.ambient_slider.setEnabled(enabled)
        self.diffuse_slider.setEnabled(enabled)
        self.specular_slider.setEnabled(enabled)
        self.light_sphere.setEnabled(enabled)
        self.follow_camera_checkbox.setEnabled(enabled)
        self.ambient_color_btn.setEnabled(enabled)
        self.diffuse_color_btn.setEnabled(enabled)
        self.specular_color_btn.setEnabled(enabled)
    
    def on_ambient_changed(self, value):
        ambient = value / 100.0
        self.ambient_label.setText(f"{ambient:.1f}")
        self.lighting_changed.emit("ambient", ambient)
    
    def on_diffuse_changed(self, value):
        diffuse = value / 100.0
        self.diffuse_label.setText(f"{diffuse:.1f}")
        self.lighting_changed.emit("diffuse", diffuse)
    
    def on_specular_changed(self, value):
        specular = value / 100.0
        self.specular_label.setText(f"{specular:.1f}")
        self.lighting_changed.emit("specular", specular)
    
    def on_light_direction_changed(self, x, y, z):
        self.light_direction_changed.emit(x, y, z)

    def on_follow_camera_changed(self, checked):
        self.follow_camera_changed.emit(checked)

    def on_ambient_color_clicked(self):
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, "Select Ambient Color")
        if color.isValid():
            self.ambient_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid gray;")
            self.ambient_color_changed.emit(color.redF(), color.greenF(), color.blueF())

    def on_diffuse_color_clicked(self):
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, "Select Diffuse Color")
        if color.isValid():
            self.diffuse_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid gray;")
            self.diffuse_color_changed.emit(color.redF(), color.greenF(), color.blueF())

    def on_specular_color_clicked(self):
        color = QColorDialog.getColor(Qt.GlobalColor.white, self, "Select Specular Color")
        if color.isValid():
            self.specular_color_btn.setStyleSheet(f"background-color: {color.name()}; border: 1px solid gray;")
            self.specular_color_changed.emit(color.redF(), color.greenF(), color.blueF())

    def reset_clipping_safe(self):
        """클리핑 패널 안전하게 리셋"""
        if self.clipping_panel is not None:
            self.clipping_panel.reset_clipping()

    def get_clipping_ranges(self):
        """ClippingPanel에서 현재 클리핑 범위 가져오기"""
        if self.clipping_panel:
            return self.clipping_panel.get_clipping_ranges()
        return None
    
    # optimization된 새로운 노드를 적용
    def apply_external_nodes(self, new_nodes):
        """외부(OptimizationPanel)에서 최적화된 TF 노드를 적용하는 함수"""
        if new_nodes is None:
            return

        print("🔄 Applying Optimized TF Nodes...")
        
        # 1. TF 위젯의 UI를 업데이트 (그래프 갱신)
        # (TransferFunctionWidget에 set_transfer_function_from_array 메서드가 있다고 가정)
        self.tf_widget.set_transfer_function_from_array(new_nodes)
        
        # 2. 내부 변수 업데이트
        self.global_tf = new_nodes
        
        # 3. ★중요★ 변경 사실을 렌더링 패널에 알림
        self.tf_changed.emit(self.global_tf)