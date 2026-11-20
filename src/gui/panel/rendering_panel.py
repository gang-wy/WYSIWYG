"""중앙 3D 렌더링 패널: VTK 렌더러 + 카메라/샘플링/저장 컨트롤. 렌더러 호출을 감싸는 창구 역할."""

import json
import os
from datetime import datetime

from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QSlider, QRadioButton,
                             QFrame, QMessageBox, QSizePolicy, QFileDialog)
from PyQt6.QtCore import Qt
from src.gui.panel.base_panel import BasePanel


class RenderingPanel(BasePanel):
    """VTKVolumeRenderer를 소유하고, 패널 메서드는 렌더러가 없으면 아무 것도 하지 않는다."""

    def __init__(self):
        try:
            from src.gui.widget.renderer_widget import VTKVolumeRenderer
            self.vtk_renderer = VTKVolumeRenderer()
        except Exception as e:
            print(f"❌ VTK 렌더러 생성 실패: {e}")
            self.vtk_renderer = None
        super().__init__("3D Volume Rendering", collapsible=False)

    def cleanup(self):
        if self.vtk_renderer:
            self.vtk_renderer.cleanup()

    def setup_content(self):
        header_layout = QHBoxLayout()
        self.title_label = QLabel("🎮 3D Volume Rendering")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.content_layout.addLayout(header_layout)

        if self.vtk_renderer:
            self.vtk_renderer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.content_layout.addWidget(self.vtk_renderer, stretch=1)

        # 카메라
        camera_group, camera_layout = self.create_group_box("", "horizontal")
        camera_layout.setSpacing(8)
        camera_layout.setContentsMargins(10, 5, 10, 5)
        btn_style = "QPushButton { height: 32px; padding: 0 5px; font-weight: bold; }"

        for text, width, callback in (("Zoom In", 80, self.zoom_in),
                                      ("Zoom Out", 80, self.zoom_out),
                                      ("Reset", 60, self.reset_view)):
            btn = QPushButton(text)
            btn.setFixedWidth(width)
            btn.setStyleSheet(btn_style)
            btn.clicked.connect(callback)
            camera_layout.addWidget(btn)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setStyleSheet("background-color: #444;")
        camera_layout.addWidget(line)

        camera_layout.addWidget(QLabel("🔍 Zoom:"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 150)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self.on_zoom_slider_changed)
        camera_layout.addWidget(self.zoom_slider, stretch=1)

        self.zoom_label = QLabel("1.0x")
        self.zoom_label.setFixedWidth(40)
        camera_layout.addWidget(self.zoom_label)
        self.content_layout.addWidget(camera_group)

        # 샘플링 레이트: rate = slider / 100, sample distance = 1 / rate
        # (초기값 200 → 0.5, renderer_widget의 current_sample_distance 초기값과 맞춤)
        quality_group, quality_layout = self.create_group_box("", "horizontal")
        quality_layout.setContentsMargins(10, 5, 10, 5)
        quality_layout.addWidget(QLabel("🔬 Sampling Rate:"))
        self.sampling_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.sampling_rate_slider.setRange(1, 1000)
        self.sampling_rate_slider.setValue(200)
        self.sampling_rate_slider.valueChanged.connect(self.on_sampling_rate_changed)
        quality_layout.addWidget(self.sampling_rate_slider, stretch=1)
        self.sampling_rate_label = QLabel("2.00x")
        self.sampling_rate_label.setFixedWidth(45)
        quality_layout.addWidget(self.sampling_rate_label)
        self.content_layout.addWidget(quality_group)

        # 저장
        save_group, save_layout = self.create_group_box("", "horizontal")
        save_layout.setContentsMargins(10, 5, 10, 5)
        self.save_current_size_radio = QRadioButton("Current")
        self.save_current_size_radio.setChecked(True)
        save_layout.addWidget(self.save_current_size_radio)
        self.save_square_radio = QRadioButton("1:1")
        save_layout.addWidget(self.save_square_radio)

        save_btn_style = "QPushButton { height: 35px; padding: 0 10px; }"
        for text, callback in (("📸 Save Image", self.save_current_rendering),
                               ("💾 Save Cam", self.save_current_camera_to_file),
                               ("📥 Load Cam", self.load_camera_from_file)):
            btn = QPushButton(text)
            btn.setStyleSheet(save_btn_style)
            btn.clicked.connect(callback)
            save_layout.addWidget(btn)
        self.content_layout.addWidget(save_group)

    # ------------------------------------------------------------------
    # SAM 포인트 오버레이
    # ------------------------------------------------------------------
    def add_point_2d(self, vtk_x, vtk_y, p_type):
        if self.vtk_renderer:
            self.vtk_renderer.add_overlay_point(vtk_x, vtk_y, p_type)

    def set_overlay_visible(self, visible):
        if self.vtk_renderer:
            if not visible:
                self.vtk_renderer.clear_overlay_points()
            self.vtk_renderer.set_overlay_visible(visible)
            self.vtk_renderer.vtk_widget.GetRenderWindow().Render()

    def clear_overlay(self):
        if self.vtk_renderer:
            self.vtk_renderer.clear_overlay_points()
            self.vtk_renderer.vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    # 볼륨 / TF / 조명
    # ------------------------------------------------------------------
    def set_volume_data(self, volume_data):
        if self.vtk_renderer:
            self.vtk_renderer.set_volume_data(volume_data)

    def update_transfer_function(self, tf_array):
        if self.vtk_renderer:
            self.vtk_renderer.update_transfer_function_optimized(tf_array)

    def set_local_probability_maps(self, prob_maps):
        if self.vtk_renderer:
            return self.vtk_renderer.set_local_probability_maps(prob_maps)
        return False

    def clear_local_probability_maps(self):
        if self.vtk_renderer:
            self.vtk_renderer.clear_local_probability_maps()

    def update_local_transfer_function(self, tf_nodes, label=1, render=True):
        if self.vtk_renderer:
            self.vtk_renderer.update_label_transfer_function(tf_nodes, label=label, render=render)

    def render(self):
        if self.vtk_renderer:
            self.vtk_renderer.render()

    def set_background_color(self, color1, color2=None):
        if self.vtk_renderer:
            self.vtk_renderer.set_background_color(color1, color2)

    def set_shading(self, enabled):
        if self.vtk_renderer:
            self.vtk_renderer.set_shading(enabled)

    def set_lighting_property(self, property_type, value):
        """property_type: 'ambient' | 'diffuse' | 'specular'"""
        if not self.vtk_renderer:
            return
        setter = {"ambient": self.vtk_renderer.set_ambient,
                  "diffuse": self.vtk_renderer.set_diffuse,
                  "specular": self.vtk_renderer.set_specular}.get(property_type)
        if setter:
            setter(value)

    def show_cluster_points(self, world_points, cluster_labels, visible_clusters, best_cluster=None):
        """visible_clusters에 든 cluster만 라벨 점 구름 오버레이로 표시."""
        if self.vtk_renderer:
            self.vtk_renderer.show_cluster_points(world_points, cluster_labels, visible_clusters, best_cluster)

    def clear_roi_actor(self):
        if self.vtk_renderer:
            self.vtk_renderer.clear_roi_actor()

    # ------------------------------------------------------------------
    # 카메라
    # ------------------------------------------------------------------
    def zoom_in(self):
        self.zoom_slider.setValue(min(150, self.zoom_slider.value() + 20))

    def zoom_out(self):
        self.zoom_slider.setValue(max(10, self.zoom_slider.value() - 20))

    def reset_view(self):
        if self.vtk_renderer:
            self.vtk_renderer.reset_camera_manual()
            self.zoom_slider.setValue(100)
            self.zoom_label.setText("1.0x")

    def on_zoom_slider_changed(self, value):
        if self.vtk_renderer:
            zoom_factor = value / 100.0
            if self.vtk_renderer.set_zoom_factor(zoom_factor):
                self.zoom_label.setText(f"{zoom_factor:.1f}x")

    def on_sampling_rate_changed(self, value):
        rate = value / 100.0
        self.sampling_rate_label.setText(f"{rate:.2f}x")
        if self.vtk_renderer:
            self.vtk_renderer.set_ray_sampling_rate(rate)

    def get_camera_info(self):
        """Returns: {position, focal_point, view_up, view_angle, distance} 또는 None"""
        if self.vtk_renderer and self.vtk_renderer.renderer:
            camera = self.vtk_renderer.renderer.GetActiveCamera()
            if camera:
                return {
                    'position': camera.GetPosition(),
                    'focal_point': camera.GetFocalPoint(),
                    'view_up': camera.GetViewUp(),
                    'view_angle': camera.GetViewAngle(),
                    'distance': camera.GetDistance()
                }
        return None

    def set_camera_info(self, camera_info):
        """카메라 상태를 적용하고 다시 렌더링한다. Returns: 성공 여부"""
        if not (self.vtk_renderer and self.vtk_renderer.renderer and camera_info):
            return False
        camera = self.vtk_renderer.renderer.GetActiveCamera()
        if not camera:
            return False
        try:
            camera.SetPosition(camera_info['position'])
            camera.SetFocalPoint(camera_info['focal_point'])
            camera.SetViewUp(camera_info['view_up'])
            camera.SetViewAngle(camera_info['view_angle'])
            # 이전 볼륨의 near/far 범위가 남지 않도록 현재 볼륨 기준으로 다시 맞춘다
            self.vtk_renderer.renderer.ResetCameraClippingRange()
            self.vtk_renderer.vtk_widget.GetRenderWindow().Render()
            return True
        except Exception as e:
            print(f"❌ 카메라 설정 실패: {e}")
            return False

    # ------------------------------------------------------------------
    # 파일 저장 / 로드
    # ------------------------------------------------------------------
    def save_current_rendering(self, return_filename=False):
        """현재 화면을 SAM 포인트 오버레이 없이 PNG로 저장한다.

        Returns: return_filename=True면 저장 경로(실패 시 None)
        """
        if not self.vtk_renderer:
            QMessageBox.critical(self, "Error", "VTK 렌더러가 초기화되지 않았습니다.")
            return None
        self.set_overlay_visible(False)
        use_square = self.save_square_radio.isChecked()
        filename = self.vtk_renderer.save_current_rendering(use_square_ratio=use_square)
        self.set_overlay_visible(True)
        if return_filename:
            return filename

        if filename:
            ratio_info = "1:1 ratio" if use_square else "current size"
            self.emit_status(f"Rendering saved ({ratio_info}): {os.path.basename(filename)}")
            QMessageBox.information(self, "Rendering Saved", f"Rendering saved:\n{filename}")
        else:
            QMessageBox.critical(self, "Error", "Failed to save rendering")
        return None

    def save_rendering_to(self, filename):
        """현재 화면(창 크기 그대로, SAM 포인트 오버레이 없이)을 filename에 PNG로 저장한다. Returns: 경로 또는 None"""
        if not self.vtk_renderer:
            return None
        self.set_overlay_visible(False)
        try:
            return self.vtk_renderer.save_current_rendering(use_square_ratio=False, filename=filename)
        finally:
            self.set_overlay_visible(True)

    def apply_clipping(self, axis, min_val, max_val):
        """axis('x'|'y'|'z')의 정규화 범위 [min_val, max_val]를 볼륨 bounds 좌표로 바꿔 클리핑한다."""
        if not self.vtk_renderer:
            return
        volume = self.vtk_renderer.clipping_manager.get_current_volume()
        if not (volume and volume.GetMapper() and volume.GetMapper().GetInput()):
            return
        axis_index = {'x': 0, 'y': 1, 'z': 2}.get(axis)
        if axis_index is None:
            return
        bounds = volume.GetMapper().GetInput().GetBounds()
        axis_min, axis_max = bounds[2 * axis_index], bounds[2 * axis_index + 1]
        min_pos = axis_min + min_val * (axis_max - axis_min)
        max_pos = axis_min + max_val * (axis_max - axis_min)
        self.vtk_renderer.set_clipping_range(axis_index, min_pos, max_pos)

    def set_clipping_enabled(self, enabled):
        if not self.vtk_renderer:
            return
        self.vtk_renderer.enable_clipping(enabled)
        if not enabled:
            self.vtk_renderer.reset_clipping()
        else:
            for axis in ('x', 'y', 'z'):
                self.apply_clipping(axis, 0.0, 1.0)

    def save_current_camera_to_file(self):
        """카메라 상태와 zoom을 JSON으로 저장한다(./resources/Camera 기본 경로)."""
        camera_info = self.get_camera_info()
        if not camera_info:
            QMessageBox.warning(self, "Warning", "카메라 정보를 가져올 수 없습니다.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Camera State",
            f"./resources/Camera/camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            "JSON Files (*.json);;All Files (*)")
        if not file_path:
            return
        if not file_path.lower().endswith('.json'):
            file_path += '.json'

        try:
            camera_data = {
                'camera_state': {
                    'position': list(camera_info['position']),
                    'focal_point': list(camera_info['focal_point']),
                    'view_up': list(camera_info['view_up']),
                    'view_angle': camera_info['view_angle'],
                    'distance': camera_info.get('distance', 0)
                },
                'zoom_factor': self.zoom_slider.value() / 100.0,
                'timestamp': datetime.now().isoformat(),
                'version': '3.0'
            }
            with open(file_path, 'w') as f:
                json.dump(camera_data, f, indent=2)
            self.emit_status(f"Camera state saved: {file_path}")
            QMessageBox.information(self, "Save Complete", f"Camera state saved!\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save camera state:\n{str(e)}")

    def load_camera_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Camera State", "./resources/Camera", "JSON Files (*.json);;All Files (*)")
        if file_path:
            self.load_camera_from_path(file_path)

    def load_camera_from_path(self, file_path):
        """카메라 JSON(camera_state, zoom_factor)을 읽어 적용한다."""
        try:
            with open(file_path, 'r') as f:
                camera_data = json.load(f)

            if 'camera_state' not in camera_data:
                QMessageBox.warning(self, "Warning", "Invalid camera state file format")
                return

            camera_state = camera_data['camera_state']
            camera_info = {
                'position': tuple(camera_state['position']),
                'focal_point': tuple(camera_state['focal_point']),
                'view_up': tuple(camera_state['view_up']),
                'view_angle': camera_state['view_angle'],
                'distance': camera_state.get('distance', 0)
            }
            if self.set_camera_info(camera_info):
                if 'zoom_factor' in camera_data:
                    self.zoom_slider.setValue(int(camera_data['zoom_factor'] * 100))
                self.emit_status("Camera state loaded")
            else:
                QMessageBox.warning(self, "Warning", "Failed to apply camera state")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load camera state:\n{str(e)}")
