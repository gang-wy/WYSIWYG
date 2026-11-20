"""VTK GPU 볼륨 렌더러 위젯: global TF 렌더링, custom shader 기반 local TF(multi-class),
SAM 포인트 2D 오버레이, cluster 점 구름 오버레이, 피킹, 스크린샷."""

import os
import traceback
from datetime import datetime

import numpy as np
import vtk
from vtkmodules.util import numpy_support
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from PyQt6.QtCore import pyqtSignal

from src.gui.rendering.clipping_manager import VolumeClippingManager
from src.gui.rendering.camera_controller import CameraController
from src.gui.rendering.lighting_manager import LightingManager
from src.gui.rendering.shader_manager import ShaderManager
from src.core.wysiwyg_tf_editor import WysiwygTFEditor

SCREENSHOT_DIR = "./resources/Rendered_Image"
DEFAULT_TF = [[0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 1.0, 0.5, 0.0, 0.1],
              [0.7, 1.0, 1.0, 1.0, 0.8], [1.0, 1.0, 1.0, 1.0, 1.0]]
BEST_CLUSTER_COLOR = np.array([255, 30, 30], dtype=np.uint8)
OTHER_CLUSTER_COLORS = np.array([   # 빨강 계열을 빼고, k가 작을 때 쓰는 앞쪽일수록 서로 멀게
    (0, 110, 255), (255, 225, 0), (40, 200, 60), (170, 80, 255), (0, 225, 225),
    (255, 135, 220), (170, 170, 170), (140, 90, 40), (0, 125, 125),
], dtype=np.uint8)


def _palette_slot(labels, best_cluster):
    """best를 건너뛴 순서로 센 OTHER_CLUSTER_COLORS 자리."""
    rank = labels if best_cluster is None else labels - (labels > best_cluster)
    return rank % len(OTHER_CLUSTER_COLORS)


def cluster_colors(labels, best_cluster=None):
    """라벨 (N,) → RGB 0~255 (N,3). best는 고정색, 나머지는 best를 건너뛴 순서로 팔레트를 돈다."""
    labels = np.asarray(labels, dtype=np.int64)
    colors = OTHER_CLUSTER_COLORS[_palette_slot(labels, best_cluster)]
    if best_cluster is not None:
        colors[labels == best_cluster] = BEST_CLUSTER_COLOR
    return colors


def set_cluster_color(cluster, best_cluster, rgb):
    """cluster의 색을 바꾼다. cluster 번호가 아니라 자리에 저장되어 다음 Run Clustering에도 유지된다."""
    if cluster == best_cluster:
        BEST_CLUSTER_COLOR[:] = rgb
    else:
        OTHER_CLUSTER_COLORS[_palette_slot(cluster, best_cluster)] = rgb


class VTKPointOverlay:
    """VTK display 좌표(좌하단 원점)에 그리는 SAM 포인트 원 마커(positive=파랑, negative=빨강)."""

    def __init__(self, renderer):
        self.renderer = renderer
        self.actors = []       # [(fill_actor, outline_actor), ...]
        self.stroke_actor = None
        self.visible = False

    def add_point(self, x, y, point_type):
        color = (0.13, 0.59, 0.95) if point_type == "positive" else (0.96, 0.26, 0.21)

        circle_source = vtk.vtkRegularPolygonSource()
        circle_source.SetNumberOfSides(20)
        circle_source.SetRadius(8)
        circle_source.SetCenter(0, 0, 0)
        circle_source.Update()
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(circle_source.GetOutputPort())
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.9)
        actor.SetPosition(x, y)

        outline_source = vtk.vtkRegularPolygonSource()
        outline_source.SetNumberOfSides(20)
        outline_source.SetRadius(9)
        outline_source.SetCenter(0, 0, 0)
        outline_source.GeneratePolygonOff()
        outline_source.Update()
        outline_mapper = vtk.vtkPolyDataMapper2D()
        outline_mapper.SetInputConnection(outline_source.GetOutputPort())
        outline_actor = vtk.vtkActor2D()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)
        outline_actor.GetProperty().SetLineWidth(2)
        outline_actor.SetPosition(x, y)

        if self.visible:
            self.renderer.AddActor2D(outline_actor)
            self.renderer.AddActor2D(actor)
        self.actors.append((actor, outline_actor))

    def set_stroke(self, points):
        """스트로크 선(VTK display 좌표 목록)을 그린다. 점이 2개 미만이면 선을 지운다."""
        if self.stroke_actor is not None:
            self.renderer.RemoveActor2D(self.stroke_actor)
            self.stroke_actor = None
        if len(points) < 2:
            return
        vtk_pts = vtk.vtkPoints()
        for x, y in points:
            vtk_pts.InsertNextPoint(x, y, 0)
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(len(points))
        for i in range(len(points)):
            line.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_pts)
        poly.SetLines(cells)
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputData(poly)
        self.stroke_actor = vtk.vtkActor2D()
        self.stroke_actor.SetMapper(mapper)
        self.stroke_actor.GetProperty().SetColor(1.0, 0.85, 0.1)
        self.stroke_actor.GetProperty().SetLineWidth(3)
        if self.visible:
            self.renderer.AddActor2D(self.stroke_actor)

    def clear_points(self):
        for actor, outline_actor in self.actors:
            self.renderer.RemoveActor2D(actor)
            self.renderer.RemoveActor2D(outline_actor)
        self.actors = []
        self.set_stroke([])

    def set_visible(self, visible):
        self.visible = visible
        actors = [a for pair in self.actors for a in reversed(pair)]
        if self.stroke_actor is not None:
            actors.insert(0, self.stroke_actor)
        for actor in actors:
            if visible:
                self.renderer.AddActor2D(actor)
            else:
                self.renderer.RemoveActor2D(actor)


class VTKVolumeRenderer(QWidget):
    """standard_volume(vtkGPUVolumeRayCastMapper) 하나를 그린다.

    local class가 없으면 VTK 기본 TF 경로, 있으면 vtkShaderProperty의 custom shader가 볼륨 전체를 그린다.
    """

    point_2d_picked = pyqtSignal(int, int)  # VTK display 좌표 (좌하단 원점)
    stroke_completed = pyqtSignal(list)     # [(x, y), ...] VTK display 좌표 — Line 모드에서 드래그를 놓을 때

    def __init__(self):
        super().__init__()
        vtk.vtkObject.GlobalWarningDisplayOff()

        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)

        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        self.picking_enabled = False
        self.current_pick_type = "positive"      # "line"이면 드래그로 스트로크를 그린다
        self.picking_observer_tag = None
        self.stroke_observer_tags = []
        self.current_stroke = None               # 그리는 중인 스트로크 [(x, y), ...]
        self.point_overlay = VTKPointOverlay(self.renderer)
        self.roi_actor = None

        self.volume_data = None
        self.voxel_spacing = (1.0, 1.0, 1.0)
        self.tf_editor = WysiwygTFEditor(lut_size=256)
        self.global_tf_nodes = None

        self.standard_volume = None
        self.standard_property = None
        self.standard_mapper = None
        self.current_sample_distance = 0.5   # rendering_panel 샘플링 슬라이더 초기값(2.00x)과 일치

        # custom shader(local TF) 상태
        self.shader_manager = ShaderManager(self)
        self.shader_property = None
        self.local_shader_active = False
        self.label_to_class = {}             # {label_id: shader class 인덱스(1..N)}
        self.shader_texture_state = None
        self.shader_observer_tag = None
        self.shader_params = {
            'ambient': 0.3, 'diffuse': 0.7, 'specular': 0.5, 'shininess': 32.0,
            'light_x': -1.0, 'light_y': 0.5, 'light_z': 0.5,
            'shading_enabled': False, 'follow_camera': False,
            'ambient_color': [1.0, 1.0, 1.0], 'diffuse_color': [1.0, 1.0, 1.0],
            'specular_color': [1.0, 1.0, 1.0],
        }

        self.clipping_manager = VolumeClippingManager(self)
        self.camera_controller = CameraController(self)
        self.lighting_manager = LightingManager(self)

        self.setup_renderer()
        self.setup_interactor()

    def cleanup(self):
        try:
            self.point_overlay.clear_points()
            camera = self.renderer.GetActiveCamera()
            if camera:
                camera.RemoveAllObservers()
            self.volume_data = None
        except Exception:
            pass

    def render(self):
        self.vtk_widget.GetRenderWindow().Render()

    def setup_renderer(self):
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.renderer.SetBackground2(0.8, 0.8, 0.8)
        self.renderer.SetGradientBackground(True)
        self.lighting_manager.setup_lighting()
        self.vtk_widget.GetRenderWindow().SetMultiSamples(8)

    def setup_interactor(self):
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.SetMotionFactor(2.0)
        self.interactor.SetInteractorStyle(style)
        self.setup_picking_observer()

    # ------------------------------------------------------------------
    # SAM 포인트 오버레이 / 피킹
    # ------------------------------------------------------------------
    def add_overlay_point(self, vtk_x, vtk_y, point_type):
        self.point_overlay.add_point(vtk_x, vtk_y, point_type)
        self.render()

    def clear_overlay_points(self):
        self.point_overlay.clear_points()
        self.render()

    def set_overlay_visible(self, visible):
        self.point_overlay.set_visible(visible)
        self.render()

    def set_interaction_enabled(self, enabled):
        """True면 trackball 카메라 조작, False면 카메라 고정(피킹 전용)."""
        style = vtk.vtkInteractorStyleTrackballCamera() if enabled else vtkInteractorStyleUser()
        self.interactor.SetInteractorStyle(style)
        self.setup_picking_observer()

    def set_picking_enabled(self, enabled):
        self.picking_enabled = enabled
        self.point_overlay.set_visible(enabled)
        self.render()

    def setup_picking_observer(self):
        if self.picking_observer_tag is not None:
            self.interactor.RemoveObserver(self.picking_observer_tag)
        for tag in self.stroke_observer_tags:
            self.interactor.RemoveObserver(tag)
        self.picking_observer_tag = self.interactor.AddObserver(
            "LeftButtonPressEvent", self.on_left_button_press, 10.0)
        self.stroke_observer_tags = [
            self.interactor.AddObserver("MouseMoveEvent", self.on_mouse_move, 10.0),
            self.interactor.AddObserver("LeftButtonReleaseEvent", self.on_left_button_release, 10.0)]

    def on_left_button_press(self, obj, event):
        """피킹 모드일 때: Line이면 스트로크 시작, 아니면 클릭 위치를 point_2d_picked로 보낸다."""
        if not self.picking_enabled:
            return
        x, y = self.interactor.GetEventPosition()
        if self.current_pick_type == "line":
            self.current_stroke = [(x, y)]
            self.point_overlay.clear_points()
            self.render()
        else:
            self.point_2d_picked.emit(x, y)

    def on_mouse_move(self, obj, event):
        if self.current_stroke is None:
            return
        x, y = self.interactor.GetEventPosition()
        if (x, y) != self.current_stroke[-1]:
            self.current_stroke.append((x, y))
            self.point_overlay.set_stroke(self.current_stroke)
            self.render()

    def on_left_button_release(self, obj, event):
        if self.current_stroke is None:
            return
        stroke, self.current_stroke = self.current_stroke, None
        self.stroke_completed.emit(stroke)

    # ------------------------------------------------------------------
    # 볼륨 / global TF
    # ------------------------------------------------------------------
    def set_volume_data(self, volume_data):
        """볼륨을 교체한다. 첫 로드면 카메라를 리셋하고, 아니면 이전 카메라를 유지한다."""
        if volume_data is None:
            return
        is_first_load = self.volume_data is None
        if not is_first_load:
            self.camera_controller.save_camera_state()

        self.volume_data = volume_data
        self.setup_standard_volume(volume_data)
        self.clipping_manager.reset_clipping()
        self.clipping_manager.update_clipping_target()

        if is_first_load:
            self.camera_controller.reset_camera_manual()
        elif not self.camera_controller.restore_camera_state():
            self.camera_controller.setup_camera(force_reset=True)
        self.render()

    def setup_standard_volume(self, volume_data):
        """GPU ray cast mapper + volume property(global TF) + 빈 shader property로 볼륨 actor를 만든다."""
        try:
            self.clear_all_volumes()
            vtk_data = self.numpy_to_vtk_imagedata(volume_data, self.voxel_spacing)

            self.standard_mapper = vtk.vtkGPUVolumeRayCastMapper()
            self.standard_mapper.SetBlendModeToComposite()
            self.standard_mapper.SetSampleDistance(self.current_sample_distance)
            self.standard_mapper.SetAutoAdjustSampleDistances(False)
            self.standard_mapper.SetInputData(vtk_data)

            self.standard_property = vtk.vtkVolumeProperty()
            self.standard_property.SetInterpolationTypeToLinear()
            tf_nodes = self.global_tf_nodes or DEFAULT_TF
            color_func, opacity_func = self.create_vtk_tf_from_array(tf_nodes)
            self.standard_property.SetColor(color_func)
            self.standard_property.SetScalarOpacity(opacity_func)

            self.standard_volume = vtk.vtkVolume()
            self.standard_volume.SetMapper(self.standard_mapper)
            self.standard_volume.SetProperty(self.standard_property)

            # local class가 등록되면 set_local_probability_maps가 여기에 GLSL을 얹는다
            self.shader_property = vtk.vtkShaderProperty()
            self.standard_volume.SetShaderProperty(self.shader_property)
            self.local_shader_active = False
            self.label_to_class = {}

            self.renderer.AddVolume(self.standard_volume)
            self.global_tf_nodes = tf_nodes

            self.clipping_manager.reset_clipping()
            if self.clipping_manager.clipping_enabled:
                self.clipping_manager.update_clipping_target()
        except Exception as e:
            print(f"표준 볼륨 설정 실패: {e}")
            traceback.print_exc()

    def clear_all_volumes(self):
        if self.standard_volume:
            self.renderer.RemoveVolume(self.standard_volume)
        self.standard_volume = None
        self.standard_mapper = None

    def numpy_to_vtk_imagedata(self, numpy_array, voxel_spacing):
        """(X, Y, Z) 배열 → float32 vtkImageData(origin 0, 주어진 spacing). 실패 시 None"""
        try:
            vtk_data = vtk.vtkImageData()
            vtk_data.SetDimensions(numpy_array.shape[:3])
            vtk_data.SetSpacing(voxel_spacing)
            vtk_data.SetOrigin(0.0, 0.0, 0.0)
            flat_data = numpy_array.ravel(order='F').astype(np.float32)
            vtk_array = numpy_support.numpy_to_vtk(flat_data, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName("scalars")
            vtk_array.SetNumberOfComponents(1)
            vtk_data.GetPointData().SetScalars(vtk_array)
            return vtk_data
        except Exception as e:
            print(f"VTK 데이터 변환 실패: {e}")
            return None

    def create_vtk_tf_from_array(self, tf_nodes, return_array=False):
        """TF node list를 256 표본으로 샘플링한다.

        Returns:
            return_array=False → (vtkColorTransferFunction, vtkPiecewiseFunction), node가 2개 미만이면 (None, None)
            return_array=True  → (color (256,256,3) uint8, opacity (256,256,1) uint8) — 모든 행이 동일
        """
        if not tf_nodes or len(tf_nodes) < 2:
            return None, None

        xs, color_lut, opacity_lut = self.tf_editor.sample_nodes_to_lut(tf_nodes, num_bins=256)

        if return_array:
            color_u8 = np.clip(color_lut * 255, 0, 255).astype(np.uint8)
            opacity_u8 = np.clip(opacity_lut * 255, 0, 255).astype(np.uint8)
            color_array = np.broadcast_to(color_u8[np.newaxis, :, :], (256, 256, 3)).copy()
            opacity_array = np.broadcast_to(opacity_u8[np.newaxis, :, np.newaxis], (256, 256, 1)).copy()
            return color_array, opacity_array

        color_func = vtk.vtkColorTransferFunction()
        opacity_func = vtk.vtkPiecewiseFunction()
        for i in range(256):
            color_func.AddRGBPoint(float(xs[i]), *color_lut[i].tolist())
            opacity_func.AddPoint(float(xs[i]), float(opacity_lut[i]))
        return color_func, opacity_func

    def update_transfer_function_optimized(self, tf_nodes):
        """global TF를 volume property와 (local shader가 켜져 있으면) class 0 텍스처에 반영한다."""
        try:
            self.global_tf_nodes = tf_nodes
            if self.standard_volume:
                color_func, opacity_func = self.create_vtk_tf_from_array(tf_nodes)
                volume_property = self.standard_volume.GetProperty()
                volume_property.SetColor(color_func)
                volume_property.SetScalarOpacity(opacity_func)
            if self.local_shader_active and self.shader_texture_state is not None:
                self.shader_manager.update_class_color_opacity_texture(
                    self.vtk_widget.GetRenderWindow(),
                    self.shader_texture_state['color_textures'][0],
                    self.shader_texture_state['opacity_textures'][0],
                    tf_nodes)
            self.render()
        except Exception as e:
            print(f"전역 TF 업데이트 실패: {e}")

    # ------------------------------------------------------------------
    # Local TF (custom shader)
    # ------------------------------------------------------------------
    def teardown_shader_textures(self, render_window):
        """local class 텍스처와 바인딩 observer를 해제한다."""
        if self.shader_observer_tag is not None and self.standard_mapper is not None:
            self.standard_mapper.RemoveObserver(self.shader_observer_tag)
            self.shader_observer_tag = None
        if self.shader_texture_state is not None and render_window is not None:
            self.shader_manager.release_all_textures(render_window, self.shader_texture_state)
        self.shader_texture_state = None

    def set_local_probability_maps(self, prob_maps):
        """local class 확률맵으로 custom shader를 구성한다.

        prob_maps: {label_id: (X,Y,Z) float32 0~1}. 서로 겹치지 않아야 한다.
        label_id는 정렬 순서대로 shader class 1..N에 매핑되고(label_to_class),
        class 0 = background(1 - Σ확률, global TF). 각 local class TF는 global 색 + opacity 1로
        임시 초기화되므로 호출 후 update_label_transfer_function으로 실제 TF를 넣어야 한다.
        Returns: 성공 여부
        """
        if self.standard_property is None or self.volume_data is None:
            print("set_local_probability_maps 실패: property 또는 volume_data가 준비되지 않음")
            return False
        if not prob_maps:
            print("set_local_probability_maps 실패: prob_maps가 비어있음")
            return False

        label_ids = sorted(prob_maps.keys())
        for lid in label_ids:
            if tuple(prob_maps[lid].shape[:3]) != tuple(self.volume_data.shape[:3]):
                print(f"set_local_probability_maps 실패: label {lid} shape {prob_maps[lid].shape}이 "
                      f"volume_data shape {self.volume_data.shape}와 불일치")
                return False
        self.label_to_class = {lid: i + 1 for i, lid in enumerate(label_ids)}

        render_window = self.vtk_widget.GetRenderWindow()
        self.teardown_shader_textures(render_window)

        num_classes = len(label_ids) + 1
        background = 1.0 - np.clip(sum(prob_maps[lid] for lid in label_ids), 0.0, 1.0)
        prob_maps_by_class = {0: background.astype(np.float32)}
        for lid in label_ids:
            prob_maps_by_class[self.label_to_class[lid]] = prob_maps[lid].astype(np.float32)

        global_tf = self.global_tf_nodes or DEFAULT_TF
        local_seed_tf = [[x, r, g, b, 1.0] for x, r, g, b, a in global_tf]
        tf_arrays_by_class = {0: global_tf}
        for lid in label_ids:
            tf_arrays_by_class[self.label_to_class[lid]] = local_seed_tf

        dims, prob_meta, color_meta, opacity_meta = self.shader_manager.build_class_textures_metadata(
            prob_maps_by_class, tf_arrays_by_class)
        prob_textures, color_textures, opacity_textures = self.shader_manager.create_all_textures(
            render_window, dims, prob_meta, color_meta, opacity_meta)
        self.shader_texture_state = {
            'prob_textures': prob_textures,
            'color_textures': color_textures,
            'opacity_textures': opacity_textures,
        }

        uniform_decs = (self.shader_manager.get_texture_declarations(num_classes)
                        + self.shader_manager.SHADING_UNIFORM_DECLS)
        shader_code = self.shader_manager.generate_shader_code(num_classes)
        self.shader_property.ClearAllFragmentShaderReplacements()
        self.shader_property.AddFragmentShaderReplacement(
            "//VTK::CustomUniforms::Dec", True, uniform_decs, False)
        self.shader_property.AddFragmentShaderReplacement(
            "//VTK::Shading::Impl", True, shader_code, False)

        self.shader_observer_tag = self.shader_manager.setup_texture_bindings(
            self.standard_mapper, self.shader_property, self.shader_texture_state)

        self.local_shader_active = True
        self.render()
        return True

    def clear_local_probability_maps(self):
        """custom shader를 해제하고 global TF 렌더링으로 돌아간다."""
        if self.shader_property is None:
            return
        render_window = self.vtk_widget.GetRenderWindow()
        self.teardown_shader_textures(render_window)
        self.shader_property.ClearAllFragmentShaderReplacements()
        self.local_shader_active = False
        self.label_to_class = {}
        render_window.Render()

    def update_label_transfer_function(self, tf_nodes, label=1, render=True):
        """label_id의 local TF 텍스처를 갱신한다. render=False면 다시 그리지 않는다."""
        if self.standard_property is None:
            return
        try:
            if self.local_shader_active and self.shader_texture_state is not None:
                color_textures = self.shader_texture_state['color_textures']
                opacity_textures = self.shader_texture_state['opacity_textures']
                class_idx = self.label_to_class.get(label)
                if class_idx is not None and class_idx < len(color_textures):
                    self.shader_manager.update_class_color_opacity_texture(
                        self.vtk_widget.GetRenderWindow(), color_textures[class_idx],
                        opacity_textures[class_idx], tf_nodes)
            if render:
                self.render()
        except Exception as e:
            print(f"local TF 업데이트 실패: {e}")

    # ------------------------------------------------------------------
    # Cluster 점 구름 오버레이
    # ------------------------------------------------------------------
    def set_point_cloud_actor(self, world_points, rgb_colors, point_size=4):
        """world_points (M,3)와 rgb_colors (M,3) 0~255로 점 구름 actor를 만들어 기존 것과 교체한다."""
        self.clear_roi_actor()
        if world_points is None or len(world_points) == 0:
            return

        points = vtk.vtkPoints()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("colors")
        for p, c in zip(world_points, rgb_colors):
            points.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
            colors.InsertNextTuple3(int(c[0]), int(c[1]), int(c[2]))

        poly = vtk.vtkPolyData()
        poly.SetPoints(points)
        poly.GetPointData().SetScalars(colors)

        glyph = vtk.vtkVertexGlyphFilter()
        glyph.SetInputData(poly)
        glyph.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.ScalarVisibilityOn()

        self.roi_actor = vtk.vtkActor()
        self.roi_actor.SetMapper(mapper)
        self.roi_actor.GetProperty().SetPointSize(point_size)
        self.renderer.AddActor(self.roi_actor)
        self.render()

    def show_cluster_points(self, world_points, cluster_labels, visible_clusters, best_cluster=None, point_size=4):
        """visible_clusters에 든 cluster만 라벨별 색(cluster_colors)으로 점 구름 표시한다."""
        if world_points is None or len(world_points) == 0:
            self.clear_roi_actor()
            return
        cluster_labels = np.asarray(cluster_labels)
        keep = np.isin(cluster_labels, visible_clusters)
        world_points, cluster_labels = world_points[keep], cluster_labels[keep]
        if len(world_points) == 0:
            self.clear_roi_actor()
            return
        self.set_point_cloud_actor(world_points, cluster_colors(cluster_labels, best_cluster), point_size)

    def clear_roi_actor(self):
        if self.roi_actor is not None:
            self.renderer.RemoveActor(self.roi_actor)
        self.roi_actor = None

    # ------------------------------------------------------------------
    # 클리핑 / 카메라 / 조명 / 샘플링 (매니저 위임)
    # ------------------------------------------------------------------
    def set_clipping_range(self, axis_index, min_pos, max_pos):
        self.clipping_manager.set_clipping_range(axis_index, min_pos, max_pos)

    def enable_clipping(self, enabled):
        self.clipping_manager.enable_clipping(enabled)

    def reset_clipping(self):
        self.clipping_manager.reset_clipping()

    def reset_camera_manual(self): self.camera_controller.reset_camera_manual()
    def set_zoom_factor(self, zoom_factor): return self.camera_controller.set_zoom_factor(zoom_factor)

    def set_shading(self, enabled): self.lighting_manager.set_shading(enabled)
    def set_ambient(self, ambient): self.lighting_manager.set_ambient(ambient)
    def set_diffuse(self, diffuse): self.lighting_manager.set_diffuse(diffuse)
    def set_specular(self, specular): self.lighting_manager.set_specular(specular)
    def set_ambient_color(self, r, g, b): self.lighting_manager.set_ambient_color(r, g, b)
    def set_diffuse_color(self, r, g, b): self.lighting_manager.set_diffuse_color(r, g, b)
    def set_specular_color(self, r, g, b): self.lighting_manager.set_specular_color(r, g, b)
    def set_light_position(self, light_type, x, y, z): self.lighting_manager.set_light_position(light_type, x, y, z)
    def set_follow_camera(self, enabled): self.lighting_manager.set_follow_camera(enabled)

    def set_background_color(self, color1, color2=None):
        """color2가 있으면 color1→color2 그라디언트 배경."""
        self.renderer.SetBackground(*color1[:3])
        if color2 is None:
            self.renderer.SetGradientBackground(False)
        else:
            self.renderer.SetBackground2(*color2[:3])
            self.renderer.SetGradientBackground(True)
        self.render()

    def set_ray_sampling_rate(self, rate):
        """sample distance = 1 / rate"""
        self.current_sample_distance = 1.0 / max(0.01, float(rate))
        if self.standard_mapper:
            self.standard_mapper.SetSampleDistance(self.current_sample_distance)
        self.render()

    # ------------------------------------------------------------------
    # 스크린샷
    # ------------------------------------------------------------------
    def save_current_rendering(self, use_square_ratio=False, filename=None):
        """현재 렌더 창을 PNG로 저장한다. use_square_ratio=True면 긴 변 기준 정사각형으로 잠시 바꿔 찍는다.
        filename을 주면 그 경로에(폴더 자동 생성), 아니면 SCREENSHOT_DIR에 시각 이름으로 저장한다.

        Returns: 저장 경로, 실패 시 None
        """
        try:
            if filename is None:
                suffix = "_1to1" if use_square_ratio else "_current"
                filename = os.path.join(SCREENSHOT_DIR,
                                        f"render_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}.png")
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)

            render_window = self.vtk_widget.GetRenderWindow()
            current_size = render_window.GetSize()
            if use_square_ratio:
                max_size = max(current_size)
                render_window.SetSize(max_size, max_size)
                render_window.Render()

            screenshot_filter = vtk.vtkWindowToImageFilter()
            screenshot_filter.SetInput(render_window)
            screenshot_filter.SetInputBufferTypeToRGBA()
            screenshot_filter.ReadFrontBufferOff()
            screenshot_filter.Update()

            writer = vtk.vtkPNGWriter()
            writer.SetFileName(filename)
            writer.SetInputConnection(screenshot_filter.GetOutputPort())
            writer.Write()

            if use_square_ratio:
                render_window.SetSize(current_size[0], current_size[1])
                render_window.Render()

            print(f"렌더링 저장: {filename}")
            return filename
        except Exception as e:
            print(f"렌더링 저장 실패: {e}")
            return None
