"""
VTK Volume Renderer with Native 2D Overlay
- Qt 오버레이 대신 VTK 자체 2D 렌더링 사용
- OS 독립적으로 동작
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from src.gui.rendering.clipping_manager import VolumeClippingManager
from src.gui.rendering.screenshot_manager import ScreenshotManager
from src.gui.rendering.camera_controller import CameraController
from src.gui.rendering.lighting_manager import LightingManager
import sys

import traceback
try:
    import vtk
    from vtkmodules.util import numpy_support
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False
    print("VTK 라이브러리가 설치되지 않았습니다.")


class VTKPointOverlay:
    """VTK 네이티브 2D 포인트 오버레이 - OS 독립적"""
    
    def __init__(self, renderer):
        self.renderer = renderer
        self.points_data = []  # [(x, y, type), ...]
        self.actors = []       # vtkActor2D 리스트
        self.visible = False
    
    def add_point(self, x, y, point_type):
        """2D 포인트 추가 (VTK 디스플레이 좌표 기준)"""
        self.points_data.append((x, y, point_type))
        
        # 색상 설정
        if point_type == "positive":
            color = (0.13, 0.59, 0.95)  # 파란색
        else:
            color = (0.96, 0.26, 0.21)  # 빨간색
        
        # 포인트를 위한 원 생성
        circle_source = vtk.vtkRegularPolygonSource()
        circle_source.SetNumberOfSides(20)  # 원처럼 보이게
        circle_source.SetRadius(8)  # 반지름 8픽셀
        circle_source.SetCenter(0, 0, 0)
        circle_source.Update()
        
        # 2D 매퍼
        mapper = vtk.vtkPolyDataMapper2D()
        mapper.SetInputConnection(circle_source.GetOutputPort())
        
        # 2D 액터
        actor = vtk.vtkActor2D()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)
        actor.GetProperty().SetOpacity(0.9)
        actor.SetPosition(x, y)  # VTK 디스플레이 좌표 (좌하단 기준)
        
        # 외곽선용 원
        outline_source = vtk.vtkRegularPolygonSource()
        outline_source.SetNumberOfSides(20)
        outline_source.SetRadius(9)
        outline_source.SetCenter(0, 0, 0)
        outline_source.GeneratePolygonOff()  # 외곽선만
        outline_source.Update()
        
        outline_mapper = vtk.vtkPolyDataMapper2D()
        outline_mapper.SetInputConnection(outline_source.GetOutputPort())
        
        outline_actor = vtk.vtkActor2D()
        outline_actor.SetMapper(outline_mapper)
        outline_actor.GetProperty().SetColor(1, 1, 1)  # 흰색 외곽선
        outline_actor.GetProperty().SetLineWidth(2)
        outline_actor.SetPosition(x, y)
        
        # 렌더러에 추가
        if self.visible:
            self.renderer.AddActor2D(outline_actor)
            self.renderer.AddActor2D(actor)
        
        self.actors.append((actor, outline_actor))
    
    def clear_points(self):
        """모든 포인트 제거"""
        for actor, outline_actor in self.actors:
            self.renderer.RemoveActor2D(actor)
            self.renderer.RemoveActor2D(outline_actor)
        self.actors = []
        self.points_data = []
    
    def set_visible(self, visible):
        """오버레이 표시/숨김"""
        self.visible = visible
        for actor, outline_actor in self.actors:
            if visible:
                self.renderer.AddActor2D(outline_actor)
                self.renderer.AddActor2D(actor)
            else:
                self.renderer.RemoveActor2D(actor)
                self.renderer.RemoveActor2D(outline_actor)
    
    def get_points(self):
        """저장된 포인트 데이터 반환"""
        return self.points_data.copy()


class VTKVolumeRenderer(QWidget):
    camera_angles_changed = pyqtSignal(float, float)
    point_2d_picked = pyqtSignal(int, int)  # VTK 디스플레이 좌표 (좌하단 기준)

    def __init__(self):
        super().__init__()
        vtk.vtkObject.GlobalWarningDisplayOff()
        
        if not VTK_AVAILABLE:
            self.setup_fallback_ui()
            return
            
        # VTK 위젯 설정
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        self.setLayout(layout)
        
        # VTK 파이프라인
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        
        # --- 피킹 및 인터랙션 상태 관리 ---
        self.picking_enabled = False
        self.current_pick_type = "positive"
        self.picking_observer_tag = None
        
        # ✅ VTK 네이티브 2D 오버레이
        self.point_overlay = VTKPointOverlay(self.renderer)

        # 렌더링 데이터
        self.volume_data = None # Wx. H x D numpy array
        self.voxel_spacing = (1.0, 1.0, 1.0)

        self.class_transfer_functions = {}
        self.class_volumes = {}
        self.class_mappers = {}
        self.class_properties = {}
        
        self.standard_volume = None
        self.standard_property = None
        self.standard_mapper = None
        
        self.current_sample_distance = 0.5

        self.clipping_manager = VolumeClippingManager(self)
        self.screenshot_manager = ScreenshotManager(self)
        self.camera_controller = CameraController(self)
        self.lighting_manager = LightingManager(self)
        
        self.setup_renderer()
        self.setup_interactor()

    def cleanup(self):
        """VTK 리소스 정리"""
        try:
            if hasattr(self, 'point_overlay'):
                self.point_overlay.clear_points()
            if hasattr(self, 'renderer') and self.renderer:
                camera = self.renderer.GetActiveCamera()
                if camera:
                    camera.RemoveAllObservers()
            self.volume_data = None
        except:
            pass

    # ============================================================
    # 2D 오버레이 관련 메서드
    # ============================================================
    
    def add_overlay_point(self, vtk_x, vtk_y, point_type):
        """VTK 좌표계로 포인트 추가 (좌하단 기준)"""
        self.point_overlay.add_point(vtk_x, vtk_y, point_type)
        self.vtk_widget.GetRenderWindow().Render()

    def clear_overlay_points(self):
        """오버레이 포인트 모두 제거"""
        self.point_overlay.clear_points()
        self.vtk_widget.GetRenderWindow().Render()
    
    def set_overlay_visible(self, visible):
        """오버레이 표시/숨김"""
        self.point_overlay.set_visible(visible)
        self.vtk_widget.GetRenderWindow().Render()

    # ============================================================
    # 피킹 및 카메라 제어
    # ============================================================

    def set_interaction_enabled(self, enabled):
        """카메라 조작 활성화/비활성화"""
        if enabled:
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.interactor.SetInteractorStyle(style)
            self.renderer.GetActiveCamera().AddObserver('ModifiedEvent', self.on_camera_modified)
        else:
            from vtkmodules.vtkInteractionStyle import vtkInteractorStyleUser
            style = vtkInteractorStyleUser()
            self.interactor.SetInteractorStyle(style)
        
        self.setup_picking_observer()

    def set_picking_enabled(self, enabled):
        """피킹 모드 활성화 여부"""
        self.picking_enabled = enabled
        self.point_overlay.set_visible(enabled)
        self.vtk_widget.GetRenderWindow().Render()

    def setup_picking_observer(self):
        """마우스 클릭 이벤트 관찰자 설정"""
        if self.picking_observer_tag is not None:
            self.interactor.RemoveObserver(self.picking_observer_tag)
            self.picking_observer_tag = None

        self.picking_observer_tag = self.interactor.AddObserver(
            "LeftButtonPressEvent", self.on_left_button_press, 10.0
        )

    def on_left_button_press(self, obj, event):
        """마우스 클릭 시 좌표 추출 - VTK 네이티브 좌표 사용"""
        if not self.picking_enabled:
            return
        click_pos = self.interactor.GetEventPosition()
        vtk_x, vtk_y = click_pos[0], click_pos[1]
        
        # 시그널 발생 (VTK 좌표 그대로)
        self.point_2d_picked.emit(vtk_x, vtk_y)

    # ============================================================
    # 기존 메서드들
    # ============================================================

    def set_clipping_range(self, axis_index, min_pos, max_pos):
        if self.clipping_manager:
            self.clipping_manager.set_clipping_range(axis_index, min_pos, max_pos)

    def enable_clipping(self, enabled):
        if self.clipping_manager:
            self.clipping_manager.enable_clipping(enabled)

    def reset_clipping(self):
        if self.clipping_manager:
            self.clipping_manager.reset_clipping()

    def set_volume_data(self, volume_data):
        if not VTK_AVAILABLE or self.renderer is None:
            return
        
        is_first_load = (self.volume_data is None)
        if not is_first_load:
            self.save_camera_state()
        
        self.volume_data = volume_data
        print(self.volume_data.shape)
        
        if volume_data is not None: 
            self.clear_all_volumes()
            self._setup_standard_volume(volume_data)

            self.clipping_manager.reset_clipping() 
            self.clipping_manager.update_clipping_target()
            
            if is_first_load:
                self.setup_camera(force_reset=True)
                self.reset_camera_manual()
            else:
                if not self.restore_camera_state():
                    self.setup_camera(force_reset=True)
            
            if hasattr(self, 'vtk_widget') and self.vtk_widget:
                self.vtk_widget.GetRenderWindow().Render()

    def _setup_standard_volume(self, volume_data):
        try:
            self.clear_all_volumes()
            vtk_data = self._numpy_to_vtk_imagedata(volume_data, self.voxel_spacing)

            self.standard_mapper = vtk.vtkGPUVolumeRayCastMapper()
            self.standard_mapper.SetBlendModeToComposite()
            self.standard_mapper.SetSampleDistance(self.current_sample_distance)
            self.standard_mapper.SetAutoAdjustSampleDistances(False)
            self.standard_mapper.SetInputData(vtk_data)
            
            self.standard_property = vtk.vtkVolumeProperty()
            self.standard_property.SetInterpolationTypeToLinear()

            if self.class_transfer_functions:
                default_tf = self.class_transfer_functions[0]
            else:
                default_tf = self._create_default_tf_array()
            color_func, opacity_func = self._create_vtk_tf_from_array(default_tf)
            self.standard_property.SetColor(color_func)
            self.standard_property.SetScalarOpacity(opacity_func)

            self.standard_volume = vtk.vtkVolume()
            self.standard_volume.SetMapper(self.standard_mapper)
            self.standard_volume.SetProperty(self.standard_property)

            self.renderer.AddVolume(self.standard_volume)
            self.class_transfer_functions = {0: default_tf}

            if self.clipping_manager:
                self.clipping_manager.reset_clipping()
                if self.clipping_manager.clipping_enabled:
                    self.clipping_manager.update_clipping_target()
                
            print("Intensity 모드 볼륨 설정 완료")
            
        except Exception as e:
            print(f"표준 볼륨 설정 실패: {e}")
            traceback.print_exc()

    def _create_vtk_tf_from_array(self, tf_nodes, return_array=False):
        tf_array = np.zeros((256, 4))
        if not tf_nodes or len(tf_nodes) < 2:
            return None, None
        
        sorted_nodes = sorted(tf_nodes, key=lambda x: x[0])
        
        def interpolate_value(intensity, channel):
            if intensity <= sorted_nodes[0][0]:
                return sorted_nodes[0][channel]
            if intensity >= sorted_nodes[-1][0]:
                return sorted_nodes[-1][channel]
            for i in range(len(sorted_nodes) - 1):
                if sorted_nodes[i][0] <= intensity <= sorted_nodes[i+1][0]:
                    t = (intensity - sorted_nodes[i][0]) / (sorted_nodes[i+1][0] - sorted_nodes[i][0])
                    return sorted_nodes[i][channel] * (1-t) + sorted_nodes[i+1][channel] * t
            return 0.0
        
        color_func = vtk.vtkColorTransferFunction()
        opacity_func = vtk.vtkPiecewiseFunction()
        
        for i in range(256):
            normalized_x = i / 255.0
            r = interpolate_value(normalized_x, 1)
            g = interpolate_value(normalized_x, 2) 
            b = interpolate_value(normalized_x, 3)
            alpha = interpolate_value(normalized_x, 4)
            
            color_func.AddRGBPoint(normalized_x, r, g, b)
            opacity_func.AddPoint(normalized_x, alpha)
            tf_array[i] = [r, g, b, alpha]

        if return_array:
            color_array = np.zeros((256,256, 3), dtype=np.uint8)
            opacity_array = np.zeros((256,256, 1), dtype=np.uint8)
            for y in range(256):
                for x in range(256):
                    color_array[y, x, 0] = int(tf_array[x, 0] * 255)
                    color_array[y, x, 1] = int(tf_array[x, 1] * 255)
                    color_array[y, x, 2] = int(tf_array[x, 2] * 255)
                    opacity_array[y, x, 0] = int(tf_array[x, 3] * 255)
            return color_array, opacity_array

        return color_func, opacity_func

    def _create_default_tf_array(self, prob=False):
        return [[0.0, 0.0, 0.0, 0.0, 0.0], [0.3, 1.0, 0.5, 0.0, 0.1], [0.7, 1.0, 1.0, 1.0, 0.8], [1.0, 1.0, 1.0, 1.0, 1.0]]

    def update_transfer_function_optimized(self, tf_nodes):
        try:
            self.class_transfer_functions[0] = tf_nodes
            if self.standard_volume:
                color_func, opacity_func = self._create_vtk_tf_from_array(tf_nodes)
                volume_property = self.standard_volume.GetProperty()
                volume_property.SetColor(color_func)
                volume_property.SetScalarOpacity(opacity_func)
            if hasattr(self, 'vtk_widget') and self.vtk_widget:
                self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            print(f"전역 TF 업데이트 실패: {e}")

    def clear_all_volumes(self):
        if self.standard_volume and self.renderer:
            self.renderer.RemoveVolume(self.standard_volume)
        self.standard_volume = None
        self.standard_mapper = None

    def _numpy_to_vtk_imagedata(self, numpy_array, voxel_spacing):
        try:
            vtk_data = vtk.vtkImageData()
            dims = (numpy_array.shape[0], numpy_array.shape[1], numpy_array.shape[2])
            vtk_data.SetDimensions(dims)
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

    def setup_renderer(self):
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.renderer.SetBackground2(0.8, 0.8, 0.8)
        self.renderer.SetGradientBackground(True)
        self.setup_lighting()
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.SetMultiSamples(8)

    def setup_interactor(self):
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        style.SetMotionFactor(2.0)
        self.renderer.GetActiveCamera().AddObserver('ModifiedEvent', self.on_camera_modified)
        self.setup_picking_observer()

    def on_camera_modified(self, obj, event):
        if hasattr(self, 'camera_controller') and self.camera_controller.is_sync_in_progress():
            return
        angles = self.get_camera_angles()
        if angles is not None:
            self.camera_angles_changed.emit(angles[0], angles[1])

    def save_camera_state(self): return self.camera_controller.save_camera_state()
    def restore_camera_state(self): return self.camera_controller.restore_camera_state()
    def get_camera_state(self): return self.camera_controller.get_camera_state()
    def set_camera_state(self, state): return self.camera_controller.set_camera_state(state)
    def setup_camera(self, force_reset=False): self.camera_controller.setup_camera(force_reset)
    def reset_camera_manual(self): self.camera_controller.reset_camera_manual()
    def get_current_zoom_factor(self): return self.camera_controller.get_current_zoom_factor()
    def set_zoom_factor(self, zoom_factor): return self.camera_controller.set_zoom_factor(zoom_factor)
    def get_camera(self): return self.camera_controller.get_camera()
    
    def set_camera_angles(self, longitude: float, latitude: float):
        if hasattr(self, 'camera_controller'):
            return self.camera_controller.set_camera_from_angles(longitude, latitude)
        return False

    def get_camera_angles(self):
        if hasattr(self, 'camera_controller'):
            return self.camera_controller.get_camera_angles()
        return (0.0, 0.0)
    
    def set_background_color(self, color1, color2=None):
        if not VTK_AVAILABLE or self.renderer is None: return
        if color2 is None:
            self.renderer.SetBackground(color1[0], color1[1], color1[2])
            self.renderer.SetGradientBackground(False)
        else:
            self.renderer.SetBackground(color1[0], color1[1], color1[2])
            self.renderer.SetBackground2(color2[0], color2[1], color2[2])
            self.renderer.SetGradientBackground(True)
        if hasattr(self, 'vtk_widget'): self.vtk_widget.GetRenderWindow().Render()

    def set_sample_distance(self, distance):
        if not VTK_AVAILABLE: return
        try:
            self.current_sample_distance = distance
            if self.standard_mapper: self.standard_mapper.SetSampleDistance(distance)
            if hasattr(self, 'vtk_widget'): self.vtk_widget.GetRenderWindow().Render()
        except Exception as e: print(f"샘플링 거리 설정 실패: {e}")

    def get_renderer(self): return self.renderer

    def setup_fallback_ui(self):
        layout = QVBoxLayout()
        label = QLabel("VTK가 설치되지 않았습니다.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        self.setLayout(layout)

    def setup_lighting(self): self.lighting_manager.setup_lighting()
    def set_shading(self, enabled): self.lighting_manager.set_shading(enabled)
    def set_ambient(self, ambient): self.lighting_manager.set_ambient(ambient)
    def set_diffuse(self, diffuse): self.lighting_manager.set_diffuse(diffuse)
    def set_specular(self, specular): self.lighting_manager.set_specular(specular)
    def set_ambient_color(self, r, g, b): self.lighting_manager.set_ambient_color(r, g, b)
    def set_diffuse_color(self, r, g, b): self.lighting_manager.set_diffuse_color(r, g, b)
    def set_specular_color(self, r, g, b): self.lighting_manager.set_specular_color(r, g, b)
    def set_light_position(self, light_type, x, y, z): self.lighting_manager.set_light_position(light_type, x, y, z)
    def set_follow_camera(self, enabled): self.lighting_manager.set_follow_camera(enabled)

    def set_ray_sampling_rate(self, rate):
        try:
            sample_distance = 1.0 / max(0.01, float(rate))
            self.set_sample_distance(sample_distance)
        except: pass

    def save_current_rendering(self, use_square_ratio=False): return self.screenshot_manager.save_current_rendering(use_square_ratio)
    def export_screenshot(self, filename, resolution=(1920, 1080)): return self.screenshot_manager.export_screenshot(filename, resolution)

    def get_world_position_from_display(self, x, y):
        """[Tracking] 화면 클릭(2D) -> 3D World 좌표 반환"""
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.005)
        if picker.Pick(x, y, 0, self.renderer):
            return picker.GetPickPosition()
        return None

    def project_world_to_display(self, world_pos):
        """[Tracking] 3D World 좌표 -> 현재 뷰의 2D 화면 좌표 변환"""
        coordinate = vtk.vtkCoordinate()
        coordinate.SetCoordinateSystemToWorld()
        coordinate.SetValue(world_pos)
        display_coord = coordinate.GetComputedDisplayValue(self.renderer)
        return display_coord # (x, y)

    def get_depth_map_array(self):
        """[Optimizer] 현재 뷰의 Depth Map (Z-Buffer) 추출 (0.0=Near, 1.0=Far)"""
        width, height = self.vtk_widget.GetRenderWindow().GetSize()
        
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.vtk_widget.GetRenderWindow())
        window_to_image.SetInputBufferTypeToZBuffer()
        window_to_image.ReadFrontBufferOff()
        window_to_image.Update()
        
        vtk_image = window_to_image.GetOutput()
        depth_data = vtkmodules.util.numpy_support.vtk_to_numpy(vtk_image.GetPointData().GetScalars())
        depth_data = depth_data.reshape(height, width)
        return np.flipud(depth_data) # Y축 반전

    def get_camera_matrices(self):
        """[Space Carving] 3D 투영을 위한 View, Projection 행렬 반환"""
        cam = self.renderer.GetActiveCamera()
        
        # 1. View Matrix
        view_matrix = cam.GetModelViewTransformMatrix()
        # 2. Projection Matrix (Aspect Ratio 등 포함)
        aspect = self.renderer.GetTiledAspectRatio()
        proj_matrix = cam.GetProjectionTransformMatrix(aspect, -1, 1)
        
        # 4x4 행렬을 numpy array로 변환
        v_mat = np.zeros((4, 4))
        p_mat = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                v_mat[i, j] = view_matrix.GetElement(i, j)
                p_mat[i, j] = proj_matrix.GetElement(i, j)
                
        return v_mat, p_mat