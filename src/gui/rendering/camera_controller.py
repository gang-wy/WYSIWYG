"""렌더러 카메라의 초기 배치, 상태 저장/복원, zoom(view angle) 제어."""


class CameraController:
    """zoom factor = 30° / view angle."""

    def __init__(self, renderer_widget):
        self.widget = renderer_widget
        self.saved_camera_state = None

    def get_camera(self):
        return self.widget.renderer.GetActiveCamera() if self.widget.renderer else None

    def render(self):
        self.widget.vtk_widget.GetRenderWindow().Render()

    def save_camera_state(self):
        """현재 카메라 위치/초점/up/view angle/clipping range를 저장한다. Returns: 성공 여부"""
        camera = self.get_camera()
        if not camera:
            return False
        self.saved_camera_state = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'view_angle': camera.GetViewAngle(),
            'clipping_range': camera.GetClippingRange(),
        }
        return True

    def restore_camera_state(self):
        """save_camera_state로 저장한 상태를 적용한다. Returns: 성공 여부"""
        camera = self.get_camera()
        if not (camera and self.saved_camera_state):
            return False
        try:
            state = self.saved_camera_state
            camera.SetPosition(state['position'])
            camera.SetFocalPoint(state['focal_point'])
            camera.SetViewUp(state['view_up'])
            camera.SetViewAngle(state['view_angle'])
            camera.SetClippingRange(state['clipping_range'])
            self.render()
            return True
        except Exception as e:
            print(f"카메라 상태 복원 실패: {e}")
            return False

    def setup_camera(self, force_reset=False):
        """기본 시점(view up = +Z, 30°)으로 두고 볼륨에 맞춰 ResetCamera + Zoom(0.8).

        force_reset=False이고 저장된 상태가 있으면 그 상태를 복원한다.
        """
        if not self.widget.renderer:
            return
        if not force_reset and self.saved_camera_state and self.restore_camera_state():
            return

        camera = self.get_camera()
        if camera:
            camera.SetPosition(1.5, 1.5, 1.5)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 0, 1)
            camera.SetViewAngle(30)
            camera.SetClippingRange(0.1, 100.0)
            if force_reset or self.saved_camera_state is None:
                self.widget.renderer.ResetCamera()
                camera.Zoom(0.8)

    def reset_camera_manual(self):
        self.saved_camera_state = None
        self.setup_camera(force_reset=True)

    def set_zoom_factor(self, zoom_factor):
        """view angle = 30 / zoom_factor로 설정한다. Returns: 성공 여부"""
        camera = self.get_camera()
        if not camera:
            return False
        camera.SetViewAngle(30.0 / max(0.1, zoom_factor))
        self.render()
        return True
