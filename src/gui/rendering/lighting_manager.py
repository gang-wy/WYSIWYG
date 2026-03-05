"""
Lighting Manager
renderer_widget.py에서 조명 관리 로직이 분리되었습니다.
"""
try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

class LightingManager:
    """VTK 및 커스텀 셰이더 조명을 통합 관리하는 클래스"""

    def __init__(self, renderer_widget):
        self.widget = renderer_widget
        self.key_light = None
        self.fill_light = None
        
        # 기본 조명 위치 정보 (기존 renderer_widget에서 이동)
        self.light_positions = {
            'key': [1, 1, 1],
            'fill': [-1, 0.5, 0.5]
        }

    def setup_lighting(self):
        """초기 조명(Key, Fill) 설정"""
        if not VTK_AVAILABLE or not self.widget.renderer:
            return

        self.widget.renderer.RemoveAllLights()
        
        # 키 라이트 생성
        self.key_light = vtk.vtkLight()
        self.key_light.SetLightTypeToSceneLight()
        self.key_light.SetPosition(*self.light_positions['key'])
        self.key_light.SetFocalPoint(0, 0, 0)
        self.key_light.SetColor(1.0, 0.95, 0.8)
        self.key_light.SetIntensity(0.8)
        self.widget.renderer.AddLight(self.key_light)
        
        # 필 라이트 생성
        self.fill_light = vtk.vtkLight()
        self.fill_light.SetLightTypeToSceneLight()
        self.fill_light.SetPosition(*self.light_positions['fill'])
        self.fill_light.SetFocalPoint(0, 0, 0)
        self.fill_light.SetColor(0.8, 0.9, 1.0)
        self.fill_light.SetIntensity(0.3)
        self.widget.renderer.AddLight(self.fill_light)

    def set_shading(self, enabled):
        """쉐이딩(Shading) 활성화 여부 설정"""
        try:
            if self.widget.standard_volume:
                prop = self.widget.standard_volume.GetProperty()
                if enabled: prop.ShadeOn()
                else: prop.ShadeOff()
            
            self._render()
        except Exception as e:
            print(f"쉐이딩 설정 실패: {e}")

    def set_ambient(self, value):
        self._set_property('ambient', value)

    def set_diffuse(self, value):
        self._set_property('diffuse', value)

    def set_specular(self, value):
        self._set_property('specular', value)

    def _set_property(self, prop_name, value):
        """Ambient, Diffuse, Specular 값 공통 설정 로직"""
        try:
            if self.widget.standard_volume:
                prop = self.widget.standard_volume.GetProperty()
                if prop_name == 'ambient': prop.SetAmbient(value)
                elif prop_name == 'diffuse': prop.SetDiffuse(value)
                elif prop_name == 'specular': prop.SetSpecular(value)
            self._render()
        except Exception as e:
            print(f"{prop_name} 설정 실패: {e}")

    def set_ambient_color(self, r, g, b):
        if self.widget.standard_volume:
            if self.key_light: self.key_light.SetAmbientColor(r, g, b)
            if self.fill_light: self.fill_light.SetAmbientColor(r, g, b)
        else:
            self.widget._apply_shader_changes()
        self._render()

    def set_diffuse_color(self, r, g, b):
        if self.widget.standard_volume:
            if self.key_light: self.key_light.SetDiffuseColor(r, g, b)
            if self.fill_light: self.fill_light.SetDiffuseColor(r, g, b)
        else:
            self.widget._apply_shader_changes()
        self._render()

    def set_specular_color(self, r, g, b):
        if self.widget.standard_volume:
            if self.key_light: self.key_light.SetSpecularColor(r, g, b)
            if self.fill_light: self.fill_light.SetSpecularColor(r, g, b)
        else:
            self.widget._apply_shader_changes()
        self._render()

    def set_light_position(self, light_type, x, y, z):
        """라이트 위치 설정"""
        if light_type == 'key' and self.key_light:
            self.key_light.SetPosition(x, y, z)
            self.light_positions['key'] = [x, y, z]
        elif light_type == 'fill' and self.fill_light:
            self.fill_light.SetPosition(x, y, z)
            self.light_positions['fill'] = [x, y, z]
        self._render()

    def set_follow_camera(self, enabled):
        """Follow Camera 모드 설정"""
        mode = vtk.vtkLight.SetLightTypeToCameraLight if enabled else vtk.vtkLight.SetLightTypeToSceneLight
        if self.key_light: mode(self.key_light)
        if self.fill_light: mode(self.fill_light)
        self._render()

    def _render(self):
        """렌더링 갱신"""
        if hasattr(self.widget, 'vtk_widget'):
            self.widget.vtk_widget.GetRenderWindow().Render()