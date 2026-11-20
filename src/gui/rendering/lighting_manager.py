"""조명/shading 설정을 VTK 기본 경로(volume property, vtkLight)와 custom shader 파라미터에 함께 반영한다.

custom shader(local class가 있을 때 사용)는 widget.shader_params를 매 프레임 읽는다.
"""
import vtk


class LightingManager:
    """Key light(따뜻한 색, 0.8) + Fill light(차가운 색, 0.3) 두 개를 관리한다."""

    def __init__(self, renderer_widget):
        self.widget = renderer_widget
        self.key_light = None
        self.fill_light = None
        self.light_positions = {'key': [1, 1, 1], 'fill': [-1, 0.5, 0.5]}

    def setup_lighting(self):
        """기존 조명을 지우고 key/fill light를 추가한다. shader 광원 방향은 key light에 맞춘다."""
        if not self.widget.renderer:
            return
        self.widget.renderer.RemoveAllLights()

        self.key_light = self._make_light(self.light_positions['key'], (1.0, 0.95, 0.8), 0.8)
        self.fill_light = self._make_light(self.light_positions['fill'], (0.8, 0.9, 1.0), 0.3)

        key = self.light_positions['key']
        self.set_shader_param(light_x=key[0], light_y=key[1], light_z=key[2])

    def _make_light(self, position, color, intensity):
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(*position)
        light.SetFocalPoint(0, 0, 0)
        light.SetColor(*color)
        light.SetIntensity(intensity)
        self.widget.renderer.AddLight(light)
        return light

    def set_shader_param(self, **values):
        params = getattr(self.widget, 'shader_params', None)
        if params is not None:
            params.update(values)

    def render(self):
        self.widget.vtk_widget.GetRenderWindow().Render()

    def set_shading(self, enabled):
        self.set_shader_param(shading_enabled=bool(enabled))
        if self.widget.standard_volume:
            prop = self.widget.standard_volume.GetProperty()
            prop.ShadeOn() if enabled else prop.ShadeOff()
        self.render()

    def set_ambient(self, value):
        self.set_property('ambient', value)

    def set_diffuse(self, value):
        self.set_property('diffuse', value)

    def set_specular(self, value):
        self.set_property('specular', value)

    def set_property(self, prop_name, value):
        """prop_name: 'ambient' | 'diffuse' | 'specular'"""
        self.set_shader_param(**{prop_name: float(value)})
        if self.widget.standard_volume:
            prop = self.widget.standard_volume.GetProperty()
            {'ambient': prop.SetAmbient, 'diffuse': prop.SetDiffuse,
             'specular': prop.SetSpecular}[prop_name](value)
        self.render()

    def _set_light_color(self, param_name, setter_name, r, g, b):
        self.set_shader_param(**{param_name: [float(r), float(g), float(b)]})
        for light in (self.key_light, self.fill_light):
            if light:
                getattr(light, setter_name)(r, g, b)
        self.render()

    def set_ambient_color(self, r, g, b):
        self._set_light_color('ambient_color', 'SetAmbientColor', r, g, b)

    def set_diffuse_color(self, r, g, b):
        self._set_light_color('diffuse_color', 'SetDiffuseColor', r, g, b)

    def set_specular_color(self, r, g, b):
        self._set_light_color('specular_color', 'SetSpecularColor', r, g, b)

    def set_light_position(self, light_type, x, y, z):
        """light_type: 'key' | 'fill'. key light는 shader 광원 방향도 갱신한다."""
        if light_type == 'key' and self.key_light:
            self.key_light.SetPosition(x, y, z)
            self.light_positions['key'] = [x, y, z]
            self.set_shader_param(light_x=float(x), light_y=float(y), light_z=float(z))
        elif light_type == 'fill' and self.fill_light:
            self.fill_light.SetPosition(x, y, z)
            self.light_positions['fill'] = [x, y, z]
        self.render()

    def set_follow_camera(self, enabled):
        """조명을 카메라 기준(Camera Light) / 장면 기준(Scene Light)으로 전환한다."""
        self.set_shader_param(follow_camera=bool(enabled))
        for light in (self.key_light, self.fill_light):
            if light:
                light.SetLightTypeToCameraLight() if enabled else light.SetLightTypeToSceneLight()
        self.render()
