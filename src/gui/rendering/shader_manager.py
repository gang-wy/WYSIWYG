"""multi-class local TF용 custom GLSL fragment shader와 GPU 텍스처를 만든다.

class마다 확률 3D 텍스처(0~1) + intensity-indexed color/opacity LUT 2D 텍스처를 올리고
shader에서 Σ alpha_i × proba_i로 블렌딩한다. class 0은 background(global TF)이며
proba_0 = 1 - clip(Σ 다른 class 확률, 0, 1).
"""
import numpy as np
import vtk


class ShaderManager:
    """GLSL 코드 생성, 텍스처 생성/갱신/해제, 매 프레임 uniform 바인딩 콜백 등록."""

    LUT_WIDTH = 256

    def __init__(self, renderer_widget):
        self.widget = renderer_widget

    def get_texture_declarations(self, num_classes):
        """Returns: class 0..num_classes-1의 probTex/colorTex/opacityTex uniform 선언 GLSL"""
        declarations = []
        for i in range(num_classes):
            declarations.append(f"uniform sampler3D probTex{i};")
            declarations.append(f"uniform sampler2D colorTex{i};")
            declarations.append(f"uniform sampler2D opacityTex{i};")
        return "\n".join(declarations)

    def generate_shader_code(self, num_classes):
        """Returns: //VTK::Shading::Impl 자리에 들어갈 GLSL

        class별 color/alpha를 확률 가중 합으로 섞고(정규화 없음), 원본 intensity gradient로
        shading한 뒤 front-to-back으로 합성한다.
        """
        lines = [
            "vec3 coord = g_dataPos;",
            "float h = 0.005;",
            "float intensity = texture(in_volume[0], coord).r;",
            # LUT 표본 i(= i/255)가 텍셀 중심 (i+0.5)/256에 오도록 좌표 보정
            f"float lut_u = (intensity * {float(self.LUT_WIDTH - 1)} + 0.5) / {float(self.LUT_WIDTH)};",
            "",
            "float val_xp = texture(in_volume[0], coord + vec3(h,0,0)).r;",
            "float val_xn = texture(in_volume[0], coord - vec3(h,0,0)).r;",
            "float val_yp = texture(in_volume[0], coord + vec3(0,h,0)).r;",
            "float val_yn = texture(in_volume[0], coord - vec3(0,h,0)).r;",
            "float val_zp = texture(in_volume[0], coord + vec3(0,0,h)).r;",
            "float val_zn = texture(in_volume[0], coord - vec3(0,0,h)).r;",
            "vec3 gradient = vec3(val_xp-val_xn, val_yp-val_yn, val_zp-val_zn) / (2.0*h);",
            "vec3 normal = (length(gradient) > 0.0001) ? normalize(-gradient) : vec3(0.0,0.0,1.0);",
            "",
        ]
        for i in range(num_classes):
            lines.append(f"float proba{i} = texture(probTex{i}, coord).r;")
        lines.append("")
        for i in range(num_classes):
            lines.append(f"vec3 color{i} = texture(colorTex{i}, vec2(lut_u, 0.5)).rgb;")
            lines.append(f"float alpha{i} = texture(opacityTex{i}, vec2(lut_u, 0.5)).r;")
        lines.append("")
        # VTK 기본 경로와 같은 sample distance opacity 보정: 1 - (1 - alpha)^(SampleDistance / UnitDistance)
        for i in range(num_classes):
            lines.append(f"alpha{i} = 1.0 - pow(max(1.0 - alpha{i}, 0.0), u_opacityCorrectionExponent);")
        lines.append("")
        lines.append("vec3 blendedColorPremultiplied = vec3(0.0);")
        lines.append("float blendedAlpha = 0.0;")
        for i in range(num_classes):
            lines.append(f"blendedColorPremultiplied += color{i} * alpha{i} * proba{i};")
            lines.append(f"blendedAlpha += alpha{i} * proba{i};")
        lines.append("")
        lines.append("float renderThreshold = 0.001;")
        lines.append("if (blendedAlpha > renderThreshold) {")
        lines.append("    if (u_shadingEnabled == 1) {")
        lines.append("        vec3 lightDirRaw = u_lightDir;")
        lines.append("        vec3 lightDir;")
        lines.append("        if (u_followCamera == 1) {")
        lines.append("            lightDir = normalize((u_inverseViewMatrix * vec4(lightDirRaw, 0.0)).xyz);")
        lines.append("        } else {")
        lines.append("            lightDir = normalize(lightDirRaw);")
        lines.append("        }")
        lines.append("        vec3 viewDir = normalize(vec3(0.0, 0.0, 1.0));")
        lines.append("        float NdotL = abs(dot(normal, lightDir));")
        lines.append("        vec3 r = normalize(2.0 * NdotL * normal - lightDir);")
        lines.append("        float RdotV = max(dot(viewDir, r), 0.0);")
        lines.append("        float spec = pow(RdotV, u_shininess);")
        lines.append("        vec3 ambientColor = blendedColorPremultiplied * u_ambientColor * u_ambient;")
        lines.append("        vec3 diffuseColor = blendedColorPremultiplied * u_diffuseColor * u_diffuse * NdotL;")
        # g_srcColor.rgb는 premultiplied — specular에도 alpha를 곱한다
        lines.append("        vec3 specularColor = u_specularColor * u_specular * spec * blendedAlpha;")
        lines.append("        g_srcColor = vec4(ambientColor + diffuseColor + specularColor, blendedAlpha);")
        lines.append("    } else {")
        lines.append("        g_srcColor = vec4(blendedColorPremultiplied, blendedAlpha);")
        lines.append("    }")
        lines.append("    g_fragColor = (1.0 - g_fragColor.a) * g_srcColor + g_fragColor;")
        lines.append("}")
        return "\n".join(lines)

    SHADING_UNIFORM_DECLS = """
uniform int u_shadingEnabled;
uniform mat4 u_inverseViewMatrix;
uniform int u_followCamera;
uniform vec3 u_lightDir;
uniform float u_ambient;
uniform float u_diffuse;
uniform float u_specular;
uniform float u_shininess;
uniform vec3 u_ambientColor;
uniform vec3 u_diffuseColor;
uniform vec3 u_specularColor;
uniform float u_opacityCorrectionExponent;
"""

    def build_class_textures_metadata(self, prob_maps_by_class, tf_arrays_by_class):
        """텍스처에 올릴 uint8 데이터를 준비한다(GPU 업로드 전).

        prob_maps_by_class: {class_id: (X,Y,Z) float32 0~1}, tf_arrays_by_class: {class_id: tf_nodes}
        (class_id는 0..N-1 연속)
        Returns: (dims, prob_meta, color_meta, opacity_meta) — meta는 [{'index', 'data'}, ...]
        """
        num_classes = len(prob_maps_by_class)
        prob_meta, color_meta, opacity_meta = [], [], []
        for class_id in range(num_classes):
            prob = prob_maps_by_class[class_id]
            prob_8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
            prob_meta.append({'index': class_id, 'data': prob_8.ravel(order='F')})

            color_arr, opacity_arr = self.widget.create_vtk_tf_from_array(
                tf_arrays_by_class[class_id], return_array=True
            )
            color_meta.append({'index': class_id, 'data': color_arr[:1]})       # (1,256,3) uint8
            opacity_meta.append({'index': class_id, 'data': opacity_arr[:1]})   # (1,256,1) uint8

        dims = next(iter(prob_maps_by_class.values())).shape
        return dims, prob_meta, color_meta, opacity_meta

    def create_all_textures(self, render_window, dims, prob_meta, color_meta, opacity_meta):
        """meta 데이터로 vtkTextureObject를 만든다. Returns: (prob_textures, color_textures, opacity_textures)"""
        prob_textures, color_textures, opacity_textures = [], [], []
        for meta in prob_meta:
            tex = vtk.vtkTextureObject()
            tex.SetContext(render_window)
            tex.Create3DFromRaw(dims[0], dims[1], dims[2], 1, vtk.VTK_UNSIGNED_CHAR, meta['data'])
            self.set_texture_parameters(tex, is_3d=True)
            prob_textures.append(tex)
        for meta in color_meta:
            tex = self.make_2d_lut_texture(render_window, meta['data'], channels=3, nearest=False)
            color_textures.append(tex)
        for meta in opacity_meta:
            tex = self.make_2d_lut_texture(render_window, meta['data'], channels=1, nearest=True)
            opacity_textures.append(tex)
        return prob_textures, color_textures, opacity_textures

    def make_2d_lut_texture(self, render_window, arr_1xWxC_uint8, channels, nearest):
        """(1, 256, C) uint8 LUT → 256×1 2D 텍스처."""
        tex = vtk.vtkTextureObject()
        tex.SetContext(render_window)
        tex.Create2DFromRaw(self.LUT_WIDTH, 1, channels, vtk.VTK_UNSIGNED_CHAR, arr_1xWxC_uint8.tobytes())
        self.set_texture_parameters(tex, is_3d=False, nearest=nearest)
        return tex

    def set_texture_parameters(self, tex, is_3d=False, nearest=False):
        tex.SetWrapS(vtk.vtkTextureObject.ClampToEdge)
        tex.SetWrapT(vtk.vtkTextureObject.ClampToEdge)
        if is_3d:
            tex.SetWrapR(vtk.vtkTextureObject.ClampToEdge)
        f = vtk.vtkTextureObject.Nearest if nearest else vtk.vtkTextureObject.Linear
        tex.SetMinificationFilter(f)
        tex.SetMagnificationFilter(f)

    def update_class_color_opacity_texture(self, render_window, color_tex, opacity_tex, tf_nodes):
        """class 하나의 color/opacity LUT 텍스처를 tf_nodes로 다시 업로드한다."""
        color_arr, opacity_arr = self.widget.create_vtk_tf_from_array(tf_nodes, return_array=True)

        color_tex.ReleaseGraphicsResources(render_window)
        color_tex.SetContext(render_window)
        color_tex.Create2DFromRaw(self.LUT_WIDTH, 1, 3, vtk.VTK_UNSIGNED_CHAR, color_arr[:1].tobytes())
        self.set_texture_parameters(color_tex, is_3d=False, nearest=False)

        opacity_tex.ReleaseGraphicsResources(render_window)
        opacity_tex.SetContext(render_window)
        opacity_tex.Create2DFromRaw(self.LUT_WIDTH, 1, 1, vtk.VTK_UNSIGNED_CHAR, opacity_arr[:1].tobytes())
        self.set_texture_parameters(opacity_tex, is_3d=False, nearest=True)

    def setup_texture_bindings(self, mapper, shader_property, state):
        """매 프레임(VolumeMapperRenderStartEvent) 텍스처와 shading uniform을 바인딩하는 콜백을 등록한다.

        state: {'prob_textures', 'color_textures', 'opacity_textures'}
        shading on/off와 ambient/diffuse/specular/광택은 volume property 값을, 광원 방향/색과
        follow camera는 widget.shader_params를 쓴다.
        Returns: observer tag
        """

        def bind_all_textures(obj, event):
            uniforms = shader_property.GetFragmentCustomUniforms()
            for i, tex in enumerate(state['prob_textures']):
                tex.Activate()
                uniforms.SetUniformi(f"probTex{i}", tex.GetTextureUnit())
            for i, tex in enumerate(state['color_textures']):
                tex.Activate()
                uniforms.SetUniformi(f"colorTex{i}", tex.GetTextureUnit())
            for i, tex in enumerate(state['opacity_textures']):
                tex.Activate()
                uniforms.SetUniformi(f"opacityTex{i}", tex.GetTextureUnit())

            params = self.widget.shader_params
            prop = self.widget.standard_property
            if prop is not None:
                shading_enabled = bool(prop.GetShade())
                ambient, diffuse = prop.GetAmbient(), prop.GetDiffuse()
                specular, shininess = prop.GetSpecular(), prop.GetSpecularPower()
            else:
                shading_enabled = params.get('shading_enabled', False)
                ambient, diffuse = params.get('ambient', 0.3), params.get('diffuse', 0.7)
                specular, shininess = params.get('specular', 0.5), params.get('shininess', 32.0)
            uniforms.SetUniformi("u_shadingEnabled", 1 if shading_enabled else 0)
            uniforms.SetUniformi("u_followCamera", 1 if params.get('follow_camera', False) else 0)
            uniforms.SetUniform3f("u_lightDir", [params.get('light_x', -1.0),
                                                  params.get('light_y', 0.5),
                                                  params.get('light_z', 0.5)])
            uniforms.SetUniformf("u_ambient", float(ambient))
            uniforms.SetUniformf("u_diffuse", float(diffuse))
            uniforms.SetUniformf("u_specular", float(specular))
            uniforms.SetUniformf("u_shininess", float(shininess))
            uniforms.SetUniform3f("u_ambientColor", params.get('ambient_color', [1.0, 1.0, 1.0]))
            uniforms.SetUniform3f("u_diffuseColor", params.get('diffuse_color', [1.0, 1.0, 1.0]))
            uniforms.SetUniform3f("u_specularColor", params.get('specular_color', [1.0, 1.0, 1.0]))
            if params.get('follow_camera', False):
                self.update_inverse_view_matrix(uniforms)

            sample_distance = getattr(self.widget, 'current_sample_distance', 1.0)
            unit_distance = 1.0
            if self.widget.standard_property is not None:
                unit_distance = self.widget.standard_property.GetScalarOpacityUnitDistance() or 1.0
            uniforms.SetUniformf("u_opacityCorrectionExponent", sample_distance / unit_distance)

        tag = mapper.AddObserver(vtk.vtkCommand.VolumeMapperRenderStartEvent, bind_all_textures)
        return tag

    def update_inverse_view_matrix(self, uniforms):
        """follow camera용 view 행렬 역행렬을 u_inverseViewMatrix에 넣는다."""
        camera = self.widget.renderer.GetActiveCamera()
        if not camera:
            return
        view_matrix = camera.GetModelViewTransformMatrix()
        inverse_view = vtk.vtkMatrix4x4()
        vtk.vtkMatrix4x4.Invert(view_matrix, inverse_view)
        values = [inverse_view.GetElement(row, col) for col in range(4) for row in range(4)]
        uniforms.SetUniformMatrix4x4("u_inverseViewMatrix", values)

    def release_all_textures(self, render_window, state):
        """state의 모든 텍스처 GPU 리소스를 해제한다."""
        for tex_list in (state.get('prob_textures', []), state.get('color_textures', []),
                         state.get('opacity_textures', [])):
            for tex in tex_list:
                tex.ReleaseGraphicsResources(render_window)
