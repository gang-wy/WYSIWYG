"""축별 min/max 클리핑 평면 6개를 관리하고 볼륨 mapper에 적용한다."""
import vtk

FAR_AWAY = 10000   # 볼륨을 자르지 않는 초기 평면 위치


class VolumeClippingManager:
    """평면 좌표는 world 단위. 각 축의 min 평면은 작은 쪽을, max 평면은 큰 쪽을 잘라낸다."""

    def __init__(self, parent_renderer):
        self.parent = parent_renderer
        self.clipping_planes = vtk.vtkPlaneCollection()
        self.clipping_enabled = False
        self.planes = {name: vtk.vtkPlane() for name in
                       ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max')}
        self.setup_clipping_planes()

    def setup_clipping_planes(self):
        """모든 평면을 볼륨 바깥 먼 곳에 두고 평면 컬렉션을 다시 만든다."""
        for i, axis in enumerate('xyz'):
            normal = [0, 0, 0]
            normal[i] = 1
            origin = [0, 0, 0]
            origin[i] = -FAR_AWAY
            self.planes[f'{axis}_min'].SetOrigin(*origin)
            self.planes[f'{axis}_min'].SetNormal(*normal)
            origin[i] = FAR_AWAY
            normal[i] = -1
            self.planes[f'{axis}_max'].SetOrigin(*origin)
            self.planes[f'{axis}_max'].SetNormal(*normal)

        self.clipping_planes.RemoveAllItems()
        for plane in self.planes.values():
            self.clipping_planes.AddItem(plane)

    def get_current_volume(self):
        return self.parent.standard_volume

    def render(self):
        self.parent.vtk_widget.GetRenderWindow().Render()

    def update_clipping_target(self):
        """클리핑 on/off 상태에 맞춰 현재 볼륨 mapper에 평면을 붙이거나 뗀다."""
        volume = self.get_current_volume()
        if volume and volume.GetMapper():
            if self.clipping_enabled:
                volume.GetMapper().SetClippingPlanes(self.clipping_planes)
            else:
                volume.GetMapper().RemoveAllClippingPlanes()
            self.render()

    def set_clipping_range(self, axis_index, min_pos, max_pos):
        """axis_index(0=X, 1=Y, 2=Z) 축의 min/max 평면을 world 좌표 min_pos/max_pos로 옮긴다."""
        axis = 'xyz'[axis_index]
        for name, pos in ((f'{axis}_min', min_pos), (f'{axis}_max', max_pos)):
            origin = [0, 0, 0]
            origin[axis_index] = pos
            self.planes[name].SetOrigin(*origin)
        if self.clipping_enabled:
            self.update_clipping_target()

    def enable_clipping(self, enabled):
        """켜면 평면을 볼륨 경계로 리셋한 새 컬렉션을 mapper에 붙이고, 끄면 모두 뗀다."""
        if self.clipping_enabled == enabled:
            return
        self.clipping_enabled = enabled

        volume = self.get_current_volume()
        if not volume or not volume.GetMapper():
            print("⚠️ No volume/mapper available for clipping")
            return

        mapper = volume.GetMapper()
        mapper.RemoveAllClippingPlanes()
        if enabled:
            self.reset_clipping()
            self.clipping_planes = vtk.vtkPlaneCollection()
            for plane in self.planes.values():
                self.clipping_planes.AddItem(plane)
            mapper.SetClippingPlanes(self.clipping_planes)
            mapper.Modified()
        self.render()

    def reset_clipping(self):
        """평면을 볼륨 bounds(±0.1 여유)로 되돌린다."""
        self.setup_clipping_planes()
        epsilon = 0.1

        volume = self.get_current_volume()
        if volume and volume.GetMapper() and volume.GetMapper().GetInput():
            bounds = volume.GetMapper().GetInput().GetBounds()
        elif self.parent.volume_data is not None:
            shape = self.parent.volume_data.shape
            spacing = self.parent.voxel_spacing or (1.0, 1.0, 1.0)
            bounds = (0, shape[0] * spacing[0], 0, shape[1] * spacing[1], 0, shape[2] * spacing[2])
        else:
            return

        for axis_index in range(3):
            self.set_clipping_range(axis_index, bounds[2 * axis_index] - epsilon,
                                    bounds[2 * axis_index + 1] + epsilon)
