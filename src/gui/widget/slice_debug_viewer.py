"""볼륨 Z-slice 위에 cluster 라벨(점 구름 또는 라벨 볼륨)을 겹쳐 보여주는 디버그 창.

점 구름 source는 slice 두께(±0.5 voxel) 안의 점을, 라벨 볼륨 source는 slice의 모든 voxel을 칠한다.
확률 볼륨 source(RF P(best))는 0~1 히트맵으로 칠하고, 마우스 위치의 값을 아래 줄에 보여준다.
"""
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QComboBox

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure

from src.gui.widget.renderer_widget import cluster_colors


class SliceDebugViewer(QDialog):
    """source(이름)별 데이터를 등록하고 Source / Cluster / Z 슬라이더로 골라 본다. 색은 메인 뷰와 같다."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Z-Slice Debug Viewer")
        self.resize(700, 820)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint)
        self.setModal(False)

        self.volume_data = None
        self.vol_min = 0.0
        self.vol_max = 1.0
        self.voxel_spacing = None
        self.model_to_world_inv = None
        self.point_sources = {}    # {name: (voxel 좌표 (N,3) float, labels (N,))}
        self.volume_sources = {}   # {name: (labels (X,Y,Z) int, cluster id 목록)} — 같은 이름이면 이쪽 우선
        self.prob_sources = {}     # {name: (X,Y,Z) float, -1 = 값 없음}
        self.best_cluster = None

        layout = QVBoxLayout(self)

        self.figure = Figure(figsize=(5, 5))
        self.figure.subplots_adjust(left=0.05, right=0.99, bottom=0.04, top=0.95)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self))   # 돋보기 = 사각형 확대, 십자 = 이동, 집 = 전체
        layout.addWidget(self.canvas)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Source"))
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        controls.addWidget(self.source_combo)

        controls.addWidget(QLabel("Cluster"))
        self.cluster_combo = QComboBox()
        self.cluster_combo.currentIndexChanged.connect(self._redraw)
        controls.addWidget(self.cluster_combo)
        layout.addLayout(controls)

        slider_row = QHBoxLayout()
        self.z_label = QLabel("Z: -")
        self.z_label.setMinimumWidth(80)
        slider_row.addWidget(self.z_label)
        self.z_slider = QSlider(Qt.Orientation.Horizontal)
        self.z_slider.valueChanged.connect(self._redraw)
        slider_row.addWidget(self.z_slider)
        layout.addLayout(slider_row)

        self.hover_label = QLabel("")
        layout.addWidget(self.hover_label)
        self.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def set_volume(self, volume_data, voxel_spacing, model_to_world=None):
        """배경 볼륨(밝기는 볼륨 전체 min/max 기준)과 world→voxel 변환 정보를 등록하고 Z를 가운데로 둔다."""
        self.volume_data = volume_data
        self.vol_min, self.vol_max = float(volume_data.min()), float(volume_data.max())
        self.voxel_spacing = np.asarray(voxel_spacing, dtype=np.float64)
        self.model_to_world_inv = (
            np.linalg.inv(np.asarray(model_to_world, dtype=np.float64))
            if model_to_world is not None else None)

        z_dim = volume_data.shape[2]
        self.z_slider.setMinimum(0)
        self.z_slider.setMaximum(max(0, z_dim - 1))
        self.z_slider.setValue(z_dim // 2)

    def set_best_cluster(self, best_cluster):
        """고정색으로 칠할 best cluster. source를 등록하기 전에 호출한다."""
        self.best_cluster = best_cluster

    def refresh_colors(self):
        """바뀐 cluster 색을 현재 선택(source / cluster / Z)을 유지한 채 다시 그린다."""
        self._redraw()

    def set_points(self, source_label, world_points, labels):
        """world 좌표 점 구름 source를 voxel 좌표로 변환해 등록한다. 비어 있으면 source를 제거한다."""
        if world_points is None or len(world_points) == 0:
            self.point_sources.pop(source_label, None)
        else:
            pts = np.asarray(world_points, dtype=np.float64)
            if self.model_to_world_inv is not None:
                homogeneous = np.hstack([pts, np.ones((len(pts), 1))])
                pts = (homogeneous @ self.model_to_world_inv.T)[:, :3]
            voxel_idx = pts / self.voxel_spacing
            self.point_sources[source_label] = (voxel_idx, np.asarray(labels))

        self._refresh_source_combo()

    def set_label_volume(self, source_label, labels):
        """(X,Y,Z) 라벨 볼륨(-1 = 라벨 없음) source를 등록한다. None이면 제거한다."""
        if labels is None:
            self.volume_sources.pop(source_label, None)
        else:
            labels = np.asarray(labels)
            cluster_ids = [int(c) for c in np.unique(labels) if c >= 0]
            self.volume_sources[source_label] = (labels, cluster_ids)
        self._refresh_source_combo()

    def set_prob_volume(self, source_label, prob):
        """(X,Y,Z) 확률 볼륨(-1 = 값 없음) source를 등록한다. None이면 제거한다."""
        if prob is None:
            self.prob_sources.pop(source_label, None)
        else:
            self.prob_sources[source_label] = np.asarray(prob)
        self._refresh_source_combo()

    def _source_names(self):
        names = list(self.point_sources.keys())
        names += [n for n in list(self.volume_sources) + list(self.prob_sources) if n not in names]
        return names

    def _refresh_source_combo(self):
        current = self.source_combo.currentText()
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        self.source_combo.addItems(self._source_names())
        idx = self.source_combo.findText(current)
        self.source_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.source_combo.blockSignals(False)
        self._refresh_cluster_combo()
        self._redraw()

    def _refresh_cluster_combo(self):
        name = self.source_combo.currentText()
        self.cluster_combo.blockSignals(True)
        self.cluster_combo.clear()
        self.cluster_combo.addItem("All", -1)
        if name in self.volume_sources:
            cluster_ids = self.volume_sources[name][1]
        elif name in self.point_sources:
            cluster_ids = sorted({int(x) for x in self.point_sources[name][1] if x >= 0})
        else:
            cluster_ids = []
        for c in cluster_ids:
            self.cluster_combo.addItem(f"Cluster {c}" + (" ★ (best)" if c == self.best_cluster else ""), c)
        self.cluster_combo.setCurrentIndex(0)
        self.cluster_combo.blockSignals(False)

    def _on_source_changed(self):
        self._refresh_cluster_combo()
        self._redraw()

    def _redraw(self):
        """현재 Z slice 배경 위에 선택한 source / cluster를 그린다."""
        if self.volume_data is None:
            return
        z = self.z_slider.value()
        z_dim = self.volume_data.shape[2]
        self.z_label.setText(f"Z: {z}/{max(0, z_dim - 1)}")

        bg = self.volume_data[:, :, z].T
        # 확대한 영역은 Z·source를 바꿔도 유지한다(같은 볼륨일 때만)
        keep = ((self.ax.get_xlim(), self.ax.get_ylim())
                if self.ax.images and self.ax.images[0].get_array().shape == bg.shape else None)
        self.ax.clear()
        self.ax.imshow(bg, cmap="gray", origin="lower", vmin=self.vol_min, vmax=self.vol_max)
        if keep is not None:
            self.ax.set_xlim(keep[0])
            self.ax.set_ylim(keep[1])

        name = self.source_combo.currentText()
        cluster_idx = self.cluster_combo.currentData()
        if name in self.prob_sources:
            sl = np.ma.masked_less(self.prob_sources[name][:, :, z].T.astype(np.float32), 0)
            self.ax.imshow(sl, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.6, origin="lower",
                           interpolation="nearest")
            self.ax.set_title(f"{name}  (blue 0 → red 1, {int((sl.filled(0) > 0.5).sum())} voxel > 0.5 @ this slice)")   # matplotlib 기본 글꼴은 한글 없음
            self.canvas.draw_idle()
            return
        if name in self.volume_sources:
            self._draw_label_slice(name, z, cluster_idx)
            self.canvas.draw_idle()
            return

        entry = self.point_sources.get(name)
        if entry is not None:
            voxel_idx, labels = entry
            near = np.abs(voxel_idx[:, 2] - z) < 0.5
            if cluster_idx is not None and cluster_idx >= 0:
                near &= labels == cluster_idx
            pts = voxel_idx[near]
            pt_labels = np.asarray(labels)[near]
            if len(pts):
                colors = cluster_colors(pt_labels, self.best_cluster) / 255.0
                self.ax.scatter(pts[:, 0], pts[:, 1], s=4, c=colors, alpha=0.7)
            self.ax.set_title(f"{name}  ({len(pts)} pt @ this slice)")
        else:
            self.ax.set_title(name or "(no source)")

        self.canvas.draw_idle()

    def _on_hover(self, event):
        """마우스 아래 voxel의 (x, y, z), 볼륨 값, 등록된 확률 볼륨 값을 보여준다."""
        if self.volume_data is None or event.inaxes is not self.ax or event.xdata is None:
            return
        x, y, z = int(round(event.xdata)), int(round(event.ydata)), self.z_slider.value()
        X, Y = self.volume_data.shape[:2]
        if not (0 <= x < X and 0 <= y < Y):
            return
        text = f"voxel ({x}, {y}, {z})  value {self.volume_data[x, y, z]}"
        for name, prob in self.prob_sources.items():
            p = float(prob[x, y, z])
            text += f"   {name} " + ("판정 불가" if p < 0 else f"{p:.3f}")
        self.hover_label.setText(text)

    def _draw_label_slice(self, name, z, cluster_idx, alpha=0.55):
        """라벨 볼륨의 z slice 전체를 cluster 색(alpha)으로 칠한다."""
        labels, _ = self.volume_sources[name]
        sl = labels[:, :, z].T
        shown = sl >= 0
        if cluster_idx is not None and cluster_idx >= 0:
            shown &= sl == cluster_idx

        rgba = np.zeros(sl.shape + (4,), dtype=np.float32)
        rgba[shown, :3] = cluster_colors(sl[shown], self.best_cluster) / 255.0
        rgba[shown, 3] = alpha
        self.ax.imshow(rgba, origin="lower", interpolation="nearest")
        self.ax.set_title(f"{name}  ({int(shown.sum())} voxel @ this slice)")
