"""TF 곡선 편집 위젯: 볼륨 히스토그램 배경 위에서 node를 드래그/추가/삭제/색 변경한다.

node 형식: [intensity(0~1), r, g, b, alpha]. 가로축은 [view_min, view_max] 구간으로 확대해 그릴 수 있다.
조작: 좌클릭 드래그 = 이동, 빈 곳 더블클릭 = 추가, node 더블클릭 = 삭제, 우클릭 = 색 변경.
"""

import numpy as np
from scipy import ndimage
from PyQt6.QtWidgets import QWidget, QColorDialog
from PyQt6.QtCore import Qt, pyqtSignal, QPoint
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QLinearGradient, QPainterPath

from src.core.wysiwyg_tf_editor import WysiwygTFEditor

DEFAULT_NODES = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
    [0.3, 1.0, 0.5, 0.0, 0.1],
    [0.7, 1.0, 1.0, 1.0, 0.8],
    [1.0, 1.0, 1.0, 1.0, 1.0]
]


class TransferFunctionWidget(QWidget):
    """node가 바뀔 때마다 tf_changed를 보낸다(set_nodes 등 코드 호출은 제외)."""

    tf_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setFixedHeight(190)
        self.setMouseTracking(True)

        self.default_nodes = DEFAULT_NODES
        self.nodes = [node[:] for node in DEFAULT_NODES]
        self.tf_editor = WysiwygTFEditor(lut_size=256)

        self.selected_node = -1
        self.dragging = False
        self.node_radius = 8

        self.histogram = None
        self.histogram_max = 1.0

        self.tf_area_top = 0
        self.tf_area_height = self.height() - 50   # 아래 50px는 색상 바 영역

        self.highlight_range = None   # ROI intensity 구간 (low, high)
        self.view_min = 0.0
        self.view_max = 1.0

    def set_volume_data(self, volume_data):
        """256-bin log 히스토그램(가우시안 σ=0.5 smoothing)을 계산해 배경으로 쓴다."""
        if volume_data is None:
            return
        try:
            hist, _ = np.histogram(volume_data.ravel(), bins=256,
                                   range=(volume_data.min(), volume_data.max()))
            self.histogram = ndimage.gaussian_filter1d(np.log1p(hist.astype(np.float32)), sigma=0.5)
            peak = float(np.max(self.histogram))
            self.histogram_max = peak if peak > 0 else 1.0
            self.update()
        except Exception as e:
            print(f"❌ 히스토그램 계산 실패: {e}")
            self.histogram = None

    # ------------------------------------------------------------------
    # 그리기
    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        self.draw_tf_area_histogram(painter)
        self.draw_enhanced_grid(painter)
        self.draw_roi_highlight(painter)
        self.draw_enhanced_tf_curve(painter)
        self.draw_nodes(painter)
        self.draw_enhanced_color_bar(painter)

    def draw_tf_area_histogram(self, painter):
        painter.fillRect(0, self.tf_area_top, self.width(), self.tf_area_height, QColor(50, 50, 50))
        if self.histogram is None:
            return

        max_height = self.tf_area_height
        n_bins = len(self.histogram)
        path = QPainterPath()
        bottom = self.tf_area_top + max_height
        last_x = 0
        for i, count in enumerate(self.histogram):
            x = self.intensity_to_x((i + 0.5) / n_bins)
            y = bottom - (count / self.histogram_max) * max_height
            if i == 0:
                path.moveTo(x, bottom)
            path.lineTo(x, y)
            last_x = x
        path.lineTo(last_x, bottom)
        path.closeSubpath()

        painter.fillPath(path, QBrush(QColor(150, 50, 50, 100)))
        painter.setPen(QPen(QColor(200, 50, 50, 180), 1))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(120, 120, 120), 1))
        painter.drawRect(0, self.tf_area_top, self.width() - 1, self.tf_area_height)

    def draw_enhanced_grid(self, painter):
        painter.setPen(QPen(QColor(80, 80, 80), 1, Qt.PenStyle.DotLine))
        for i in range(11):
            x = int((i / 10.0) * self.width())
            painter.drawLine(x, self.tf_area_top, x, self.height() - 40)
        for i in range(6):
            y = int(self.tf_area_top + (i / 5.0) * self.tf_area_height)
            painter.drawLine(0, y, self.width(), y)

    def draw_enhanced_tf_curve(self, painter):
        """opacity 곡선(흰 선)."""
        if len(self.nodes) < 2:
            return
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        points = [QPoint(x, int((1 - self.interpolate_alpha(self.x_to_intensity(x))) * self.tf_area_height
                                + self.tf_area_top))
                  for x in range(self.width())]
        if len(points) > 1:
            painter.drawPolyline(points)

    def draw_enhanced_color_bar(self, painter):
        """하단 색상 바: node 색을 alpha로 어두운 배경과 섞은 그라디언트 + 체커보드."""
        bar_height = 25
        bar_y = self.height() - bar_height - 5
        gradient = QLinearGradient(0, bar_y, self.width(), bar_y)
        span = max(self.view_max - self.view_min, 1e-6)
        bg = 50

        for node in sorted(self.nodes, key=lambda n: n[0]):
            pos = max(0.0, min(1.0, (node[0] - self.view_min) / span))
            r, g, b, alpha = (max(0, min(1, v)) for v in node[1:5])
            gradient.setColorAt(pos, QColor(int(r * 255 * alpha + bg * (1 - alpha)),
                                            int(g * 255 * alpha + bg * (1 - alpha)),
                                            int(b * 255 * alpha + bg * (1 - alpha))))
        painter.fillRect(0, bar_y, self.width(), bar_height, QBrush(gradient))

        checker_size = 8
        for x in range(0, self.width(), checker_size * 2):
            for y in range(bar_y, bar_y + bar_height, checker_size):
                if ((x // checker_size) + (y // checker_size)) % 2 == 0:
                    painter.fillRect(x, y, checker_size, checker_size, QColor(255, 255, 255, 15))

    def draw_nodes(self, painter):
        for i, node in enumerate(self.nodes):
            x = int(self.intensity_to_x(node[0]))
            y = int((1 - node[4]) * self.tf_area_height + self.tf_area_top)
            color = QColor(int(node[1] * 255), int(node[2] * 255), int(node[3] * 255))
            if i == self.selected_node:
                painter.setPen(QPen(QColor(255, 255, 0), 3))
            else:
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(x - self.node_radius, y - self.node_radius,
                                2 * self.node_radius, 2 * self.node_radius)

    def draw_roi_highlight(self, painter):
        """highlight_range 구간을 연노랑 반투명 띠로 표시한다."""
        if self.highlight_range is None:
            return
        low, high = self.highlight_range
        x0 = int(self.intensity_to_x(low))
        x1 = int(self.intensity_to_x(high))
        painter.save()
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 230, 80, 70))
        painter.drawRect(x0, self.tf_area_top, max(1, x1 - x0), self.tf_area_height)
        painter.restore()

    # ------------------------------------------------------------------
    # node 보간
    # ------------------------------------------------------------------
    def interpolate_node(self, intensity):
        """Returns: intensity 위치의 보간 node [x, r, g, b, a]"""
        if len(self.nodes) < 2:
            return [intensity, 0.0, 0.0, 0.0, 0.0]
        return WysiwygTFEditor.interpolate_node_at(self.nodes, intensity)

    def interpolate_alpha(self, intensity):
        return self.interpolate_node(intensity)[4]

    def interpolate_color(self, intensity):
        return self.interpolate_node(intensity)[1:4]

    # ------------------------------------------------------------------
    # 마우스 편집
    # ------------------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_node = self.find_node_at_pos(event.position())
            self.dragging = self.selected_node >= 0
            self.update()
        elif event.button() == Qt.MouseButton.RightButton:
            node_idx = self.find_node_at_pos(event.position())
            if node_idx >= 0:
                self.change_node_color(node_idx)

    def mouseMoveEvent(self, event):
        """선택 node를 이동한다. intensity는 양옆 node와 0.01 이상 간격을 유지한다."""
        if not (self.dragging and self.selected_node >= 0):
            return
        x = max(0, min(event.position().x(), self.width() - 1))
        y = max(self.tf_area_top, min(event.position().y(), self.height() - 50))
        intensity = self.x_to_intensity(x)
        alpha = 1 - (y - self.tf_area_top) / self.tf_area_height

        order = sorted(range(len(self.nodes)), key=lambda i: self.nodes[i][0])
        pos = order.index(self.selected_node)
        min_intensity = self.nodes[order[pos - 1]][0] + 0.01 if pos > 0 else 0.0
        max_intensity = self.nodes[order[pos + 1]][0] - 0.01 if pos < len(order) - 1 else 1.0
        intensity = max(max(0.0, min_intensity), min(min(1.0, max_intensity), intensity))

        self.nodes[self.selected_node][0] = intensity
        self.nodes[self.selected_node][4] = max(0, min(1, alpha))
        self.update()
        self.tf_changed.emit()

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def mouseDoubleClickEvent(self, event):
        """node 위 더블클릭 = 삭제(최소 2개 유지), 빈 곳 더블클릭 = 보간 색으로 node 추가."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        node_idx = self.find_node_at_pos(event.position())
        if node_idx >= 0:
            if len(self.nodes) > 2:
                del self.nodes[node_idx]
                self.selected_node = -1
                self.update()
                self.tf_changed.emit()
            else:
                print("⚠️ 최소 2개 노드는 유지해야 합니다")
        elif self.tf_area_top <= event.position().y() <= self.tf_area_top + self.tf_area_height:
            intensity = self.x_to_intensity(event.position().x())
            alpha = 1 - (event.position().y() - self.tf_area_top) / self.tf_area_height
            r, g, b = self.interpolate_color(intensity)
            self.nodes.append([intensity, r, g, b, max(0, min(1, alpha))])
            self.nodes.sort(key=lambda n: n[0])
            self.update()
            self.tf_changed.emit()

    def change_node_color(self, node_idx):
        current_color = QColor(*(int(v * 255) for v in self.nodes[node_idx][1:4]))
        color = QColorDialog.getColor(current_color, self, f"노드 {node_idx} 색상 선택")
        if color.isValid():
            self.nodes[node_idx][1:4] = [color.red() / 255.0, color.green() / 255.0, color.blue() / 255.0]
            self.update()
            self.tf_changed.emit()

    def find_node_at_pos(self, pos):
        """Returns: pos(위젯 좌표)에 있는 node 인덱스, 없으면 -1"""
        for i, node in enumerate(self.nodes):
            x = self.intensity_to_x(node[0])
            y = (1 - node[4]) * self.tf_area_height + self.tf_area_top
            if (pos.x() - x) ** 2 + (pos.y() - y) ** 2 <= self.node_radius ** 2:
                return i
        return -1

    # ------------------------------------------------------------------
    # node 설정 / 조회
    # ------------------------------------------------------------------
    def reset_to_default(self):
        self.nodes = [node[:] for node in DEFAULT_NODES]
        self.selected_node = -1
        self.update()
        self.tf_changed.emit()

    def get_nodes(self):
        """Returns: node 리스트 복사본"""
        return [node[:] for node in self.nodes]

    def set_nodes(self, nodes):
        """node를 교체한다(tf_changed 없음). 2개 미만이면 기본 TF로 대체."""
        if not nodes:
            return
        self.nodes = [node[:] for node in nodes]
        if len(self.nodes) < 2:
            self.nodes = [node[:] for node in DEFAULT_NODES]
        self.selected_node = -1
        self.update()

    def set_transfer_function_from_array(self, tf_array):
        self.set_nodes(tf_array)

    def get_opacity_lut_for_nodes(self, nodes):
        """Returns: nodes의 (256,) opacity LUT, node가 2개 미만이면 0 배열"""
        if nodes is None or len(nodes) < 2:
            return np.zeros(256, dtype=np.float32)
        _, _, opacity_lut = self.tf_editor.sample_nodes_to_lut(nodes, num_bins=256)
        return opacity_lut

    # ------------------------------------------------------------------
    # 가로축 범위 / ROI 표시
    # ------------------------------------------------------------------
    def set_view_range(self, low=None, high=None):
        """가로축 표시 구간을 [low, high]로 확대한다. 인자가 없으면 [0, 1]."""
        if low is None or high is None:
            self.view_min, self.view_max = 0.0, 1.0
        else:
            low = float(np.clip(low, 0.0, 1.0))
            high = float(np.clip(high, 0.0, 1.0))
            if high - low < 1e-4:
                low, high = 0.0, 1.0
            self.view_min, self.view_max = min(low, high), max(low, high)
        self.update()

    def x_to_intensity(self, x):
        span = max(self.view_max - self.view_min, 1e-6)
        return self.view_min + (x / max(self.width() - 1, 1)) * span

    def intensity_to_x(self, intensity):
        span = max(self.view_max - self.view_min, 1e-6)
        return (intensity - self.view_min) / span * (self.width() - 1)

    def set_highlight_range(self, value):
        """value: (low, high) in [0, 1] 또는 None"""
        if value is None:
            self.highlight_range = None
        else:
            low, high = (max(0.0, min(1.0, float(v))) for v in value)
            self.highlight_range = (min(low, high), max(low, high))
        self.update()

    def clear_highlight_range(self):
        self.highlight_range = None
        self.update()
