"""SAM 패널: 포인트 입력 모드, 포인트 저장/로드, SAM 마스크 표시, Check SAM / Run Clustering 버튼."""

import json
import os
from datetime import datetime
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton,
                             QButtonGroup, QFileDialog, QSizePolicy, QMessageBox)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage
from src.gui.panel.base_panel import BasePanel


class OptimizationPanel(BasePanel):
    """picked_points: [{'pos': (vtk_x, vtk_y), 'type': 'positive' | 'negative'}, ...]"""

    set_mode_changed = pyqtSignal(bool)
    point_type_changed = pyqtSignal(str)
    check_sam = pyqtSignal()
    points_updated = pyqtSignal(list)
    run_clustering_requested = pyqtSignal()

    def __init__(self):
        self.current_point_type = "positive"
        self.picked_points = []
        super().__init__("Optimization & SAM", collapsible=False)

    def setup_content(self):
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        header_layout = QHBoxLayout()
        self.title_label = QLabel("🎯 SAM & Optimization")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.content_layout.addLayout(header_layout)

        self.image_display = QLabel("No Image Loaded")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setStyleSheet("background-color: #000000; border: 1px solid #444; border-radius: 4px;")
        self.image_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_layout.addWidget(self.image_display, stretch=1)

        base_btn_style = "QPushButton { height: 34px; font-weight: bold; padding: 0 10px; }"
        toggle_btn_style = base_btn_style + """
            QPushButton:checked#set_btn {
                background-color: #2ecc71;
                color: white;
                border: 1px solid #27ae60;
            }
            QPushButton:checked#pos_btn { background-color: #2196F3; color: white; }
            QPushButton:checked#neg_btn { background-color: #FF5252; color: white; }
            QPushButton:checked#line_btn { background-color: #F9A825; color: white; }
        """

        # 포인트 모드 / 타입 / 저장·로드
        point_group, point_layout = self.create_group_box("", "horizontal")
        point_layout.setContentsMargins(10, 5, 10, 5)
        point_layout.setSpacing(5)

        self.set_btn = QPushButton("Enable Set Mode")
        self.set_btn.setObjectName("set_btn")
        self.set_btn.setCheckable(True)
        self.set_btn.setStyleSheet(toggle_btn_style)

        self.pos_btn = QPushButton("Positive")
        self.pos_btn.setObjectName("pos_btn")
        self.pos_btn.setCheckable(True)
        self.pos_btn.setChecked(True)

        self.neg_btn = QPushButton("Negative")
        self.neg_btn.setObjectName("neg_btn")
        self.neg_btn.setCheckable(True)

        # Line: 드래그로 스트로크를 그리면 Unity처럼 호 길이 기준 positive 점 4개로 바꾼다(그을 때마다 교체)
        self.line_btn = QPushButton("Line")
        self.line_btn.setObjectName("line_btn")
        self.line_btn.setCheckable(True)
        self.line_btn.setToolTip("드래그로 선을 그린다. 놓으면 선을 따라 균등한 positive 점 4개로 바뀐다 "
                                 "(Unity 스트로크와 같은 규칙). 새로 그으면 이전 점은 지워진다.")

        for btn in (self.pos_btn, self.neg_btn, self.line_btn):
            btn.setEnabled(False)
            btn.setVisible(False)
            btn.setStyleSheet(toggle_btn_style)

        self.save_btn = QPushButton("💾 Save Pts")
        self.load_btn = QPushButton("📥 Load Pts")
        self.save_btn.setStyleSheet(base_btn_style)
        self.load_btn.setStyleSheet(base_btn_style)

        self.type_group = QButtonGroup(self)
        self.type_group.addButton(self.pos_btn)
        self.type_group.addButton(self.neg_btn)
        self.type_group.addButton(self.line_btn)

        for btn in [self.set_btn, self.pos_btn, self.neg_btn, self.line_btn, self.save_btn, self.load_btn]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            point_layout.addWidget(btn)
        self.content_layout.addWidget(point_group)

        # Check SAM → Run Clustering
        action_group, action_layout = self.create_group_box("", "horizontal")
        action_layout.setContentsMargins(10, 5, 10, 5)
        action_layout.setSpacing(8)

        self.check_btn = QPushButton("🔍 Check SAM")
        self.check_btn.setStyleSheet(base_btn_style)

        self.clustering_btn = QPushButton("🧩 Run Clustering")
        self.clustering_btn.setStyleSheet(
            "QPushButton {background-color: #2E7D32; color: white; font-weight: bold; border-radius: 4px; "
            "height: 34px; padding: 0 10px; } QPushButton:disabled { background-color: #555; color: #aaa; }")
        self.clustering_btn.setEnabled(False)
        self.clustering_btn.setToolTip("먼저 'Check SAM'으로 마스크를 생성해야 활성화됩니다.")

        for btn in [self.check_btn, self.clustering_btn]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            action_layout.addWidget(btn)
        self.content_layout.addWidget(action_group)

        self.set_btn.toggled.connect(self.on_set_mode_toggled)
        self.pos_btn.clicked.connect(lambda: self.change_point_type("positive"))
        self.neg_btn.clicked.connect(lambda: self.change_point_type("negative"))
        self.line_btn.clicked.connect(lambda: self.change_point_type("line"))
        self.save_btn.clicked.connect(self.save_points)
        self.load_btn.clicked.connect(self.load_points)
        self.check_btn.clicked.connect(lambda: self.check_sam.emit())
        self.clustering_btn.clicked.connect(lambda: self.run_clustering_requested.emit())

    def set_clustering_available(self, available: bool):
        self.clustering_btn.setEnabled(available)

    def on_set_mode_toggled(self, checked):
        self.set_btn.setText("Set Mode: ON" if checked else "Enable Set Mode")
        for btn in (self.pos_btn, self.neg_btn, self.line_btn):
            btn.setVisible(checked)
            btn.setEnabled(checked)
        self.set_mode_changed.emit(checked)

    def change_point_type(self, p_type):
        """p_type: "positive" | "negative" | "line" """
        self.current_point_type = p_type
        self.pos_btn.setChecked(p_type == "positive")
        self.neg_btn.setChecked(p_type == "negative")
        self.line_btn.setChecked(p_type == "line")
        self.point_type_changed.emit(p_type)

    def save_points(self):
        """picked_points를 JSON으로 저장한다(./resources/Points 기본 경로)."""
        if not self.picked_points:
            QMessageBox.warning(self, "Warning", "저장할 포인트가 없습니다.")
            return
        default_path = "./resources/Points"
        os.makedirs(default_path, exist_ok=True)
        default_file = os.path.join(default_path, f"points_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Points", default_file, "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.picked_points, f, indent=4)
            self.emit_status(f"Saved: {os.path.basename(file_path)}")

    def load_points(self):
        default_path = "./resources/Points"
        os.makedirs(default_path, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Points", default_path, "JSON Files (*.json)")
        if file_path:
            self.load_points_from_path(file_path)

    def load_points_from_path(self, file_path):
        """포인트 JSON을 읽어 picked_points를 교체하고 points_updated를 보낸다."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                self.picked_points = data
                self.points_updated.emit(self.picked_points)
                self.emit_status(f"Loaded {len(self.picked_points)} points.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"로드 실패: {e}")

    def set_image_by_path(self, file_path):
        """이미지 파일(SAM 마스크)을 표시 영역 크기에 맞춰 보여준다."""
        if not os.path.exists(file_path):
            self.image_display.setText("File Not Found")
            return
        pixmap = QPixmap.fromImage(QImage(file_path))
        self.image_display.setPixmap(pixmap.scaled(
            self.image_display.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
