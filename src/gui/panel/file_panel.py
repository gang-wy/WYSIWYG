"""볼륨 데이터와 INR feature volume을 불러오는 패널."""

import os
import numpy as np
import nibabel as nib
from PyQt6.QtWidgets import QPushButton, QLabel, QFileDialog, QDialog, QMessageBox
from PyQt6.QtCore import pyqtSignal

from src.gui.panel.base_panel import BasePanel
from src.gui.dialogs.raw_data_dialog import RawDataDialog
from src.gui.data.volume_loader import VolumeLoader


class FilePanel(BasePanel):
    """로드 결과는 volume_data / voxel_spacing / volume_name / inr_feature_volume에 저장하고 시그널로 알린다."""

    volume_loaded = pyqtSignal(object)        # (X, Y, Z) 정규화된 볼륨
    inr_features_loaded = pyqtSignal(object)  # (X, Y, Z, C) INR feature volume

    def __init__(self):
        super().__init__("Load Data", collapsible=False)
        self.volume_loader = VolumeLoader()
        self.volume_data = None
        self.voxel_spacing = (1.0, 1.0, 1.0)
        self.volume_name = None
        self.inr_feature_volume = None
        # Stage A 덤프에 기록하는 경로(분석 스크립트가 같은 볼륨/INR을 다시 읽는 데 씀)
        self.volume_path = None
        self.raw_params = None
        self.inr_path = None

    def setup_content(self):
        self.load_volume_btn = QPushButton("📂 Load Volume Data")
        self.load_volume_btn.setMinimumHeight(40)
        self.load_volume_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover { background-color: #1976D2; }
        """)
        self.load_volume_btn.clicked.connect(self.load_volume_data)
        self.content_layout.addWidget(self.load_volume_btn)

        self.info_label = QLabel("No volume loaded")
        self.info_label.setStyleSheet("color: #888; padding: 5px;")
        self.info_label.setWordWrap(True)
        self.content_layout.addWidget(self.info_label)

        self.load_inr_btn = QPushButton("🧠 Load INR Features (.nii)")
        self.load_inr_btn.setMinimumHeight(34)
        self.load_inr_btn.setStyleSheet("""
            QPushButton {
                background-color: #9C27B0;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #7B1FA2; }
            QPushButton:disabled { background-color: #666; color: #999; }
        """)
        self.load_inr_btn.setEnabled(False)   # 볼륨 로드 후 활성화
        self.load_inr_btn.clicked.connect(self.load_inr_features)
        self.content_layout.addWidget(self.load_inr_btn)

        self.inr_info_label = QLabel("")
        self.inr_info_label.setStyleSheet("color: #888; padding: 2px; font-size: 11px;")
        self.inr_info_label.setWordWrap(True)
        self.content_layout.addWidget(self.inr_info_label)

    def load_volume_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Volume Data", "./resources/Volume_Data",
            "All Supported Files (*.nii *.nii.gz *.npy *.raw *.dat);;"
            "Volume Files (*.nii *.nii.gz *.npy);;"
            "Raw Files (*.raw *.dat);;"
            "All Files (*)"
        )
        if file_path:
            self.load_volume_from_path(file_path)

    def load_volume_from_path(self, file_path, raw_params=None):
        """볼륨을 로드하고 volume_loaded를 보낸다. raw 파일이고 raw_params가 없으면 파라미터 입력창을 띄운다."""
        try:
            if file_path.endswith(('.raw', '.dat')) and raw_params is None:
                dialog = RawDataDialog(self)
                if dialog.exec() != QDialog.DialogCode.Accepted:
                    self.emit_status("Raw data load cancelled")
                    return
                raw_params = dialog.get_parameters()

            self.volume_data, self.voxel_spacing = self.volume_loader.load(file_path, raw_params)
            if self.volume_data is None:
                raise ValueError("볼륨 데이터 처리 실패")

            filename = os.path.basename(file_path)
            self.volume_name = filename
            self.volume_path, self.raw_params, self.inr_path = file_path, raw_params, None
            shape, spacing = self.volume_data.shape, self.voxel_spacing
            self.info_label.setText(
                f"📊 {filename}\n"
                f"Shape: {shape[0]} × {shape[1]} × {shape[2]}\n"
                f"Spacing: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f}"
            )
            self.info_label.setStyleSheet("color: #4CAF50; padding: 5px;")
            self.load_inr_btn.setEnabled(True)

            self.inr_feature_volume = None
            self.inr_info_label.setText("")

            self.volume_loaded.emit(self.volume_data)
            self.emit_status(f"Volume loaded: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.emit_status("Data load failed")
            self.info_label.setText(f"❌ Load failed: {str(e)}")
            self.info_label.setStyleSheet("color: #f44336; padding: 5px;")

    def load_inr_features(self):
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "먼저 볼륨 데이터를 로드해주세요.")
            return
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load INR Feature Volume", "./resources/Volume_Feature",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        if file_path:
            self.load_inr_features_from_path(file_path)

    def load_inr_features_from_path(self, file_path):
        """INR feature .nii를 (X, Y, Z, C) 그대로(추가 정규화 없이) 로드하고 inr_features_loaded를 보낸다.

        4차원이 아니거나 앞 3축이 볼륨 shape와 다르면 오류창을 띄우고 로드하지 않는다.
        """
        try:
            inr_data = np.asarray(nib.load(file_path).get_fdata(), dtype=np.float32)

            if inr_data.ndim != 4:
                QMessageBox.critical(
                    self, "Error",
                    f"INR feature volume은 (X, Y, Z, C) 4차원이어야 합니다. 현재 shape: {inr_data.shape}")
                return
            if inr_data.shape[:3] != self.volume_data.shape:
                QMessageBox.critical(
                    self, "Error",
                    f"INR feature shape {inr_data.shape[:3]} does not match "
                    f"volume shape {self.volume_data.shape}!")
                return

            self.inr_feature_volume = inr_data
            self.inr_path = file_path
            filename = os.path.basename(file_path)
            num_channels = inr_data.shape[-1]
            self.inr_info_label.setText(f"🧠 {filename} | channels: {num_channels}")
            self.inr_info_label.setStyleSheet("color: #9C27B0; padding: 2px; font-size: 11px;")

            self.inr_features_loaded.emit(self.inr_feature_volume)
            self.emit_status(f"INR features loaded: {filename} ({num_channels} channels), shape {inr_data.shape}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Failed to load INR features: {str(e)}")
            self.inr_info_label.setText("❌ INR load failed")
            self.inr_info_label.setStyleSheet("color: #f44336; padding: 2px; font-size: 11px;")
