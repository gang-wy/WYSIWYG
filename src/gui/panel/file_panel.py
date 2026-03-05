"""
간결화된 파일 로드 패널 - 볼륨 로드 + Segmentation 로드
"""

import os
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal, Qt

from src.gui.panel.base_panel import BasePanel
from src.gui.dialogs.raw_data_dialog import RawDataDialog
from src.gui.data.volume_loader import VolumeLoader


class FilePanel(BasePanel):
    """간결화된 파일 로드 패널"""
    
    # 시그널 정의
    volume_loaded = pyqtSignal(object)  # 볼륨 데이터 로드 완료
    segmentation_loaded = pyqtSignal(object)  # [NEW] Segmentation 데이터 로드 완료
    
    def __init__(self):
        super().__init__("Load Data", collapsible=False)
        self.volume_loader = VolumeLoader() 
        self.volume_data = None
        self.voxel_spacing = (1.0, 1.0, 1.0)
        self.volume_name = None
        self.segmentation_data = None  # [NEW]
        
    def setup_content(self):
        """내용 설정"""
        # 로드 버튼
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
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        self.load_volume_btn.clicked.connect(self.load_volume_data)
        self.content_layout.addWidget(self.load_volume_btn)
        
        # 볼륨 정보 라벨
        self.info_label = QLabel("No volume loaded")
        self.info_label.setStyleSheet("color: #888; padding: 5px;")
        self.info_label.setWordWrap(True)
        self.content_layout.addWidget(self.info_label)
        
        # [NEW] Segmentation 로드 버튼
        self.load_seg_btn = QPushButton("🧩 Load Segmentation (.nii)")
        self.load_seg_btn.setMinimumHeight(34)
        self.load_seg_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                font-weight: bold;
                border-radius: 5px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        self.load_seg_btn.setEnabled(False)  # 볼륨 로드 전에는 비활성화
        self.load_seg_btn.clicked.connect(self.load_segmentation_data)
        self.content_layout.addWidget(self.load_seg_btn)
        
        # [NEW] Segmentation 정보 라벨
        self.seg_info_label = QLabel("")
        self.seg_info_label.setStyleSheet("color: #888; padding: 2px; font-size: 11px;")
        self.seg_info_label.setWordWrap(True)
        self.content_layout.addWidget(self.seg_info_label)
    
    def load_volume_data(self):
        """볼륨 데이터 로드"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Volume Data", "./resources/Volume_Data", 
            "All Supported Files (*.nii *.nii.gz *.npy *.raw *.dat);;"
            "Volume Files (*.nii *.nii.gz *.npy);;"
            "Raw Files (*.raw *.dat);;"
            "All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            raw_params = None
            
            # Raw 파일인 경우 다이얼로그로 파라미터 받기
            if file_path.endswith(('.raw', '.dat')):
                dialog = RawDataDialog(self)
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    raw_params = dialog.get_parameters()
                else:
                    self.emit_status("Raw data load cancelled")
                    return
            
            print(f"[Debug] Loading {file_path}")
            print(f"[Debug] Raw Params: {raw_params}")

            # VolumeLoader로 로드
            self.volume_data, self.voxel_spacing = self.volume_loader.load(
                file_path, raw_params
            )
            
            if self.volume_data is None:
                raise ValueError("볼륨 데이터 처리 실패")
            
            # 정보 업데이트
            filename = os.path.basename(file_path)
            self.volume_name = filename
            shape = self.volume_data.shape
            spacing = self.voxel_spacing
            self.info_label.setText(
                f"📊 {filename}\n"
                f"Shape: {shape[0]} × {shape[1]} × {shape[2]}\n"
                f"Spacing: {spacing[0]:.2f} × {spacing[1]:.2f} × {spacing[2]:.2f}"
            )
            self.info_label.setStyleSheet("color: #4CAF50; padding: 5px;")
            
            # [NEW] Segmentation 버튼 활성화
            self.load_seg_btn.setEnabled(True)
            
            # 기존 segmentation 초기화
            self.segmentation_data = None
            self.seg_info_label.setText("")
            
            # 시그널 발송
            self.volume_loaded.emit(self.volume_data)
            self.emit_status(f"Volume loaded: {filename}")
            
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.emit_status("Data load failed")
            self.info_label.setText(f"❌ Load failed: {str(e)}")
            self.info_label.setStyleSheet("color: #f44336; padding: 5px;")

    def load_segmentation_data(self):
        """[NEW] Segmentation NIfTI 파일 로드"""
        if self.volume_data is None:
            QMessageBox.warning(self, "Warning", "먼저 볼륨 데이터를 로드해주세요.")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Segmentation Mask", "./resources/Volume_Data",
            "NIfTI Files (*.nii *.nii.gz);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            import nibabel as nib
            
            nii_img = nib.load(file_path)
            seg_data = np.array(nii_img.get_fdata(), order='F')
            
            # Shape 검증
            if seg_data.shape != self.volume_data.shape:
                QMessageBox.critical(
                    self, "Error",
                    f"Segmentation shape {seg_data.shape} does not match "
                    f"volume shape {self.volume_data.shape}!"
                )
                return
            
            # Binary 변환 (0 이상인 값을 모두 1로)
            seg_binary = (seg_data > 0).astype(np.uint8)
            roi_count = np.sum(seg_binary)
            
            if roi_count == 0:
                QMessageBox.warning(self, "Warning", "Segmentation mask is empty!")
                return
            
            self.segmentation_data = seg_binary
            
            filename = os.path.basename(file_path)
            self.seg_info_label.setText(
                f"🧩 {filename} | ROI voxels: {roi_count:,}"
            )
            self.seg_info_label.setStyleSheet("color: #FF9800; padding: 2px; font-size: 11px;")
            
            # 시그널 발송
            self.segmentation_loaded.emit(self.segmentation_data)
            self.emit_status(f"Segmentation loaded: {filename} ({roi_count:,} ROI voxels)")
            
            print(f"✅ Segmentation loaded: {filename}")
            print(f"   Shape: {seg_binary.shape}, ROI voxels: {roi_count}")
            
        except Exception as e:
            error_msg = f"Failed to load segmentation: {str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.seg_info_label.setText(f"❌ Seg load failed")
            self.seg_info_label.setStyleSheet("color: #f44336; padding: 2px; font-size: 11px;")
            import traceback
            traceback.print_exc()