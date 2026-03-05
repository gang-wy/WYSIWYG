import json
import os
import numpy as np
from datetime import datetime
from PyQt6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, 
                           QSlider, QButtonGroup, QFileDialog, QSizePolicy, QMessageBox, QDoubleSpinBox,QLineEdit)
from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QPixmap, QImage
from src.gui.panel.base_panel import BasePanel
from src.core.tf_optimizer import TFOptimizer
import vtk
import numpy as np

class OptimizationPanel(BasePanel):
    """SAM 결과 이미지 표시 및 최적화 컨트롤 패널"""
    
    # 시그널 정의
    set_mode_changed = pyqtSignal(bool)
    point_type_changed = pyqtSignal(str)    
    check_sam = pyqtSignal()
    points_updated = pyqtSignal(list)
    run_optimization_requested = pyqtSignal(dict)
    
    # TF 변경 요청 시그널 (MainWindow -> TFPanel로 전달)
    tf_update_requested = pyqtSignal(list)

    # [NEW] 텍스트 프롬프트 요청 시그널 추가 (text_promt)
    run_text_sam_requested = pyqtSignal(str)

    # [NEW] Visibility-based optimization 시그널
    run_visibility_optimization_requested = pyqtSignal(dict)

    def __init__(self):
        self.is_set_mode = False
        self.current_point_type = "positive"
        self.picked_points = [] 
        self.last_analyzer_result = None
        self.last_text_analyzer_result = None
        super().__init__("Optimization & SAM", collapsible=False)

    def setup_content(self):
        """RenderingPanel과 완벽하게 대칭을 이루도록 레이아웃 조정"""
        self.content_layout.setContentsMargins(0, 0, 0, 0)

        # 1. 헤더 영역
        header_layout = QHBoxLayout()
        self.title_label = QLabel("🎯 SAM & Optimization")
        self.title_label.setStyleSheet("font-size: 18px; font-weight: bold; padding: 10px;")
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()
        self.content_layout.addLayout(header_layout)
        
        # 2. SAM Canvas
        self.image_display = QLabel("No Image Loaded")
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setStyleSheet("background-color: #000000; border: 1px solid #444; border-radius: 4px;")
        self.image_display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.content_layout.addWidget(self.image_display, stretch=1)
        
        # 스타일 정의
        base_btn_style = "QPushButton { height: 34px; font-weight: bold; padding: 0 10px; }"
        toggle_btn_style = base_btn_style + """
            QPushButton:checked#set_btn { 
                background-color: #2ecc71; 
                color: white; 
                border: 1px solid #27ae60;
            }
            QPushButton:checked#pos_btn { background-color: #2196F3; color: white; }
            QPushButton:checked#neg_btn { background-color: #FF5252; color: white; }
        """

        # 3. Mode & Type Selection
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
        self.pos_btn.setEnabled(False)
        self.pos_btn.setStyleSheet(toggle_btn_style)
        
        self.neg_btn = QPushButton("Negative")
        self.neg_btn.setObjectName("neg_btn")
        self.neg_btn.setCheckable(True)
        self.neg_btn.setEnabled(False)
        self.neg_btn.setStyleSheet(toggle_btn_style)

        self.pos_btn.setVisible(False)
        self.neg_btn.setVisible(False)
        
        self.save_btn = QPushButton("💾 Save Pts")
        self.load_btn = QPushButton("📥 Load Pts")
        self.save_btn.setStyleSheet(base_btn_style)
        self.load_btn.setStyleSheet(base_btn_style)

        self.type_group = QButtonGroup(self)
        self.type_group.addButton(self.pos_btn)
        self.type_group.addButton(self.neg_btn)

        for btn in [self.set_btn, self.pos_btn, self.neg_btn, self.save_btn, self.load_btn]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            point_layout.addWidget(btn)
            
        self.content_layout.addWidget(point_group)
        
        # 4. Optimization Settings
        opt_group, opt_layout = self.create_group_box("", "horizontal")
        opt_layout.setContentsMargins(10, 0, 10, 0)
        opt_layout.setSpacing(15)
        
        # Maxiter Slider
        opt_layout.addWidget(QLabel(" Maxiter:"))
        self.thresh_slider = QSlider(Qt.Orientation.Horizontal)
        self.thresh_slider.setRange(10, 200) # 범위 현실화 (10~200회)
        self.thresh_slider.setValue(50)
        self.thresh_slider.setTickInterval(10)
        opt_layout.addWidget(self.thresh_slider, stretch=1) 
        
        self.thresh_label = QLabel("50")
        self.thresh_label.setFixedWidth(30)
        self.thresh_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        opt_layout.addWidget(self.thresh_label)

        # FTOL SpinBox
        opt_layout.addWidget(QLabel(" Tolerance:"))
        self.spin_ftol = QDoubleSpinBox()
        self.spin_ftol.setRange(1e-9, 1.0)
        self.spin_ftol.setDecimals(5)
        self.spin_ftol.setSingleStep(0.0001)
        self.spin_ftol.setValue(0.001)
        self.spin_ftol.setToolTip("Function Tolerance (종료 기준)")
        self.spin_ftol.setFixedWidth(100)
        opt_layout.addWidget(self.spin_ftol)
        
        self.content_layout.addWidget(opt_group)
        
        # ⭐ [NEW] Optimizer Selection
        optimizer_group, optimizer_layout = self.create_group_box("", "horizontal")
        optimizer_layout.setContentsMargins(10, 5, 10, 5)
        optimizer_layout.setSpacing(10)

        optimizer_layout.addWidget(QLabel(" Method:"))

        self.nelder_mead_radio = QPushButton("Visibility-based (Comp.)")
        self.nelder_mead_radio.setCheckable(True)
        self.nelder_mead_radio.setStyleSheet(base_btn_style)
        self.nelder_mead_radio.setToolTip("Comparison method: Visibility-based TF optimization using 3D segmentation mask")

        self.pytorch_radio = QPushButton("PyTorch Gradient (Fast)")
        self.pytorch_radio.setCheckable(True)
        self.pytorch_radio.setChecked(True)
        self.pytorch_radio.setStyleSheet(base_btn_style)

        self.optimizer_button_group = QButtonGroup(self)
        self.optimizer_button_group.addButton(self.nelder_mead_radio)
        self.optimizer_button_group.addButton(self.pytorch_radio)

        for btn in [self.nelder_mead_radio, self.pytorch_radio]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            optimizer_layout.addWidget(btn)

        self.content_layout.addWidget(optimizer_group)

        # 5. Action Buttons
        save_group, save_layout = self.create_group_box("", "horizontal")
        save_layout.setContentsMargins(10, 5, 10, 5)
        save_layout.setSpacing(8)
        
        self.check_btn = QPushButton("🔍 Check SAM")
        self.check_btn.setStyleSheet(base_btn_style)
        
        self.run_btn = QPushButton("🚀 Run Optimization")
        self.run_btn.setStyleSheet("QPushButton {background-color: #2E7D32; color: white; font-weight: bold; border-radius: 4px; height: 34px; padding: 0 10px; }")

        for btn in [self.check_btn, self.run_btn]:
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            save_layout.addWidget(btn)
            
        self.content_layout.addWidget(save_group)

        # [NEW] Text Promp UI 추가
        text_group, text_layout = self.create_group_box("", "horizontal")
        text_layout.setContentsMargins(10, 5, 10, 5)
        
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Enter text prompt (e.g. 'kidney', 'tumor')")
        self.text_input.setStyleSheet("QLineEdit { padding: 5px; border-radius: 4px; border: 1px solid #555; }")
        
        self.text_run_btn = QPushButton("Run Text SAM")
        self.text_run_btn.setStyleSheet("QPushButton { background-color: #5c6bc0; color: white; font-weight: bold; padding: 5px; }")
        
        text_layout.addWidget(self.text_input, stretch=1)
        text_layout.addWidget(self.text_run_btn)
        
        self.content_layout.addWidget(text_group)

        # 시그널 연결
        self.set_btn.toggled.connect(self.on_set_mode_toggled)
        self.pos_btn.clicked.connect(lambda: self.change_point_type("positive"))
        self.neg_btn.clicked.connect(lambda: self.change_point_type("negative"))
        self.run_btn.clicked.connect(self.on_run_optimization_clicked) # [수정] 핸들러 변경
        self.thresh_slider.valueChanged.connect(self.on_thresh_changed)

        self.save_btn.clicked.connect(self.save_points)
        self.load_btn.clicked.connect(self.load_points)
        self.check_btn.clicked.connect(lambda: self.check_sam.emit())

        # [NEW] 시그널 연결
        self.text_run_btn.clicked.connect(self.on_run_text_sam_clicked)

    # 텍스트 SAM 버튼 클릭 핸들러
    def on_run_text_sam_clicked(self):
        """텍스트 SAM 실행 버튼 클릭 핸들러"""
        text = self.text_input.text().strip()
        if not text:
            QMessageBox.warning(self, "Warning", "프롬프트 텍스트를 입력해주세요.")
            return
        
        # MainWindow로 시그널 발송
        self.run_text_sam_requested.emit(text)

    # --- 최적화 실행 로직 ---
    def on_run_optimization_clicked(self):
        """
        최적화 버튼 클릭 시:
        직접 로직을 돌리지 않고 파라미터만 모아서 MainWindow로 시그널 발송
        """
        # Visibility-based mode: segmentation 필요, SAM 불필요
        if self.nelder_mead_radio.isChecked():
            params = {
                'maxiter': self.thresh_slider.value(),
                'ftol': self.spin_ftol.value(),
                'use_visibility': True,
            }
            self.run_visibility_optimization_requested.emit(params)
            return
        
        # PyTorch (Adam) mode: SAM 결과 필요
        if not hasattr(self, 'last_analyzer_result') or self.last_analyzer_result is None:
            QMessageBox.warning(self, "Warning", "먼저 SAM을 실행하여 분석 데이터를 생성해주세요.")
            return

        # UI에서 설정값 가져오기
        params = {
            'maxiter': self.thresh_slider.value(),
            'ftol': self.spin_ftol.value(),
            'use_pytorch': self.pytorch_radio.isChecked() 
        }
        
        # MainWindow로 요청
        self.run_optimization_requested.emit(params)

    def project_points_to_binary_mask(self, renderer, picked_points):
        """
        VTK 렌더러를 사용하여 3D 포인트들을 2D로 투영하고 binary mask 생성
        
        Args:
            renderer: VTKVolumeRenderer 인스턴스
            picked_points: List of 3D points (can be list of arrays or list of dicts)
            
        Returns:
            numpy array: Binary mask (H, W) with 0 and 1 values
        """
        try:
            # VTK 렌더러와 렌더 윈도우 가져오기
            if not hasattr(renderer, 'renderer') or not renderer.renderer:
                self.emit_status("Error: VTK renderer not available")
                return None
                
            vtk_renderer = renderer.renderer
            render_window = renderer.vtk_widget.GetRenderWindow()
            
            # 현재 렌더 윈도우 크기 가져오기
            width, height = render_window.GetSize()
            
            if width == 0 or height == 0:
                self.emit_status("Error: Invalid render window size")
                return None
            
            # Binary mask 초기화 (모두 0으로)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            
            # 카메라 가져오기
            camera = vtk_renderer.GetActiveCamera()
            if not camera:
                self.emit_status("Error: Camera not available")
                return None
            
            # vtkCoordinate를 사용한 월드->디스플레이 좌표 변환
            coordinate = vtk.vtkCoordinate()
            coordinate.SetCoordinateSystemToWorld()
            
            projected_2d_points = []
            
            for point_data in picked_points:
                # 데이터 타입에 따라 처리
                if isinstance(point_data, dict):
                    world_pos = point_data.get('world_pos')
                    point_type = point_data.get('point_type', 'positive')
                elif isinstance(point_data, (list, tuple, np.ndarray)):
                    # numpy array나 리스트인 경우
                    world_pos = point_data
                    point_type = 'positive'  # 기본값
                else:
                    continue
                
                if world_pos is None or len(world_pos) != 3:
                    continue
                
                # 3D 월드 좌표 설정
                coordinate.SetValue(float(world_pos[0]), float(world_pos[1]), float(world_pos[2]))
                
                # 디스플레이 좌표로 변환 (픽셀 좌표)
                display_pos = coordinate.GetComputedDisplayValue(vtk_renderer)
                
                # VTK의 디스플레이 좌표는 좌하단이 원점이므로 Y 좌표 반전
                x_pixel = int(display_pos[0])
                y_pixel = int(height - display_pos[1] - 1)
                
                # 이미지 범위 내에 있는지 확인
                if 0 <= x_pixel < width and 0 <= y_pixel < height:
                    projected_2d_points.append({
                        'pixel_pos': (x_pixel, y_pixel),
                        'world_pos': world_pos,
                        'point_type': point_type
                    })
                    
                    # Positive 포인트만 마스크에 표시
                    if point_type == "positive":
                        binary_mask[y_pixel, x_pixel] = 1
                        
                    # print(f"✓ Projected: World {world_pos} -> Pixel ({x_pixel}, {y_pixel})")
                else:
                    print(f"⚠ Point outside view: ({x_pixel}, {y_pixel})")
            
            if len(projected_2d_points) == 0:
                self.emit_status("Warning: No points projected within view")
                return None
            
            self.emit_status(f"✓ Projected {len(projected_2d_points)} points to 2D mask")
            return {
                'binary_mask': binary_mask,
                'projected_points': projected_2d_points
            }
            
        except Exception as e:
            import traceback
            self.emit_status(f"Error in projection: {str(e)}")
            print(f"❌ Projection error: {e}")
            traceback.print_exc()
            return None
    
    def set_image_from_binary_mask(self, binary_mask: np.ndarray):
        """
        binary_mask: (H, W) with values in {0,1} or {0,255} or bool
        -> self.image_display(QLabel)에 표시
        """
        if binary_mask is None:
            self.image_display.setText("No Mask")
            return

        # 1) uint8 (0~255)로 정규화
        if binary_mask.dtype == np.bool_:
            mask_u8 = (binary_mask.astype(np.uint8) * 255)
        else:
            m = binary_mask
            # 값이 0~1이면 255 스케일
            if m.max() <= 1:
                mask_u8 = (m.astype(np.float32) * 255).astype(np.uint8)
            else:
                mask_u8 = m.astype(np.uint8)

        # 2) QImage 생성 (Grayscale8)
        h, w = mask_u8.shape
        bytes_per_line = w  # Grayscale8은 1 byte/pixel
        qimg = QImage(mask_u8.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)

        # ⚠️ 중요: numpy 메모리 참조 문제 방지 (copy 필수)
        qimg = qimg.copy()

        # 3) 기존 함수로 표시
        self.update_image(qimg)

    # --- 데이터 수신 (MainWindow에서 호출) ---
    def set_analyzer_result(self, result):
        """MainWindow에서 SAM 실행 후 분석된 데이터를 저장"""
        self.last_analyzer_result = result
        self.run_btn.setEnabled(True)
        print("✅ Point Analysis data received in OptimizationPanel.")

    def set_text_analyzer_result(self,result):
        """MainWindow에서 SAM 실행 후 분석된 데이터를 저장"""
        self.last_text_analyzer_result = result
        self.run_btn.setEnabled(True)
        print("✅ Text Analysis data received in OptimizationPanel.")

    # --- 기존 UI 핸들러들 ---
    def on_set_mode_toggled(self, checked):
        self.set_btn.setText("Set Mode: ON" if checked else "Enable Set Mode")
        self.pos_btn.setVisible(checked)
        self.neg_btn.setVisible(checked)
        self.pos_btn.setEnabled(checked)
        self.neg_btn.setEnabled(checked)
        self.set_mode_changed.emit(checked)

    def change_point_type(self, p_type):
        self.current_point_type = p_type
        self.pos_btn.setChecked(p_type == "positive")
        self.neg_btn.setChecked(p_type == "negative")
        self.point_type_changed.emit(p_type)

    def on_thresh_changed(self, value):
        self.thresh_label.setText(f"{value}")

    def save_points(self):
        if not self.picked_points:
            QMessageBox.warning(self, "Warning", "저장할 포인트가 없습니다.")
            return
        default_path = "./resources/Points"
        os.makedirs(default_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_file = os.path.join(default_path, f"points_{timestamp}.json")
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Points", default_file, "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(self.picked_points, f, indent=4)
            self.emit_status(f"Saved: {os.path.basename(file_path)}")

    def load_points(self):
        default_path = "./resources/Points"
        if not os.path.exists(default_path): os.makedirs(default_path, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Points", default_path, "JSON Files (*.json)")
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self.picked_points = data
                    self.points_updated.emit(self.picked_points)
                    self.emit_status(f"Loaded {len(self.picked_points)} points.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"로드 실패: {e}")

    def set_image_from_binary_mask(self, binary_mask):
        if binary_mask is None: return
        if binary_mask.dtype == np.bool_:
            mask_u8 = (binary_mask.astype(np.uint8) * 255)
        else:
            mask_u8 = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask.astype(np.uint8)
            
        h, w = mask_u8.shape
        qimg = QImage(mask_u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        self.update_image(qimg)

    def set_image_by_path(self, file_path):
        if os.path.exists(file_path):
            self.update_image(QImage(file_path))
        else:
            self.image_display.setText("File Not Found")

    def update_image(self, qimage):
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(self.image_display.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
        self.image_display.setPixmap(scaled_pixmap)

    def emit_status(self, msg):
        # BasePanel에는 statusbar 접근 권한이 없으므로 print로 대체하거나
        # MainWindow에서 연결된 시그널을 통해 처리해야 함.
        print(f"[OptPanel] {msg}")