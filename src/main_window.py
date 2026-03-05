import os
from datetime import datetime
from PIL import Image
import numpy as np

# VTK 환경 설정
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
os.environ['VTK_USE_OSMESA'] = '1'

import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

# 패널 임포트
from src.gui.panel.file_panel import FilePanel
from src.gui.panel.tf_panel import TransferFunctionPanel
from src.gui.panel.rendering_panel import RenderingPanel
from src.gui.panel.optimization_panel import OptimizationPanel
from src.core.support_sam import SAMService
from src.core.feature_analyzer import FeatureAnalyzer
from src.core.tf_optimizer import TFOptimizer
from src.core.support_optimization import OptimizationWorker
from src.core.utils.tf_utils import find_target_range_from_tents

class VolumeRenderingMainWindow(QMainWindow):
    """간결화된 메인 윈도우 - Standard 렌더링 및 SAM 최적화 지원"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Rendering & Optimization Tool")
        self.setGeometry(100, 100, 1800, 900)
        self.setMinimumSize(1800, 900)
        self.setMaximumSize(1800, 900)
        
        self.volume_data = None
        self.voxel_spacing = (1.0, 1.0, 1.0)
        self.segmentation_data = None  # [NEW] 3D binary segmentation mask
        
        self.sam_model = SAMService()
        self.sam_model.loaded.connect(lambda: print("SAM loaded"))
        self.sam_model.predicted.connect(self.on_mask)
        self.sam_model.text_predicted.connect(self.on_text_mask)  # [NEW] 텍스트 SAM 결과 연결
        self.sam_model.error.connect(self.on_sam_error)
        self.sam_model.load_async()

        self.init_ui()
        self.create_panels()
        self.connect_signals()
        
        self.statusBar().showMessage("Ready - Load volume data to start")
        self.tf_panel.reset_background_color()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # 1. 왼쪽 패널
        left_container = QWidget()
        left_container.setFixedWidth(600)
        left_container.setStyleSheet("background-color: #2d2d2d;")
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #2d2d2d; }")
        
        scroll_content = QWidget()
        self.left_layout = QVBoxLayout(scroll_content)
        self.left_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(scroll_content)
        
        left_main_layout = QVBoxLayout(left_container)
        left_main_layout.addWidget(scroll_area)
        main_layout.addWidget(left_container)

        # 2. 중앙 렌더링 영역
        self.center_widget = QWidget()
        self.center_layout = QVBoxLayout(self.center_widget)
        self.center_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.center_widget, stretch=1)

        # 3. 오른쪽 최적화 패널 영역
        self.right_container = QWidget()
        self.right_layout = QVBoxLayout(self.right_container)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.right_container, stretch=1)

    def create_panels(self):
        self.file_panel = FilePanel()
        self.left_layout.addWidget(self.file_panel)

        self.tf_panel = TransferFunctionPanel()
        self.left_layout.addWidget(self.tf_panel)
        self.left_layout.addStretch(1)

        self.rendering_panel = RenderingPanel()
        self.center_layout.addWidget(self.rendering_panel)
        
        self.optimization_panel = OptimizationPanel()
        self.right_layout.addWidget(self.optimization_panel)
        
        self.tf_panel.rendering_panel = self.rendering_panel

    def connect_signals(self):
        self.file_panel.volume_loaded.connect(self.on_volume_loaded)
        self.tf_panel.tf_changed.connect(self.on_tf_changed)
        self.tf_panel.background_color_changed.connect(self.on_background_color_changed)
        self.tf_panel.shading_changed.connect(self.on_shading_changed)
        self.tf_panel.lighting_changed.connect(self.on_lighting_changed)
        self.tf_panel.light_direction_changed.connect(self.on_light_direction_changed)
        self.tf_panel.follow_camera_changed.connect(self.on_follow_camera_changed)
        self.tf_panel.ambient_color_changed.connect(self.on_ambient_color_changed)
        self.tf_panel.diffuse_color_changed.connect(self.on_diffuse_color_changed)
        self.tf_panel.specular_color_changed.connect(self.on_specular_color_changed)
        self.tf_panel.clipping_changed.connect(self.on_clipping_changed)
        self.tf_panel.clipping_enabled_changed.connect(self.on_clipping_enabled_changed)

        # Optimization Panel 시그널
        self.optimization_panel.set_mode_changed.connect(self.on_set_mode_changed)
        self.optimization_panel.point_type_changed.connect(self.on_point_type_changed)
        self.optimization_panel.check_sam.connect(self.on_check_sam)
        self.optimization_panel.points_updated.connect(self.on_points_updated)
        
        self.optimization_panel.tf_update_requested.connect(self.tf_panel.apply_external_nodes)
        self.optimization_panel.run_optimization_requested.connect(self.start_optimization_process)
        self.optimization_panel.run_text_sam_requested.connect(self.on_run_text_sam)  # [NEW] 텍스트 SAM 연결
        
        # [NEW] Segmentation 로드 시그널
        self.file_panel.segmentation_loaded.connect(self.on_segmentation_loaded)
        
        # [NEW] Visibility-based optimization 시그널
        self.optimization_panel.run_visibility_optimization_requested.connect(
            self.start_visibility_optimization_process
        )

        # 렌더러의 2D 피킹 시그널 연결
        if hasattr(self.rendering_panel.vtk_renderer, 'point_2d_picked'):
            self.rendering_panel.vtk_renderer.point_2d_picked.connect(self.on_point_2d_picked)

    def on_set_mode_changed(self, enabled):
        """Set 모드 변경 시 데이터 및 화면 강제 초기화"""
        
        # 1. 데이터 초기화
        self.optimization_panel.picked_points = []
        self.optimization_panel.points_updated.emit([])

        # 2. VTK 오버레이 초기화
        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.clear_overlay_points()
            self.rendering_panel.set_overlay_visible(enabled)

        # 3. VTK 상호작용 설정
        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.set_interaction_enabled(not enabled)
            self.rendering_panel.vtk_renderer.set_picking_enabled(enabled)
        
        status = "SET MODE: ON (Pick Points)" if enabled else "SET MODE: OFF (Navigation)"
        self.statusBar().showMessage(status)

    def on_point_type_changed(self, p_type):
        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.current_pick_type = p_type

    def on_point_2d_picked(self, vtk_x, vtk_y):
        """점이 찍혔을 때 - VTK 좌표 그대로 사용"""
        p_type = self.optimization_panel.current_point_type
        
        # 1. Optimization 패널의 데이터에 추가
        # (중복 방지 등은 패널 내부 로직이나 여기서 처리 가능)
        new_point = {'pos': (vtk_x, vtk_y), 'type': p_type}
        self.optimization_panel.picked_points.append(new_point)
        
        # 2. VTK 렌더러에 시각적 오버레이 추가
        self.rendering_panel.add_point_2d(vtk_x, vtk_y, p_type)
        
        # 로그 출력
        print(f"Point Added: {vtk_x}, {vtk_y} ({p_type})")

    def on_check_sam(self):
        """SAM Check 버튼 클릭 시 실행"""

        points_data = self.optimization_panel.picked_points
        if not points_data:
            QMessageBox.warning(self, "Warning", "선택된 포인트가 없습니다. Set Mode에서 포인트를 먼저 찍어주세요.")
            return

        self.statusBar().showMessage("Capturing Scene & Running SAM... Please wait.")
        QApplication.processEvents()  # UI 갱신

        # 1. 현재 화면 캡처
        image_path = self.rendering_panel.save_current_rendering(return_filename=True)
        if not image_path:
            self.statusBar().showMessage("Error: Failed to capture screenshot.")
            return
        self.last_captured_image_path = image_path  # ⭐ NEW: color reference용 저장

        # 2. 이미지 높이 구하기 (좌표 변환용)
        # 이미지 파일에서 읽거나 렌더러에서 가져옴
        from PIL import Image
        pil_img = Image.open(image_path)
        img_width, img_height = pil_img.size

        input_points = []
        input_labels = []

        for pt in points_data:
            # 딕셔너리에서 좌표와 타입 추출
            vtk_x, vtk_y = pt['pos']
            p_type = pt['type']
            
            # [중요] 좌표 변환
            # VTK 렌더러(Pick)는 좌하단(Bottom-Left)이 (0,0)입니다.
            # 이미지는 좌상단(Top-Left)이 (0,0)이므로 Y축을 뒤집어야 합니다.
            img_x = int(vtk_x)
            img_y = int(img_height - vtk_y) 
            
            # 이미지 범위를 벗어나지 않도록 클램핑(안전장치)
            img_x = max(0, min(img_x, img_width - 1))
            img_y = max(0, min(img_y, img_height - 1))

            input_points.append([img_x, img_y])
            
            # 라벨 변환 (SAM 요구사항: Positive=1, Negative=0)
            label = 1 if p_type == 'positive' else 0
            input_labels.append(label)

        self.sam_model.predict_async(image_path, input_points, input_labels)

    def on_points_updated(self, points_list):
        """
        [핵심] OptimizationPanel에서 Load 또는 Clear가 발생했을 때 호출됨.
        points_list 구조: [{'pos': (x, y), 'type': 'positive'}, ...]
        """
        if not self.rendering_panel.vtk_renderer:
            return

        # 1. VTK 화면의 기존 점들 모두 제거
        self.rendering_panel.clear_overlay()

        # 2. 리스트에 있는 점들을 다시 VTK에 그림
        for pt in points_list:
            # json 로드 시 리스트로 들어올 수 있으므로 튜플로 변환 등 안전처리
            pos = pt['pos']
            p_type = pt['type']
            x, y = pos[0], pos[1]
            
            # RenderingPanel의 메서드를 통해 오버레이 추가
            self.rendering_panel.add_point_2d(x, y, p_type)
        
        # 3. 화면 갱신 (RenderingPanel 내부에서 add_point_2d 시 Render를 호출하도록 수정했지만, 한번 더 확실히)
        if hasattr(self.rendering_panel.vtk_renderer, 'vtk_widget'):
            self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow().Render()
            
        print(f"🔄 View Updated: {len(points_list)} points loaded/cleared.")

    # --- SAM 관련 핸들러들 ---
    def on_mask(self, result):

        # Unpack (mask, logits) tuple from SAM predict
        if isinstance(result, tuple):
            mask, sam_logits = result
        else:
            mask = result
            sam_logits = None

        if mask is None:
            self.statusBar().showMessage("SAM Result: No mask generated.")
            print("No mask result.")
            return
        
        # 폴더가 없으면 생성
        save_dir = "./resources/masks"
        os.makedirs(save_dir, exist_ok=True)

        # 파일명 생성 (mask_YYYYMMDD_HHMMSS.png)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"mask_{timestamp}.png"
        file_path = os.path.join(save_dir, file_name)

        try:
            # Boolean(True/False) -> uint8(255/0) 변환
            # True 부분은 255(흰색), False 부분은 0(검은색)이 됨
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            # 이미지로 저장
            Image.fromarray(mask_uint8).save(file_path)

            self.statusBar().showMessage(f"Mask saved: {file_name}")
            
            # [중요] 경로가 필요하다면 여기서 멤버 변수에 저장하거나 다른 함수로 전달
            self.current_mask_path = file_path 

        except Exception as e:
            print(f"❌ Failed to save mask: {e}")
            self.statusBar().showMessage("Error saving mask file.")

        # ---------------------------------------------------------
        # 2. UI 시각화 (기존 로직)
        # ---------------------------------------------------------
        if hasattr(self.optimization_panel, 'set_image_by_path'):
            self.optimization_panel.set_image_by_path(self.current_mask_path)

        try:
            # 1. 좌표계 보정 (Screen Scaling)
            h, w = mask.shape
            sw, sh = self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow().GetSize()
            sx, sy = sw/w, sh/h

            # 2. 마스크 영역 샘플링
            y_idx, x_idx = np.where(mask > 0)
            stride = 5 # 성능을 위해 샘플링

            # [DEBUG] 추가
            print(f"🔍 Mask Debug:")
            print(f"   Mask shape: {mask.shape}")
            print(f"   Total white pixels: {len(y_idx)}")
            print(f"   After stride={stride}: {len(y_idx[::stride])}")

            coords = np.column_stack((x_idx[::stride]*sx, (h - y_idx[::stride])*sy))

            # Sample logit values at subsampled mask pixels (for per-point certainty)
            logit_values_subsampled = None
            sam_confidence = None
            if sam_logits is not None:
                logit_values_subsampled = sam_logits[y_idx[::stride], x_idx[::stride]]
                # Global confidence: mean certainty inside mask
                sam_confidence = float(np.mean(1.0 / (1.0 + np.exp(-sam_logits[y_idx, x_idx]))))
                print(f"   SAM confidence (global): {sam_confidence:.3f}")

            # 3. Feature Analysis 실행
            analyzer = FeatureAnalyzer(
                self.rendering_panel.vtk_renderer.renderer,
                self.volume_data, self.voxel_spacing,
                volume_actor=self.rendering_panel.vtk_renderer.standard_volume
            )
            
            # 현재 TF의 Opacity LUT 획득
            tf_lut = self.tf_panel.tf_widget.get_opacity_lut() 

            # 클리핑 영역 가져오기
            clipping_ranges = None
            if self.tf_panel and hasattr(self.tf_panel, 'get_clipping_ranges'):
                clipping_ranges = self.tf_panel.get_clipping_ranges()
            print(f"📦 Clipping ranges: {clipping_ranges}")
            
            # ROI Voxel Profile 획득
            results= analyzer.analyze_roi_profile(coords, tf_lut, peeling_depth=1, sample_mode="interval", clipping_ranges=clipping_ranges)
            # - 'profiles': (N, M) 2D Array of intensity profiles (Zero-padded)
            # - 'target_indices': (N,) Indices of the target point in each profile
            # - 'picked_points': List of (x, y, z) world coordinates
            # - 'picked_intensities': List of intensity values at picked points
            # - 'target_range': (min, max) intensity range of targets

            sam_points_data = self.optimization_panel.picked_points  # points_data 대신 이걸로
            sam_screen_coords = np.array([[pt['pos'][0], pt['pos'][1]] for pt in sam_points_data])
            analyzer.visualize_sam_point_opacity(sam_screen_coords, tf_lut, save_dir="ray_profiles_sam_points")
            
            if len(results.get('picked_intensities', [])) > 0:
                # ⭐⭐⭐ Intensity 필터링 ⭐⭐⭐
                picked_points = results['picked_points']
                picked_intensities = results['picked_intensities']
                success_indices = results.get('success_indices', None)

                # Map logit values to successfully picked points
                picked_logit_values = None
                if logit_values_subsampled is not None and success_indices is not None:
                    picked_logit_values = logit_values_subsampled[success_indices]
                
                target_median = np.median(picked_intensities)
                target_std = np.std(picked_intensities)
                
                print(f"\n🎯 Intensity Filtering:")
                print(f"   Original points: {len(picked_points)}")
                print(f"   Median: {target_median:.1f}, Std: {target_std:.1f}")
                
                # ±1σ 범위만 유지
                valid_mask = (
                    (picked_intensities >= target_median - target_std) &
                    (picked_intensities <= target_median + target_std)
                )
                
                # 필터링된 결과
                filtered_points = picked_points[valid_mask]
                filtered_intensities = picked_intensities[valid_mask]
                
                print(f"   After ±1σ: {len(filtered_points)} points ({len(filtered_points)/len(picked_points)*100:.1f}%)")
                
                # 너무 많이 제거되면 완화
                if len(filtered_points) < len(picked_points) * 0.3:
                    print(f"   ⚠️ Too aggressive! Using ±3σ instead.")
                    valid_mask = (
                        (picked_intensities >= target_median - 3 * target_std) &
                        (picked_intensities <= target_median + 3 * target_std)
                    )
                    filtered_points = picked_points[valid_mask]
                    filtered_intensities = picked_intensities[valid_mask]
                    print(f"   After ±3σ: {len(filtered_points)} points")
                
                # # ⭐ 시각화
                # self.rendering_panel.clear_3d_markers()
                
                # # 일부만 표시 (성능)
                # for p in filtered_points[::10]:
                #     self.rendering_panel.add_point_3d(p, "positive")
                
                # ⭐⭐⭐ Ray directions 필터링 ⭐⭐⭐
                filtered_ray_directions = None
                if 'initial_ray_directions' in results and results['initial_ray_directions'] is not None:
                    filtered_ray_directions = results['initial_ray_directions'][valid_mask]
                
                # ⭐⭐⭐ Profiles & target_indices 필터링 ⭐⭐⭐
                filtered_profiles = None
                filtered_target_indices = None
                
                if 'profiles' in results and results['profiles'] is not None:
                    filtered_profiles = results['profiles'][valid_mask]
                
                if 'target_indices' in results and results['target_indices'] is not None:
                    filtered_target_indices = results['target_indices'][valid_mask]
                
                # 통계
                target_val = np.median(filtered_intensities)

                # TF tent 기반 target_range 계산
                current_tf_nodes = self.tf_panel.tf_widget.get_nodes()
                volume_range = (self.volume_data.min(), self.volume_data.max())
                min_v, max_v = find_target_range_from_tents(
                    filtered_intensities, 
                    current_tf_nodes, 
                    volume_range=volume_range
                )
                
                print(f"\n📊 Filtered ROI Feature:")
                print(f"   Count: {len(filtered_intensities)}")
                print(f"   Mean: {np.mean(filtered_intensities):.5f}")
                print(f"   Median: {target_val:.5f}")
                print(f"   Range: [{min_v:.5f}, {max_v:.5f}]")
                
                # ⭐⭐⭐ [NEW] Color Reference 계산 (LAB 색공간) ⭐⭐⭐
                try:
                    import kornia
                    import torch
                    
                    # on_check_sam()에서 저장해둔 캡처 이미지 사용
                    captured_image_path = getattr(self, 'last_captured_image_path', None)
                    if captured_image_path is None:
                        raise ValueError("No captured image path available")
                    
                    captured_image = np.array(Image.open(captured_image_path).convert('RGB')) / 255.0
                    mask_bool = mask > 0
                    
                    # RGB → LAB 변환
                    rgb_tensor = torch.tensor(captured_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
                    lab_tensor = kornia.color.rgb_to_lab(rgb_tensor)  # (1,3,H,W)
                    lab_image = lab_tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H,W,3)
                    
                    masked_pixels = lab_image[mask_bool]
                    
                    if len(masked_pixels) > 0:
                        color_reference = {
                            'mean': torch.tensor(np.mean(masked_pixels, axis=0).astype(np.float32)),
                            'std': torch.tensor(np.std(masked_pixels, axis=0).astype(np.float32)),
                            'source_resolution': (h, w)
                        }
                        print(f"🎨 Color Reference Computed (LAB):")
                        print(f"   Mean LAB: [L:{color_reference['mean'][0]:.3f}, a:{color_reference['mean'][1]:.3f}, b:{color_reference['mean'][2]:.3f}]")
                        print(f"   Std LAB: [L:{color_reference['std'][0]:.3f}, a:{color_reference['std'][1]:.3f}, b:{color_reference['std'][2]:.3f}]")
                    else:
                        color_reference = None
                        print("⚠️ No pixels in mask for color reference")
                except Exception as e:
                    print(f"⚠️ Color reference computation failed: {e}")
                    color_reference = None

                # ⭐ Per-point certainty weights from SAM logits
                point_certainty_weights = None
                if picked_logit_values is not None:
                    filtered_logit_values = picked_logit_values[valid_mask]
                    # sigmoid → 1.0 = certain interior, 0.5 = boundary
                    point_certainty_weights = 1.0 / (1.0 + np.exp(-filtered_logit_values))
                    print(f"   SAM certainty weights: min={point_certainty_weights.min():.3f}, "
                          f"max={point_certainty_weights.max():.3f}, "
                          f"mean={point_certainty_weights.mean():.3f}")

                # ⭐ analyzer_results 구성 (모두 필터링된 데이터)
                analyzer_results = {
                    'profiles': filtered_profiles,
                    'target_indices': filtered_target_indices,
                    'target_range': (min_v, max_v),
                    'picked_points': filtered_points,
                    'initial_ray_directions': filtered_ray_directions,
                    "current_tf_nodes": self.tf_panel.tf_widget.get_nodes(),
                    'color_reference': color_reference,
                    'point_certainty_weights': point_certainty_weights,
                    'sam_confidence': sam_confidence,
                }
                
                self.optimization_panel.set_analyzer_result(analyzer_results)


        except Exception as e:
            print(f"Feature Analysis Error: {e}")

    def on_sam_error(self, msg):
        print("SAM error:", msg)

    def on_run_text_sam(self, text_prompt):
        """텍스트 프롬프트 기반 SAM 실행"""
        if not text_prompt:
            QMessageBox.warning(self, "Warning", "텍스트 프롬프트를 입력해주세요.")
            return
        
        self.statusBar().showMessage(f"Running Text SAM with prompt: '{text_prompt}'...")
        QApplication.processEvents()
        
        self.rendering_panel.clear_3d_markers()

        # 1. 현재 화면 캡처
        image_path = self.rendering_panel.save_current_rendering(return_filename=True)
        if not image_path:
            self.statusBar().showMessage("Error: Failed to capture screenshot.")
            return
        
        print(f"📸 Captured image: {image_path}")
        print(f"🔤 Running Text SAM with prompt: '{text_prompt}'")
        
        # 2. SAM 텍스트 예측 비동기 호출
        self.sam_model.predict_text_async(image_path, text_prompt)
    
    def on_text_mask(self, mask):
        """텍스트 SAM 결과 처리"""
        if mask is None:
            self.statusBar().showMessage("Text SAM: No objects found.")
            QMessageBox.information(self, "Result", "지정된 프롬프트에 해당하는 객체를 찾지 못했습니다.")
            return
        
        print(f"✅ Text SAM mask received: shape={mask.shape}, dtype={mask.dtype}")
        
        # 마스크 표시
        self.optimization_panel.set_image_from_binary_mask(mask)
        
        # 마스크 저장 (디버깅용)
        save_dir = "./resources/text_sam_results/"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mask_path = os.path.join(save_dir, f"text_mask_{timestamp}.png")
        
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(mask_path)
        
        self.statusBar().showMessage(f"Text SAM completed. Mask saved to: {mask_path}")
        print(f"💾 Mask saved: {mask_path}")

        try:
            # 1. 좌표계 보정 (Screen Scaling)
            h, w = mask.shape
            sw, sh = self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow().GetSize()
            sx, sy = sw/w, sh/h

            # 2. 마스크 영역 샘플링
            y_idx, x_idx = np.where(mask > 0)
            stride = 5 # 성능을 위해 샘플링
            coords = np.column_stack((x_idx[::stride]*sx, (h - y_idx[::stride])*sy))

            # 3. Feature Analysis 실행
            analyzer = FeatureAnalyzer(
                self.rendering_panel.vtk_renderer.renderer,
                self.volume_data, self.voxel_spacing,
                volume_actor=self.rendering_panel.vtk_renderer.standard_volume
            )
            
            # 현재 TF의 Opacity LUT 획득
            tf_lut = self.tf_panel.tf_widget.get_opacity_lut() 
            
            # ROI Voxel Profile 획득
            results= analyzer.analyze_roi_profile(coords, tf_lut, peeling_depth=1, sample_mode="interval")
            # - 'profiles': (N, M) 2D Array of intensity profiles (Zero-padded)
            # - 'target_indices': (N,) Indices of the target point in each profile
            # - 'picked_points': List of (x, y, z) world coordinates
            # - 'picked_intensities': List of intensity values at picked points
            # - 'target_range': (min, max) intensity range of targets
            

            if len(results.get('picked_intensities', [])) > 0:
                # 시각적 확인을 위해 3D 마커 표시
                self.rendering_panel.clear_3d_markers()
                # #                     
                # for p in world_points[::10]:  # 너무 많으면 렌더링 느려지므로 샘플링 // 전체 확인용 
                #     self.rendering_panel.add_point_3d(p, "positive")                
                # for p in results['picked_points'][::10]:  # 너무 많으면 렌더링 느려지므로 샘플링 // 전체 확인용 
                #      self.rendering_panel.add_point_3d(p, "positive")

                # 4. 통계 기반 최적화 준비
                target_val = np.median(results['picked_intensities'])
                min_v = min(results['picked_intensities'])
                max_v = max(results['picked_intensities'])
                print(f"mean intensity value in ROI: {np.mean(results['picked_intensities']):.1f}")
                print(f"📊 ROI Feature: Median Value={target_val:.1f}, Count={len(results['picked_intensities'])}")

                analyzer_results = {
                    'profiles': results['profiles'],
                    'target_indices': results['target_indices'],
                    'target_range': (min_v, max_v),
                    'picked_points' : results['picked_points'],
                    'initial_ray_directions': results.get('initial_ray_directions', None),  # ⭐ 추가
                    "current_tf_nodes": self.tf_panel.tf_widget.get_nodes(),
                }
                self.optimization_panel.set_text_analyzer_result(analyzer_results)

        except Exception as e:
            print(f"Feature Analysis Error: {e}")

    def on_sam_error(self, msg):
        print("SAM error:", msg)

    def start_optimization_process(self, params):
        """최적화 스레드 시작 준비 및 실행"""
        
        # 1. 데이터 준비 (Analyzer Result)
        analyzer_data = self.optimization_panel.last_analyzer_result
        if not analyzer_data: return

        text_analyzer_data = self.optimization_panel.last_text_analyzer_result

        # ⭐ 3D world coordinates (초기 picks)
        picked_points = analyzer_data.get('picked_points', [])
        text_picked_points = text_analyzer_data.get('picked_points', []) if text_analyzer_data else []

        
        # 2. RenderingPanel에 Projection 요청 (Controller 역할)
        proj_data = self.rendering_panel.get_projection_data(picked_points)
        text_proj_data = self.rendering_panel.get_projection_data(text_picked_points) if text_picked_points else None

        if not proj_data or not proj_data['projected_points']:
            QMessageBox.warning(self, "Error", "포인트 투영 실패")
            return
            
        if text_picked_points:
            text_proj_data = self.rendering_panel.get_projection_data(text_picked_points)
        else:
            text_proj_data = None
        
        # 3. 데이터 패키징
        gt_mask = proj_data['binary_mask']
        gt_text_mask = text_proj_data['binary_mask'] if text_proj_data else None
        projected_points_list = [p['pixel_pos'] for p in proj_data['projected_points']]
        projected_text_point_list = [p['pixel_pos'] for p in text_proj_data['projected_points']] if text_proj_data else []

        # select random single point for verification
        if len(projected_points_list) > 1:
            import random
            selected = random.choice(projected_points_list)
            print(f"투영된 클릭 포인트 중 랜덤 선택: {selected}")
        else:
            selected = projected_points_list[0] if projected_points_list else None # 안전장치

        # ⭐ text는 있을 때만 선택
        if projected_text_point_list and len(projected_text_point_list) > 1:
            text_selected = random.choice(projected_text_point_list)
            print(f"투영된 텍스트 포인트 중 랜덤 선택: {text_selected}")
        else:
            text_selected = projected_text_point_list[0] if projected_text_point_list else None
        
        # GT 마스크 확인용 UI 업데이트
        self.optimization_panel.set_image_from_binary_mask(gt_mask)

        # 4. SAM Wrapper 준비
        sam_wrapper = self.sam_model._worker._sam

        # 5. Optimizer 인스턴스 생성 (구버전 호환용, worker 내부에서는 안쓸 수도 있음)
        current_nodes = self.tf_panel.tf_widget.get_nodes()
        optimizer = TFOptimizer(analyzer_data, current_nodes)

        # 6. Worker & Thread 생성
        self.opt_thread = QThread()
        self.opt_worker = OptimizationWorker(
            optimizer, 
            sam_wrapper, 
            gt_mask, 
            gt_text_mask,
            selected,
            projected_points_list,
            picked_points,
            text_selected,
            projected_text_point_list,
            self.tf_panel.get_clipping_ranges()
        )

        # ⭐ Initial ray directions와 intensities 전달
        if analyzer_data.get('initial_ray_directions') is not None:
            self.opt_worker.initial_ray_directions = analyzer_data['initial_ray_directions']
            print(f"✅ Initial ray directions set: {len(self.opt_worker.initial_ray_directions)} directions")

        if analyzer_data.get('picked_intensities') is not None:
            self.opt_worker.initial_intensities = analyzer_data['picked_intensities']
            print(f"✅ Initial intensities set: {len(self.opt_worker.initial_intensities)} values")


        # ⭐⭐⭐ [핵심 수정] PyTorch 옵션 적용 (조건문 제거하고 항상 실행) ⭐⭐⭐
        use_pytorch = params.get('use_pytorch', False)
        method_name = "Adam" if use_pytorch else "Nelder-Mead"
        vol_name = getattr(self, 'current_volume_name', 'UnknownVolume')
        vol_name = vol_name.replace(" ", "_")

        # 1) 공통 데이터 준비
        volume_data = self.volume_data
        spacing = self.voxel_spacing
        initial_nodes = self.tf_panel.tf_widget.get_nodes()
        
        # Camera 정보 가져오기
        camera = self.rendering_panel.vtk_renderer.renderer.GetActiveCamera()
        camera_info = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'view_angle': camera.GetViewAngle()
        }
        
        # Feature Analyzer 준비
        feature_analyzer = None
        if hasattr(self, 'volume_data') and self.volume_data is not None:
            from src.core.feature_analyzer import FeatureAnalyzer
            feature_analyzer = FeatureAnalyzer(
                self.rendering_panel.vtk_renderer.renderer,
                self.volume_data,
                self.voxel_spacing,
                volume_actor=self.rendering_panel.vtk_renderer.standard_volume
            )

        use_vtk_rendering = True 
        render_window = self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow()
        vtk_width, vtk_height = render_window.GetSize()
        vtk_resolution = (vtk_width, vtk_height)

        # 2) PyTorch Optimizer 초기화 (무조건 실행!)
        try:
            self.opt_worker.set_pytorch_optimizer(
                volume_data=volume_data,
                spacing=spacing,
                initial_nodes=initial_nodes,
                camera_info=camera_info,
                feature_analyzer=feature_analyzer,
                use_vtk_rendering=use_vtk_rendering,
                vtk_resolution=vtk_resolution,
                target_range=analyzer_data.get('target_range'),
                initial_ray_directions=analyzer_data.get('initial_ray_directions'),
                color_reference=analyzer_data.get('color_reference'),
                gt_mask=gt_mask,
                point_certainty_weights=analyzer_data.get('point_certainty_weights'),
                sam_confidence=analyzer_data.get('sam_confidence'),
            )
            
            # 3) 모드 설정 (Adam vs Nelder-Mead)
            self.opt_worker.use_pytorch = use_pytorch
            self.opt_worker.optimization_method_name = method_name
            self.opt_worker.volume_name = vol_name.split('.')[0]  # 확장자 제거

            # PyTorch optimizer에도 initial data 전달
            if hasattr(self.opt_worker, 'pytorch_optimizer') and self.opt_worker.pytorch_optimizer:
                if analyzer_data.get('initial_ray_directions') is not None:
                    self.opt_worker.pytorch_optimizer.initial_ray_directions = analyzer_data['initial_ray_directions']
                if analyzer_data.get('picked_intensities') is not None:
                    self.opt_worker.pytorch_optimizer.initial_intensities = analyzer_data['picked_intensities']
            
            print(f"✅ Optimizer initialized. Mode: {'PyTorch(Adam)' if use_pytorch else 'Nelder-Mead'}")

        except Exception as e:
            print(f"❌ Optimizer initialization failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Optimizer 초기화 실패:\n{e}")
            return # 초기화 실패시 중단

        self.opt_worker.moveToThread(self.opt_thread)

        # 시그널 연결
        self.opt_thread.started.connect(self.opt_worker.run)
        self.opt_worker.finished.connect(self.opt_thread.quit)
        self.opt_worker.finished.connect(self.opt_worker.deleteLater)
        self.opt_thread.finished.connect(self.opt_thread.deleteLater)
        
        self.opt_worker.grid_samples_ready.connect(self.on_grid_samples_ready)
        self.opt_worker.renderer_ref = self.rendering_panel.vtk_renderer
        self.opt_worker.feature_analyzer = feature_analyzer # worker에도 analyzer 직접 할당 (Legacy 지원)
        
        # 동기화 렌더링 요청 처리
        self.opt_worker.request_render_sync.connect(self.handle_sync_render_request)
        self.opt_worker.result_ready.connect(self.on_optimization_finished)

        # 시작
        self.statusBar().showMessage(f"Running Optimization ({'Adam' if use_pytorch else 'Nelder-Mead'})...")
        self.opt_thread.start()

    def handle_sync_render_request(self, new_nodes):
        """
        워커 스레드가 멈춰있는 동안 메인 스레드에서 실행됨
        """
        if hasattr(self.tf_panel, 'tf_widget'):
            self.tf_panel.tf_widget.set_nodes(new_nodes)
            self.rendering_panel.update_transfer_function(self.tf_panel.tf_widget.get_nodes())

        
        self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow().Render()
        
        if self.rendering_panel.vtk_renderer:
            # 렌더링 윈도우 가져오기
            render_window = self.rendering_panel.vtk_renderer.vtk_widget.GetRenderWindow()
            
            # (옵션) 렌더러가 데이터 변경을 인지하도록 강제 Modified 호출이 필요할 수 있음
            self.rendering_panel.vtk_renderer.standard_property.Modified()
            
            # 그리기 명령
            render_window.Render()
        
        # 4. 화면 캡처 (이제 바뀐 화면이 캡처됨)
        image_path = self.rendering_panel.save_current_rendering(return_filename=True)
        
        # 4. 결과 경로를 워커에게 전달하고 워커 깨움
        self.opt_worker.set_rendered_image(image_path)

    def on_optimization_finished(self, final_nodes):
        self.statusBar().showMessage("Optimization Completed.")
        QMessageBox.information(self, "Success", "최적화가 완료되었습니다.")
        # 최종 TF 적용은 이미 handle_sync_render_request의 마지막 호출로 되어있음

    # [NEW] Segmentation 로드 핸들러
    def on_segmentation_loaded(self, seg_data):
        """Segmentation 데이터 저장"""
        self.segmentation_data = seg_data
        print(f"✅ Segmentation stored in MainWindow: shape={seg_data.shape}")

    # [NEW] Visibility-based Optimization 시작
    def start_visibility_optimization_process(self, params):
        """Visibility-based TF optimization 시작 (comparison method)"""
        
        # 1. Segmentation 체크
        if self.segmentation_data is None:
            QMessageBox.warning(self, "Error", 
                "Segmentation mask가 로드되지 않았습니다.\n"
                "먼저 '🧩 Load Segmentation' 버튼으로 segmentation을 로드해주세요.")
            return
        
        if self.volume_data is None:
            QMessageBox.warning(self, "Error", "볼륨 데이터가 로드되지 않았습니다.")
            return
        
        # 2. 데이터 준비
        volume_data = self.volume_data
        spacing = self.voxel_spacing
        initial_nodes = self.tf_panel.tf_widget.get_nodes()
        
        # Camera 정보 가져오기
        camera = self.rendering_panel.vtk_renderer.renderer.GetActiveCamera()
        camera_info = {
            'position': camera.GetPosition(),
            'focal_point': camera.GetFocalPoint(),
            'view_up': camera.GetViewUp(),
            'view_angle': camera.GetViewAngle()
        }
        
        vol_name = getattr(self, 'current_volume_name', 'UnknownVolume')
        vol_name = vol_name.replace(" ", "_").split('.')[0]
        
        # 3. Worker & Thread 생성
        self.opt_thread = QThread()
        
        # Visibility 모드에서는 SAM 관련 데이터 불필요 → dummy 값 전달
        self.opt_worker = OptimizationWorker(
            optimizer=None,
            sam_wrapper=None,
            gt_mask=None,
            gt_text_mask=None,
            points_2d=None,
            projected_2d_points=[],
            initial_3d_points=[],
            text_points_2d=None,
            projected_text_2d_points=[],
            clipping_ranges=self.tf_panel.get_clipping_ranges()
        )
        
        # 4. Visibility Optimizer 초기화
        try:
            self.opt_worker.set_visibility_optimizer(
                volume_data=volume_data,
                spacing=spacing,
                initial_nodes=initial_nodes,
                camera_info=camera_info,
                segmentation_mask=self.segmentation_data,
            )
            
            self.opt_worker.use_visibility = True
            self.opt_worker.optimization_method_name = "Visibility-based"
            self.opt_worker.volume_name = vol_name
            
            print(f"✅ Visibility Optimizer initialized.")
            
        except Exception as e:
            print(f"❌ Visibility Optimizer initialization failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Visibility Optimizer 초기화 실패:\n{e}")
            return
        
        # 5. Thread 연결 & 시작
        self.opt_worker.moveToThread(self.opt_thread)
        
        self.opt_thread.started.connect(self.opt_worker.run)
        self.opt_worker.finished.connect(self.opt_thread.quit)
        self.opt_worker.finished.connect(self.opt_worker.deleteLater)
        self.opt_thread.finished.connect(self.opt_thread.deleteLater)
        
        self.opt_worker.request_render_sync.connect(self.handle_sync_render_request)
        self.opt_worker.result_ready.connect(self.on_optimization_finished)
        
        self.statusBar().showMessage("Running Visibility-based Optimization...")
        self.opt_thread.start()


    def closeEvent(self, e):
        self.sam.shutdown()
        super().closeEvent(e)

    # --- 볼륨 핸들러들 ---
    def on_volume_loaded(self, volume_data):
        self.volume_data = volume_data
        self.voxel_spacing = self.file_panel.voxel_spacing
        self.rendering_panel.vtk_renderer.voxel_spacing = self.voxel_spacing
        self.rendering_panel.set_volume_data(volume_data)
        self.tf_panel.set_volume_data(volume_data)
        self.tf_panel.reset_clipping_safe()
        self.current_volume_name = self.file_panel.volume_name

    def on_tf_changed(self, tf_array):
        self.rendering_panel.update_transfer_function(tf_array)
    
    def on_background_color_changed(self, color1, color2):
        self.rendering_panel.set_background_color(color1, color2)
    
    def on_shading_changed(self, enabled):
        self.rendering_panel.set_shading(enabled)
    
    def on_lighting_changed(self, property_type, value):
        self.rendering_panel.set_lighting_property(property_type, value)

    def on_light_direction_changed(self, x, y, z):
        if self.rendering_panel.vtk_renderer:
            self.rendering_panel.vtk_renderer.set_light_position('key', x, y, z)

    def on_follow_camera_changed(self, enabled):
        self.rendering_panel.vtk_renderer.set_follow_camera(enabled)

    def on_ambient_color_changed(self, r, g, b):
        self.rendering_panel.vtk_renderer.set_ambient_color(r, g, b)

    def on_diffuse_color_changed(self, r, g, b):
        self.rendering_panel.vtk_renderer.set_diffuse_color(r, g, b)

    def on_specular_color_changed(self, r, g, b):
        self.rendering_panel.vtk_renderer.set_specular_color(r, g, b)
        
    def on_clipping_changed(self, axis, min_val, max_val):
        self.rendering_panel.apply_clipping(axis, min_val, max_val)
    
    def on_clipping_enabled_changed(self, enabled):
        self.rendering_panel.set_clipping_enabled(enabled)

    def closeEvent(self, event):
        """애플리케이션 종료 시 안전한 정리"""
        # 1. 최적화 스레드가 실행 중이면 중단 요청
        if hasattr(self, 'opt_thread') and self.opt_thread.isRunning():
            self.opt_thread.quit()
            if not self.opt_thread.wait(3000):  # 3초 대기
                self.opt_thread.terminate()
                self.opt_thread.wait()
        
        # 2. SAM 서비스 종료 (QThread 정리)
        if hasattr(self, 'sam_model'):
            self.sam_model.shutdown()
        
        # 3. VTK 렌더링 정리
        if hasattr(self, 'rendering_panel'):
            self.rendering_panel.cleanup()
        super().closeEvent(event)


    def on_grid_samples_ready(self, grid_3d_points):
        """Grid sampling된 포인트들을 파란색으로 3D 시각화"""
        print(f"🔵 Visualizing {len(grid_3d_points)} grid sample points in blue")
        
        for p in grid_3d_points:
            self.rendering_panel.add_point_3d(p, "grid")  # "grid" 타입으로 구분