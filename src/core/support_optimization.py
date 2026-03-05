"""
Optimization Worker for Transfer Function Optimization

백그라운드에서 최적화 루프를 실행하는 워커.
렌더링이 필요할 때만 메인 스레드와 동기화(Sync)합니다.
"""

import os
import numpy as np
import PIL.Image
from PyQt6.QtCore import QObject, pyqtSignal, QMutex, QWaitCondition
from datetime import datetime

# 공통 유틸리티 import
try:
    from utils.common import tf_nodes_to_opacity_lut, sample_grid_representative_points, project_points_to_screen
except ImportError:
    from .utils.common import tf_nodes_to_opacity_lut, sample_grid_representative_points, project_points_to_screen


class OptimizationWorker(QObject):
    """
    백그라운드에서 최적화 루프를 실행하는 워커.
    """
    finished = pyqtSignal()
    result_ready = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    request_render_sync = pyqtSignal(list)  # arg: new_tf_nodes

    grid_samples_ready = pyqtSignal(list)

    def __init__(self, optimizer, sam_wrapper, gt_mask, gt_text_mask, 
                points_2d, projected_2d_points, initial_3d_points,
                text_points_2d, projected_text_2d_points, clipping_ranges):
        super().__init__()
        self.optimizer = optimizer
        self.sam = sam_wrapper
        self.gt_mask = gt_mask
        self.gt_text_mask = gt_text_mask
        self.points_2d = points_2d
        self.projected_2d_points = projected_2d_points
        self.feature_analyzer = None 
        self.initial_3d_points = initial_3d_points
        self.initial_ray_directions = None
        self.initial_intensities = None
        self.text_points_2d = text_points_2d
        self.projected_text_2d_points = projected_text_2d_points
        self.clipping_ranges = clipping_ranges

        self.save_dir = "./resources/optimization_images/"
        self.optimization_method_name = "Unknown"
        self.volume_name = "Volume"

        os.makedirs(self.save_dir, exist_ok=True)
        
        # 결과 폴더 변수들
        self.result_base_dir = None
        self.rendering_dir = None
        self.sam_masks_dir = None
        self.tf_graphs_dir = None
        self.points_overlay_dir = None
        self.ray_profiles_dir = None
        self.iteration_counter = 0

        # 동기화 도구
        self.sync_mutex = QMutex()
        self.sync_condition = QWaitCondition()
        self.latest_image_path = None
        self.converged = False

        # PyTorch 옵션
        self.use_pytorch = False
        self.pytorch_optimizer = None
        
        # [NEW] Visibility-based 옵션
        self.use_visibility = False
        self.visibility_optimizer = None

    def set_pytorch_optimizer(self, volume_data, spacing, initial_nodes, camera_info,
                            feature_analyzer=None, use_vtk_rendering=True,
                            vtk_resolution=None, target_range=None,
                            initial_ray_directions=None, color_reference=None, gt_mask=None,
                            point_certainty_weights=None, sam_confidence=None):

        """PyTorch Optimizer 설정"""
        from src.core.diff_optimizer import DiffOptimizer
        import torch

        # M1/M2 GPU 지원
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        print(f"🔧 Optimizer using device: {device}")

        self.pytorch_optimizer = DiffOptimizer(
            volume_data=volume_data,
            spacing=spacing,
            initial_nodes=initial_nodes,
            camera_info=camera_info,
            sam_wrapper=self.sam,
            projection_points_3d=self.initial_3d_points,
            device=device,
            feature_analyzer=feature_analyzer,
            use_vtk_rendering=use_vtk_rendering,
            vtk_resolution=vtk_resolution,
            target_range=target_range,
            initial_ray_directions=initial_ray_directions,
            clipping_ranges=self.clipping_ranges,
            color_reference=color_reference,
            gt_mask=gt_mask,
            point_certainty_weights=point_certainty_weights,
            sam_confidence=sam_confidence,
        )

    def set_visibility_optimizer(self, volume_data, spacing, initial_nodes, camera_info,
                                  segmentation_mask):
        """[NEW] Visibility-based Optimizer 설정"""
        from src.core.visibility_optimizer import VisibilityOptimizer
        import torch

        # Device 선택
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        print(f"🔧 Visibility Optimizer using device: {device}")

        self.visibility_optimizer = VisibilityOptimizer(
            volume_data=volume_data,
            spacing=spacing,
            initial_nodes=initial_nodes,
            camera_info=camera_info,
            segmentation_mask=segmentation_mask,
            device=device,
        )

    def setup_result_folders(self):
        """Optimization 시작 시 결과 저장 폴더 구조 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{timestamp}_{self.optimization_method_name}_{self.volume_name}"
        self.result_base_dir = f"./resources/results/{folder_name}"
        
        self.rendering_dir = os.path.join(self.result_base_dir, "rendering")
        self.sam_masks_dir = os.path.join(self.result_base_dir, "sam_masks")
        self.tf_graphs_dir = os.path.join(self.result_base_dir, "tf_graphs")
        self.points_overlay_dir = os.path.join(self.result_base_dir, "points_overlay")
        self.ray_profiles_dir = os.path.join(self.result_base_dir, "ray_profiles")  # NEW
        
        for folder in [self.rendering_dir, self.sam_masks_dir, 
                        self.tf_graphs_dir, self.points_overlay_dir,
                        self.ray_profiles_dir]:  # ray_profiles 추가
            os.makedirs(folder, exist_ok=True)
        
        print(f"📁 Result folders created: {self.result_base_dir}")
        self.iteration_counter = 0
        return self.result_base_dir

    def save_reprojected_gt_mask(self, renderer):
        """GT Mask를 현재 view에 reprojection하여 저장"""
        if self.result_base_dir is None:
            return
        
        try:
            render_window = renderer.vtk_widget.GetRenderWindow()
            projection_results = project_points_to_screen(
                self.initial_3d_points, renderer.renderer, render_window
            )
            
            if not projection_results:
                print("⚠️ No points to reproject for GT mask")
                return
            
            pts_2d = [p[1] for p in projection_results]
            width, height = render_window.GetSize()
            
            reprojected_mask = np.zeros((height, width), dtype=np.uint8)
            
            for x, y in pts_2d:
                if 0 <= x < width and 0 <= y < height:
                    # 반지름 1로 줄이기
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dx * dx + dy * dy <= 1:

                                nx, ny = x + dx, y + dy
                                if 0 <= nx < width and 0 <= ny < height:
                                    reprojected_mask[ny, nx] = 255
            
            save_path = os.path.join(self.result_base_dir, "reprojected_gt_mask.png")
            PIL.Image.fromarray(reprojected_mask).save(save_path)
            print(f"💾 Reprojected GT mask saved: {save_path}")
            
        except Exception as e:
            print(f"❌ Failed to save reprojected GT mask: {e}")
            import traceback
            traceback.print_exc()

    def save_iteration_results(self, image_path, full_mask, visible_mask, 
                                    new_nodes, visible_samples, all_samples,
                                    visibility_results=None):
        """각 iteration 결과 저장"""
        if self.result_base_dir is None:
            return
        
        self.iteration_counter += 1
        iter_num = self.iteration_counter
        
        try:
            # 1. Rendering 이미지 복사
            if image_path and os.path.exists(image_path):
                import shutil
                rendering_dest = os.path.join(self.rendering_dir, f"iter_{iter_num:03d}.png")
                shutil.copy(image_path, rendering_dest)
            
            # 2. SAM masks 저장
            if full_mask is not None:
                full_mask_img = PIL.Image.fromarray((full_mask * 255).astype(np.uint8))
                full_mask_img.save(os.path.join(self.sam_masks_dir, f"full_mask_iter_{iter_num:03d}.png"))
            
            if visible_mask is not None:
                visible_mask_img = PIL.Image.fromarray((visible_mask * 255).astype(np.uint8))
                visible_mask_img.save(os.path.join(self.sam_masks_dir, f"visible_mask_iter_{iter_num:03d}.png"))
            
            # 3. TF Graph 저장
            self._save_tf_graph(new_nodes, iter_num)
            
            # 4. Points Overlay 저장
            self._save_points_overlay(image_path, visible_samples, all_samples, iter_num, visibility_results)
            
            # 5. Ray Profile 저장 (NEW)
            # self._save_ray_profiles(all_samples, visibility_results, iter_num)
            
        except Exception as e:
            print(f"⚠️ Failed to save iteration {iter_num} results: {e}")

    def _save_ray_profiles(self, all_samples, visibility_results, iter_num):
        """Ray Profile 시각화 저장"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        # pytorch_optimizer에서 디버그 데이터 가져오기
        if not hasattr(self, 'pytorch_optimizer') or self.pytorch_optimizer is None:
            return
        
        debug_data_list = getattr(self.pytorch_optimizer, 'last_debug_data_list', None)
        if debug_data_list is None or len(debug_data_list) == 0:
            return
        
        grid_samples = getattr(self.pytorch_optimizer, 'last_grid_samples', [])
        if not grid_samples:
            return
        
        # visibility_flags 재구성
        visibility_flags = []
        for sample in grid_samples:
            idx = sample['idx']
            if visibility_results and idx in visibility_results:
                is_vis, _ = visibility_results[idx]
                visibility_flags.append(is_vis)
            else:
                visibility_flags.append(True)  # 기본값
        
        num_samples = len(grid_samples)
        if num_samples == 0:
            return
        
        # Grid layout 계산
        cols = min(4, num_samples)
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        if num_samples == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (sample, debug_data, is_visible) in enumerate(zip(grid_samples, debug_data_list, visibility_flags)):
            row, col = i // cols, i % cols
            ax = axes[row, col]
            
            idx = sample['idx']
            pt = sample['pt']
            
            # Early exit case
            if 'early_exit_reason' in debug_data:
                ax.text(0.5, 0.5, f"Early Exit:\n{debug_data['early_exit_reason']}", 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f"Sample #{i} (idx={idx})\n2D: {pt}", fontsize=10)
                continue
            
            t_values = debug_data.get('t_values', np.array([]))
            intensities = debug_data.get('intensities', np.array([]))
            raw_opacities = debug_data.get('raw_opacities', np.array([]))
            effective_opacities = debug_data.get('effective_opacities', np.array([]))
            cumulative_opacity = debug_data.get('cumulative_opacity', np.array([]))
            is_target_tissue = debug_data.get('is_target_tissue', np.array([]))
            is_clipped = debug_data.get('is_clipped', np.array([]))
            opacity_threshold = debug_data.get('opacity_threshold', 0.99)
            
            if len(t_values) == 0:
                ax.text(0.5, 0.5, "No samples", ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"Sample #{i} (idx={idx})", fontsize=10)
                continue
            
            # Plot lines
            ax.plot(t_values, intensities, 'gray', linewidth=1, alpha=0.7, label='Intensity')
            ax.plot(t_values, raw_opacities, 'b-', linewidth=1.5, alpha=0.7, label='Raw Opacity')
            ax.plot(t_values, effective_opacities, 'orange', linewidth=2, label='Effective Opacity')
            ax.plot(t_values, cumulative_opacity, 'r-', linewidth=2, label='Cumulative Opacity')
            
            # Threshold line
            ax.axhline(y=opacity_threshold, color='red', linestyle='--', alpha=0.5, 
                      label=f'Threshold ({opacity_threshold})')
            
            # Shade target tissue skip regions (녹색)
            if len(is_target_tissue) > 0 and np.any(is_target_tissue):
                for j in range(len(t_values)):
                    if is_target_tissue[j]:
                        ax.axvspan(t_values[max(0, j-1)], t_values[min(len(t_values)-1, j+1)], 
                                  alpha=0.2, color='green')
            
            # Shade clipped regions (보라색)
            if len(is_clipped) > 0 and np.any(is_clipped):
                for j in range(len(t_values)):
                    if is_clipped[j]:
                        ax.axvspan(t_values[max(0, j-1)], t_values[min(len(t_values)-1, j+1)], 
                                  alpha=0.2, color='purple')
            
            # Title with visibility info
            final_acc = cumulative_opacity[-1] if len(cumulative_opacity) > 0 else 0
            visibility_str = "VISIBLE" if is_visible else "OCCLUDED"
            title_color = 'green' if is_visible else 'red'
            
            # TF Color Bar 추가 (X축 아래)
            colors_rgb = debug_data.get('colors', None)
            if colors_rgb is not None and len(colors_rgb) > 0:
                # 컬러바 높이 영역 (-0.15 ~ -0.05)
                for j in range(len(t_values) - 1):
                    ax.axvspan(t_values[j], t_values[j+1], 
                              ymin=0, ymax=0.06,  # 축 기준 비율 (하단 6%)
                              facecolor=colors_rgb[j], 
                              alpha=0.9,
                              edgecolor='none')
            
            ax.set_xlabel('Distance along ray')
            ax.set_ylabel('Value (0-1)')
            ax.set_ylim(-0.08, 1.1)  # 컬러바 공간 확보를 위해 약간 더 아래로
            ax.set_title(f"Sample #{i} (idx={idx}) | {visibility_str}\n"
                        f"Acc Opacity: {final_acc:.3f} | 2D: ({int(pt[0])}, {int(pt[1])})", 
                        fontsize=9, color=title_color)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper left', fontsize=7)
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(self.ray_profiles_dir, f"ray_profiles_iter_{iter_num:03d}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()
        
        print(f"   📊 Ray profiles saved: {save_path}")

    def _save_tf_graph(self, nodes, iter_num):
        """TF Graph 시각화 저장 - 색상별 filled tent 스타일"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        try:
            sorted_nodes = sorted(nodes, key=lambda x: x[0])

            # 고해상도 LUT (opacity + color)
            lut_size = 512
            x_vals = np.linspace(0, 255, lut_size)
            opacity_vals = np.zeros(lut_size)
            color_vals = np.zeros((lut_size, 3))

            for i in range(lut_size):
                intensity = x_vals[i] / 255.0
                if intensity <= sorted_nodes[0][0]:
                    opacity_vals[i] = sorted_nodes[0][4]
                    color_vals[i] = sorted_nodes[0][1:4]
                elif intensity >= sorted_nodes[-1][0]:
                    opacity_vals[i] = sorted_nodes[-1][4]
                    color_vals[i] = sorted_nodes[-1][1:4]
                else:
                    for j in range(len(sorted_nodes) - 1):
                        if sorted_nodes[j][0] <= intensity <= sorted_nodes[j + 1][0]:
                            denom = sorted_nodes[j + 1][0] - sorted_nodes[j][0]
                            t = (intensity - sorted_nodes[j][0]) / denom if denom > 0 else 0
                            opacity_vals[i] = sorted_nodes[j][4] * (1 - t) + sorted_nodes[j + 1][4] * t
                            for c in range(3):
                                color_vals[i, c] = sorted_nodes[j][1 + c] * (1 - t) + sorted_nodes[j + 1][1 + c] * t
                            break

            # --- 그리기 ---
            fig, ax = plt.subplots(figsize=(10, 3), facecolor='white')
            ax.set_facecolor('white')

            # 각 노드 쌍 구간을 polygon으로 깔끔하게 채우기 (빈틈 제거 방식 적용)
            for j in range(len(sorted_nodes) - 1):
                x_start = sorted_nodes[j][0] * 255
                x_end = sorted_nodes[j + 1][0] * 255
                
                y_start = sorted_nodes[j][4]
                y_end = sorted_nodes[j + 1][4]

                # 두 노드의 opacity가 모두 0에 가까우면 그릴 필요 없음
                if y_start < 0.001 and y_end < 0.001:
                    continue

                # 구간 대표 색상: 두 노드 중 Opacity(y값)가 더 높은 쪽의 색상을 사용 (기존 peak_idx 로직 대체)
                if y_start >= y_end:
                    seg_color = [max(0, min(1, c)) for c in sorted_nodes[j][1:4]]
                else:
                    seg_color = [max(0, min(1, c)) for c in sorted_nodes[j+1][1:4]]

                # 512개 배열을 마스킹하지 않고, 정확한 시작점과 끝점을 이용해 사각형/사다리꼴 생성
                poly_x = [x_start, x_end, x_end, x_start]
                poly_y = [y_start, y_end, 0, 0]

                # 핵심 해결책: edgecolor를 seg_color로 동일하게 맞추고 약간의 두께(linewidth)를 주어 경계선 틈새를 메움
                ax.fill(poly_x, poly_y, color=seg_color, alpha=0.9, edgecolor=seg_color, linewidth=0.4)

            # 전체 opacity outline (검정 선)
            ax.plot(x_vals, opacity_vals, color='black', linewidth=1.0, alpha=0.6)

            ax.set_xlim(0, 255)
            ax.set_ylim(0, min(1.0, np.max(opacity_vals) * 1.2 + 0.05))
            ax.set_title(f'Transfer Function — Iteration {iter_num}', fontsize=10, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # 네모 테두리
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#000000")
                spine.set_linewidth(2.0)
                
            plt.tight_layout()

            save_path = os.path.join(self.tf_graphs_dir, f"iter_{iter_num:03d}.png")
            plt.savefig(save_path, dpi=150, facecolor='white', bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ TF graph save failed: {e}")

    def _save_points_overlay(self, image_path, visible_samples, all_samples, iter_num, visibility_results=None):
        """렌더링 이미지 위에 visible(초록)/invisible(빨강) 포인트 + 누적 opacity 표시"""
        from PIL import ImageDraw, ImageFont
        
        try:
            if not image_path or not os.path.exists(image_path):
                return
            
            img = PIL.Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()
            
            visible_indices = set([s['idx'] for s in visible_samples]) if visible_samples else set()
            
            for sample in all_samples:
                pt = sample['pt']
                idx = sample['idx']
                x, y = int(pt[0]), int(pt[1])
                
                is_visible = idx in visible_indices
                color = (0, 255, 0) if is_visible else (255, 0, 0)
                radius = 2
                
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                            fill=color, outline=(255, 255, 255), width=2)
                
                if visibility_results and idx in visibility_results:
                    _, acc_opacity = visibility_results[idx]
                    opacity_text = f"{acc_opacity:.2f}"
                    text_x = x + radius + 3
                    text_y = y - radius - 5
                    draw.rectangle([text_x - 1, text_y - 1, text_x + 30, text_y + 12], fill=(0, 0, 0, 180))
                    draw.text((text_x, text_y), opacity_text, fill=(255, 255, 255), font=font)
            
            save_path = os.path.join(self.points_overlay_dir, f"iter_{iter_num:03d}.png")
            img.save(save_path)
            
        except Exception as e:
            print(f"⚠️ Points overlay save failed: {e}")

    def run(self):
        import time
        start_time = time.time()
        try:
            self.setup_result_folders()
            
            # [NEW] Visibility-based 모드
            if self.use_visibility:
                if not self.visibility_optimizer:
                    raise ValueError("Visibility Optimizer has not been initialized.")
                
                if hasattr(self.visibility_optimizer, 'initial_nodes'):
                    self._save_tf_graph(self.visibility_optimizer.initial_nodes, 0)
                
                final_nodes = self._run_visibility_optimization()
                self.result_ready.emit(final_nodes)
                return
            
            if self.pytorch_optimizer and hasattr(self.pytorch_optimizer, 'initial_nodes'):
                initial_lut_nodes = self.pytorch_optimizer.tf.to_nodes_direct()
                self._save_tf_graph(initial_lut_nodes, 0)

            if hasattr(self, 'renderer_ref') and self.renderer_ref:
                self.save_reprojected_gt_mask(self.renderer_ref)
            
            # [수정됨] DiffOptimizer 인스턴스가 생성되어 있어야 함
            if not self.pytorch_optimizer:
                raise ValueError("Optimizer has not been initialized. Call set_pytorch_optimizer first.")

            # Adam (PyTorch Loop) vs Nelder-Mead 분기
            if self.use_pytorch: 
                final_nodes = self._run_pytorch_optimization() # Adam 방식
            else:
                final_nodes = self._run_nelder_mead_optimization() # Nelder-Mead 방식
            
            self.result_ready.emit(final_nodes)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            elapsed = time.time() - start_time
            print(f"⏱️ Total optimization time: {elapsed*1000:.0f}ms ({elapsed:.2f}s)")
            self.finished.emit()

    def _run_nelder_mead_optimization(self):
        """Nelder-Mead 최적화 실행 (GUI 단순 동기화)"""
        print("🚀 Starting NELDER-MEAD optimization...")

        # Nelder-Mead의 매 스텝마다 실행될 콜백 함수 정의
        def step_callback(info):
            # 1. 정보 추출
            new_nodes = info['nodes']
            iteration = info['iteration']
            loss_value = info['loss']
            image_path = info.get('image_path') # PyTorch가 저장한 이미지 경로
            full_mask = info.get('full_mask')
            visible_mask = info.get('visible_mask')
            
            # 2. GUI 업데이트 (Render Sync)
            # 메인 스레드에 "이 TF로 화면 좀 그려줘"라고 요청
            self.sync_mutex.lock()
            self.latest_image_path = None
            self.request_render_sync.emit(new_nodes)
            
            # 메인 스레드가 그릴 때까지 잠깐 대기 (화면 갱신용)
            # 1초만 기다려보고 안 되면 그냥 넘어감 (계산은 이미 끝났으므로)
            is_signaled = self.sync_condition.wait(self.sync_mutex, 1000)
            self.sync_mutex.unlock()
            
            # 3. 결과 저장 (시각적 확인용)
            # DiffOptimizer가 이미 이미지를 저장했으므로 그 경로를 사용
            if image_path and os.path.exists(image_path):
                self.save_iteration_results(
                    image_path=image_path,
                    full_mask=full_mask,     # 속도를 위해 매번 저장하지 않음 (필요시 추가)
                    visible_mask=visible_mask,
                    new_nodes=new_nodes,
                    visible_samples=[], # 시각화 생략 (속도)
                    all_samples=[],
                    visibility_results={}
                )
            
            # [중요] 리턴값 없음
            # 이제 Loss 계산은 DiffOptimizer가 알아서 하므로, Worker는 값을 리턴할 필요가 없습니다.
            return 

        # ---------------------------------------------------------
        
        # 최적화 실행 (콜백 전달)
        final_nodes = self.pytorch_optimizer.optimize(
            num_iterations=100,
            method='nelder-mead',
            callback=step_callback  # <--- 단순화된 콜백 전달
        )
        
        return final_nodes
    

    def _run_pytorch_optimization(self):
        """PyTorch 최적화 실행 (VTK 렌더링 동기화 지원)"""
        num_iterations = 100
        
        for i in range(num_iterations):
            # VTK 렌더링 사용 시 이미지 먼저 요청
            if self.pytorch_optimizer.use_vtk_rendering:
                current_nodes = self.pytorch_optimizer.get_current_nodes()
                
                self.sync_mutex.lock()
                self.latest_image_path = None
                self.request_render_sync.emit(current_nodes)
                is_signaled = self.sync_condition.wait(self.sync_mutex, 5000)
                image_path = self.latest_image_path
                self.sync_mutex.unlock()
                
                if not is_signaled or not image_path:
                    print(f"⚠️ VTK rendering timeout at iteration {i + 1}")
                    continue
                
                self.pytorch_optimizer.set_external_image(image_path)
            
            result = self.pytorch_optimizer.optimize_step()
            
            if result.get('waiting_for_render', False):
                print(f"⏳ Waiting for VTK rendering at iteration {i + 1}...")
                continue
            
            print(f"[PyTorch] Iteration {i + 1}/{num_iterations}: Loss={result['loss']:.4f}")
                
            # 시각화 실험
            # # [NEW] 첫 iteration에서 grid samples 3D 시각화 후 중단
            # if i == 0:
            #     if hasattr(self.pytorch_optimizer, 'last_grid_samples') and self.pytorch_optimizer.last_grid_samples:
            #         grid_3d_points = []
            #         for sample in self.pytorch_optimizer.last_grid_samples:
            #             idx = sample['idx']
            #             if idx < len(self.initial_3d_points):
            #                 grid_3d_points.append(self.initial_3d_points[idx])
                    
            #         print(f"🔵 Emitting {len(grid_3d_points)} grid sample 3D points")
            #         self.grid_samples_ready.emit(grid_3d_points)
                
            #     print("⏹️ Stopping after first iteration for visualization")
            #     break


            # iteration 결과 저장
            if hasattr(self.pytorch_optimizer, 'last_full_mask'):
                self.save_iteration_results(
                    image_path=self.latest_image_path if self.pytorch_optimizer.use_vtk_rendering else result.get('image_path'),
                    full_mask=self.pytorch_optimizer.last_full_mask,
                    visible_mask=self.pytorch_optimizer.last_visible_mask,
                    new_nodes=result['nodes'],
                    visible_samples=self.pytorch_optimizer.last_visible_samples,
                    all_samples=self.pytorch_optimizer.last_grid_samples,
                    visibility_results=getattr(self.pytorch_optimizer, 'last_visibility_results', None)
                )

            if result.get('converged', False):
                print(f"✅ Early stopping at iteration {i + 1}")
                break
            
            if (i + 1) % 10 == 0 and not self.pytorch_optimizer.use_vtk_rendering:
                self.request_render_sync.emit(result['nodes'])
        
        return self.pytorch_optimizer.get_best_nodes()
    

    def _run_visibility_optimization(self):
        """[NEW] Visibility-based 최적화 실행"""
        print("🚀 Starting VISIBILITY-BASED optimization...")
        
        num_iterations = 100
        
        # Visibility map 저장 경로를 optimizer에 전달
        vis_map_dir = os.path.join(self.result_base_dir, "visibility_maps")
        os.makedirs(vis_map_dir, exist_ok=True)
        self.visibility_optimizer.vis_map_save_dir = vis_map_dir
        
        def step_callback(result):
            new_nodes = result['nodes']
            iteration = result['iteration']
            
            # GUI 업데이트 (Render Sync) → 렌더링 이미지도 캡처됨
            self.sync_mutex.lock()
            self.latest_image_path = None
            self.request_render_sync.emit(new_nodes)
            is_signaled = self.sync_condition.wait(self.sync_mutex, 2000)
            image_path = self.latest_image_path
            self.sync_mutex.unlock()
            
            # 렌더링 이미지 저장
            if image_path and os.path.exists(image_path):
                import shutil
                rendering_dest = os.path.join(self.rendering_dir, f"iter_{iteration:03d}.png")
                shutil.copy(image_path, rendering_dest)
            
            # TF graph 저장
            self._save_tf_graph(new_nodes, iteration)
        
        final_nodes = self.visibility_optimizer.optimize(
            num_iterations=num_iterations,
            callback=step_callback
        )
        
        return final_nodes

    def set_rendered_image(self, path):
        """메인 스레드가 렌더링 완료 후 호출하는 함수"""
        self.sync_mutex.lock()
        self.latest_image_path = path
        self.sync_condition.wakeAll()
        self.sync_mutex.unlock()