"""
Screenshot Manager
renderer_widget.py에서 분리됨
"""
import os
from datetime import datetime

try:
    import vtk
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False


## screenshot_manager.py 수정본

class ScreenshotManager:
    """렌더링 스크린샷 관리 클래스"""
    
    def __init__(self, renderer_widget):
        """
        Args:
            renderer_widget: VTKMultiVolumeRenderer 인스턴스
        """
        self.renderer_widget = renderer_widget
        # 자주 사용하는 속성들을 미리 참조해두면 코드가 깔끔해집니다.
        self.output_dir = "./resources/Rendered_Image"
    
    def save_current_rendering(self, use_square_ratio=False):
        """현재 렌더링 저장"""
        # 에러 발생 지점 수정: self.renderer -> self.renderer_widget.renderer
        if not VTK_AVAILABLE or self.renderer_widget.renderer is None:
            return None
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ratio_suffix = "_1to1" if use_square_ratio else "_current"
            filename = os.path.join(self.output_dir, f"render_{timestamp}{ratio_suffix}.png")
            
            # vtk_widget 접근 수정
            render_window = self.renderer_widget.vtk_widget.GetRenderWindow()
            current_size = render_window.GetSize()
            
            if use_square_ratio:
                max_size = max(current_size)
                render_window.SetSize(max_size, max_size)
                render_window.Render()
            
            # 스크린샷 촬영
            screenshot_filter = vtk.vtkWindowToImageFilter()
            screenshot_filter.SetInput(render_window)
            screenshot_filter.SetInputBufferTypeToRGBA()
            screenshot_filter.ReadFrontBufferOff()
            screenshot_filter.Update()
            
            # PNG 저장
            writer = vtk.vtkPNGWriter()
            writer.SetFileName(filename)
            writer.SetInputConnection(screenshot_filter.GetOutputPort())
            writer.Write()
            
            # 원본 크기로 복원
            if use_square_ratio:
                render_window.SetSize(current_size[0], current_size[1])
                render_window.Render()
            
            print(f"렌더링 저장: {filename}")
            return filename
            
        except Exception as e:
            print(f"렌더링 저장 실패: {e}")
            return None
    
    def export_screenshot(self, filename, resolution=(1920, 1080)):
        """고해상도 스크린샷 저장"""
        # 에러 발생 지점 수정
        if not VTK_AVAILABLE or self.renderer_widget.renderer is None:
            return False
        
        try:
            render_window = self.renderer_widget.vtk_widget.GetRenderWindow()
            original_size = render_window.GetSize()
            
            # 고해상도로 설정
            render_window.SetSize(resolution[0], resolution[1])
            render_window.Render()
            
            # 스크린샷 촬영 및 저장 로직 동일...
            # (중략)
            
            return True
        except Exception as e:
            print(f"스크린샷 저장 실패: {e}")
            return False