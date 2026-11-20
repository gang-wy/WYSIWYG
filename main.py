"""앱 진입점: 실행 환경 변수 설정 후 Fusion 다크 테마로 메인 윈도우를 띄운다."""
import sys
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Windows는 forkserver 미지원
if sys.platform == "win32":
    os.environ["JOBLIB_START_METHOD"] = "loky"
else:
    os.environ["JOBLIB_START_METHOD"] = "forkserver"

# VTK 렌더링 백엔드 (OSMesa 소프트웨어 렌더링)
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.2'
os.environ['VTK_USE_OSMESA'] = '1'

def main():
    """QApplication과 메인 윈도우를 만들고 이벤트 루프를 돈다. Returns: 종료 코드"""
    clear_python_cache()
    os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QPalette, QColor

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 다크 테마
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))
    app.setPalette(palette)

    from src.main_window import VolumeRenderingMainWindow
    window = VolumeRenderingMainWindow()
    window.show()

    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        print("\n⚠️ Keyboard interrupt detected")
        exit_code = 0
    finally:
        import gc
        gc.collect()

    return exit_code

def clear_python_cache():
    """현재 폴더 아래의 __pycache__ 폴더와 .pyc 파일을 지운다."""
    import shutil
    import glob
    
    try:
        cache_dirs = []
        for root, dirs, files in os.walk('.'):
            for dir_name in dirs:
                if dir_name == '__pycache__':
                    cache_path = os.path.join(root, dir_name)
                    cache_dirs.append(cache_path)
        
        for cache_dir in cache_dirs:
            try:
                shutil.rmtree(cache_dir)
            except Exception as e:
                pass
        
        pyc_files = glob.glob('**/*.pyc', recursive=True)
        for pyc_file in pyc_files:
            try:
                os.remove(pyc_file)
            except Exception as e:
                pass
        
    except Exception as e:
        print(f"❌ 캐시 정리 중 오류: {e}")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)