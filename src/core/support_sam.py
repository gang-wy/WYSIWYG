# src/core/support_sam.py
from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt
import traceback
import sys

# [수정 1] 함수 내부가 아닌 여기서 미리 import 하여 메인 컨텍스트에서 로드되도록 함
# PyTorch 등 C++ 확장은 서브 스레드에서 처음 로드하면 충돌 위험이 큼
from src.core.sam_wrapper import SAMWrapper 

class _SAMWorker(QObject):
    loaded = pyqtSignal()
    predicted = pyqtSignal(object)
    error = pyqtSignal(str)
    
    # [NEW] 텍스트 예측 결과 시그널
    text_predicted = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self._sam = None # 초기화 명시

    @pyqtSlot()
    def load(self):
        try:
            # [수정 2] 이미 import된 클래스 사용
            if self._sam is None:
                self._sam = SAMWrapper()
            
            # 모델 로드 (시간이 걸리므로 스레드에서 수행하는 것은 맞음)
            self._sam.load_model()
            self.loaded.emit()
        except Exception:
            self.error.emit(traceback.format_exc())

    @pyqtSlot(str, object, object)
    def predict(self, image_path, points, labels):
        try:
            if self._sam is None:
                self._sam = SAMWrapper()
                self._sam.load_model()

            if not self._sam.is_loaded:
                self._sam.load_model()

            mask, logits = self._sam.predict(image_path, points, labels)
            self.predicted.emit((mask, logits))
        except Exception:
            self.error.emit(traceback.format_exc())

    @pyqtSlot(str, str)
    def predict_text(self, image_path, text_prompt):
        """[NEW] 텍스트 프롬프트 예측 슬롯"""
        try:
            mask = self._sam.predict_text(image_path, text_prompt)
            # 텍스트 예측 전용 시그널 발송
            self.text_predicted.emit(mask)
        except Exception:
            self.error.emit(traceback.format_exc())


class SAMService(QObject):
    # UI가 받는 시그널
    loaded = pyqtSignal()
    predicted = pyqtSignal(object)
    error = pyqtSignal(str)

    # [NEW] 텍스트 예측 결과 시그널 (UI 연결용)
    text_predicted = pyqtSignal(object)

    # UI -> worker 트리거용 시그널
    _req_load = pyqtSignal()
    _req_predict = pyqtSignal(str, object, object)
    # [NEW] 텍스트 예측 요청 시그널
    _req_predict_text = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._thread = QThread(self)
        self._worker = _SAMWorker()
        self._worker.moveToThread(self._thread)

        # 트리거 시그널을 worker 슬롯에 연결 (QueuedConnection)
        self._req_load.connect(self._worker.load, type=Qt.ConnectionType.QueuedConnection)
        self._req_predict.connect(self._worker.predict, type=Qt.ConnectionType.QueuedConnection)
        # [NEW] 텍스트 예측 연결
        self._req_predict_text.connect(self._worker.predict_text, type=Qt.ConnectionType.QueuedConnection)

        # worker 결과를 외부로 전달
        self._worker.loaded.connect(self.loaded)
        self._worker.predicted.connect(self.predicted)
        self._worker.error.connect(self.error)

        # [NEW] 텍스트 결과 연결
        self._worker.text_predicted.connect(self.text_predicted)
        self._worker.error.connect(self.error)

        self._thread.start()

    def shutdown(self):
        """안전한 스레드 종료"""
        print("🔄 Requesting SAM thread termination...")
        
        # 스레드에 종료 요청
        self._thread.quit()
        
        # 최대 3초 대기
        if not self._thread.wait(3000):
            print("⚠️ SAM thread did not quit gracefully, forcing termination")
            self._thread.terminate()
            self._thread.wait()
        
        print("✅ SAM thread terminated")

    def load_async(self):
        self._req_load.emit()

    def predict_async(self, image_path, points, labels):
        self._req_predict.emit(image_path, points, labels)

    def predict_text_async(self, image_path, text_prompt):
        """[NEW] 외부에서 호출하는 비동기 텍스트 예측 함수"""
        self._req_predict_text.emit(image_path, text_prompt)