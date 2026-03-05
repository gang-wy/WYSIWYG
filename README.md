# SAM-TF-OPT: AI-Powered Transfer Function Optimization System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)
![VTK](https://img.shields.io/badge/VTK-9.0+-red.svg)
![License](https://img.shields.io/badge/License-Research-yellow.svg)

**SAM (Segment Anything Model)과 최적화 알고리즘을 활용한 3D 볼륨 렌더링 Transfer Function 자동 조정 도구**

[Features](#-핵심-기능) • [Installation](#-설치-방법) • [Usage](#-사용-방법) • [Architecture](#-시스템-아키텍처) • [Contributing](#-기여)

</div>

---

## 📌 Overview

의료 영상, 과학 시각화 등에서 사용되는 3D 볼륨 데이터의 Transfer Function(TF)을 자동으로 최적화하는 AI 기반 도구입니다. 사용자가 관심 영역(ROI)을 지정하면, SAM 모델과 최적화 알고리즘이 해당 영역을 최적으로 가시화하는 TF를 자동 생성합니다.

### 🎯 주요 특징
- 🤖 **AI 기반 ROI 선택**: Point/Text 프롬프트로 간편한 영역 지정
- 🔬 **3D Feature Analysis**: Ray-casting 기반 정밀한 intensity profile 추출
- ⚡ **자동 최적화**: 기하학적 loss 기반 TF 파라미터 자동 조정
- 🎨 **실시간 렌더링**: VTK GPU 가속 고품질 볼륨 렌더링

---

## 🚀 핵심 기능

### 1️⃣ 다양한 볼륨 데이터 지원
- **지원 포맷**: NIfTI (`.nii`, `.nii.gz`), NumPy (`.npy`), Raw (`.raw`, `.dat`)
- **자동 전처리**: 정규화, 차원 검증, Fortran order 변환
- **커스텀 파라미터**: Raw 데이터 로드 시 대화형 파라미터 설정

### 2️⃣ SAM 기반 스마트 ROI 선택
#### Point-based SAM
```
사용자 클릭 → 2D 마스크 생성 → 3D ROI 추출
```
- 렌더링 화면에서 직접 포인트 선택
- Positive/Negative 포인트로 정밀 제어
- 실시간 마스크 프리뷰

#### Text-based SAM
```
"kidney" 입력 → AI가 자동으로 해당 구조 인식 및 세그멘테이션
```
- 자연어 프롬프트 지원
- 복잡한 해부학적 구조 자동 인식
- 다중 객체 동시 선택 가능

### 3️⃣ 자동 TF 최적화
```python
# 최적화 프로세스
1. ROI 내 3D intensity profile 추출 (Ray-casting)
2. 초기 TF로 렌더링 수행
3. SAM 마스크 vs 렌더링 결과 비교
4. Geometric Loss 계산
5. Nelder-Mead로 TF 파라미터 조정
6. 수렴할 때까지 2-5 반복
```

**최적화 Loss Function**:
- `Inclusion Penalty`: ROI 포인트가 마스크 외부에 있는 정도
- `Boundary Distance`: 포인트가 마스크 경계에 너무 가까운 정도  
- `Minimum Area`: 마스크 크기 최소화 (과적합 방지)

### 4️⃣ 고급 렌더링 제어
- **조명**: Key/Fill Light, Follow Camera 모드
- **Shading**: Ambient/Diffuse/Specular 세밀 조정
- **Clipping**: 6방향 독립적 볼륨 자르기
- **카메라**: 위치 저장/복원, Zoom 제어

---

## 💻 설치 방법

### 시스템 요구사항
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 11+
- **Python**: 3.10 이상
- **GPU**: CUDA 지원 GPU 권장 (VTK Ray-casting + SAM 추론)
- **RAM**: 16GB 이상 권장 (대용량 볼륨 처리 시)

### 패키지 설치

```bash
# 기본 패키지
pip install --break-system-packages \
    PyQt6 \
    vtk \
    numpy \
    scipy \
    nibabel \
    pillow

# AI 모델 (SAM3)
pip install --break-system-packages \
    torch \
    transformers \
    huggingface_hub
```

### SAM 모델 다운로드
첫 실행 시 자동으로 다운로드됩니다 (~2GB). 수동 다운로드 시:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="facebook/sam3",
    filename="sam3.pt",
    token="YOUR_HF_TOKEN"
)
```

---

## 📖 사용 방법

### 빠른 시작

```bash
# 프로그램 실행
python main.py
```

### 기본 워크플로우

#### 1단계: 볼륨 데이터 로드
```
좌측 "Load Volume Data" 클릭 → 파일 선택
```
- NIfTI 파일: 바로 로드
- Raw 파일: 파라미터 다이얼로그에서 차원/타입 지정

#### 2단계: ROI 선택 (두 가지 방법)

**방법 A - Point-based SAM**
```
1. 우측 "Enable Set Mode" 체크
2. 렌더링 화면에서 관심 영역 클릭 (여러 점 가능)
3. "Check SAM" 버튼 클릭 → 마스크 확인
```

**방법 B - Text-based SAM**
```
1. 텍스트 입력창에 프롬프트 입력 (예: "kidney", "tumor")
2. "Run Text SAM" 버튼 클릭
3. AI가 자동으로 해당 구조 인식
```

#### 3단계: 최적화 실행
```
1. Maxiter 슬라이더 조정 (10-200, 기본값: 50)
2. Tolerance 설정 (기본값: 0.001)
3. "Run Optimization" 버튼 클릭
4. 진행 상황 모니터링 (상태바)
```

#### 4단계: 결과 저장
```
좌측 "Save" 버튼 → TF 프리셋 저장
우측 중앙 "Save Image" → 렌더링 이미지 저장
```

### 고급 기능

#### TF 수동 편집
```
좌측 TF 편집기에서:
- 노드 드래그: 위치/투명도 조정
- 더블클릭: 노드 추가/삭제
- 우클릭: 색상 변경
```

#### 카메라 뷰 저장
```
중앙 "Save Cam" 버튼 → JSON 파일로 저장
나중에 "Load Cam"으로 동일한 시점 복원
```

#### Clipping으로 내부 구조 보기
```
좌측 "Clipping Controls" 펼치기
→ X/Y/Z축 Min/Max 슬라이더 조정
```

---

## 🏗️ 시스템 아키텍처

### 전체 구조

```
main.py (진입점)
│
└─ VolumeRenderingMainWindow (main_window.py)
    │
    ├─ 왼쪽 패널 (Left Layout)
    │   ├─ FilePanel          # 데이터 로드
    │   └─ TFPanel            # TF 편집, Shading, Clipping
    │
    ├─ 중앙 패널 (Center Layout)
    │   └─ RenderingPanel
    │       └─ VTKVolumeRenderer (renderer_widget.py)
    │           ├─ CameraController      # 카메라 상태 관리
    │           ├─ LightingManager       # 조명 제어
    │           ├─ ClippingManager       # Clipping 평면
    │           ├─ ScreenshotManager     # 이미지 저장
    │           └─ VTKPointOverlay       # 2D 포인트 오버레이
    │
    └─ 오른쪽 패널 (Right Layout)
        └─ OptimizationPanel
            ├─ SAM 결과 표시 (QLabel with QPixmap)
            ├─ Point/Text 입력 UI
            └─ 최적화 파라미터 설정
```

### 주요 데이터 흐름

#### 1. 데이터 로드 파이프라인
```
User → FilePanel.load_volume_data()
     → VolumeLoader.load()
        → [NIfTILoader / NpyLoader / RawLoader].load()
        → VolumeProcessor.process()
     → MainWindow.on_volume_loaded()
     → RenderingPanel.set_volume_data()
     → VTKVolumeRenderer._setup_standard_volume()
```

#### 2. SAM 실행 플로우
```
User Click → MainWindow.on_point_2d_picked()
          → OptimizationPanel.picked_points.append()
          
User → "Check SAM" 버튼
    → MainWindow.on_check_sam()
    → RenderingPanel.save_current_rendering() (스크린샷)
    → SAMService.predict_async()  [비동기 QThread]
    → SAMWrapper.predict() [PyTorch 추론]
    → MainWindow.on_mask() [시그널 콜백]
    → FeatureAnalyzer.analyze_roi_profile()
       ├─ Ray-casting으로 3D intensity profile 추출
       └─ picked_points, picked_intensities 반환
    → OptimizationPanel.set_analyzer_result()
```

#### 3. 최적화 루프
```
User → "Run Optimization" 버튼
    → MainWindow.start_optimization_process()
    → OptimizationWorker 생성 (QThread)
    
[Worker Thread]
loop until convergence:
    OptimizationWorker.calculate_step_loss()
    ├─ TF 파라미터 조정
    ├─ request_render_sync 시그널 발송 [→ Main Thread]
    └─ wait (QWaitCondition)
    
[Main Thread]
MainWindow.handle_sync_render_request()
├─ TFPanel.apply_external_nodes() (TF 업데이트)
├─ RenderingPanel.update_transfer_function()
├─ RenderingPanel.save_current_rendering()
└─ OptimizationWorker.set_rendered_image() (Worker 깨우기)
    
[Worker Thread]
    ├─ SAMWrapper.predict() (새 마스크 생성)
    ├─ Geometric Loss 계산
    └─ Nelder-Mead 업데이트
    
→ 수렴 시 result_ready 시그널
→ MainWindow.on_optimization_finished()
```

---

## 📂 코드 구조

### 디렉토리 트리

```
SAM-TF-OPT/
├── main.py                          # 진입점
├── check_depend.py                  # 의존성 체크
├── README.md
│
├── resources/                       # 리소스 디렉토리
│   ├── Camera/                      # 카메라 뷰 JSON
│   ├── Points/                      # 수동 선택 포인트
│   ├── TFs/                         # TF 프리셋
│   ├── Rendered_Image/              # 스크린샷
│   └── Volume_Data/                 # 샘플 데이터
│
└── src/
    ├── main_window.py               # 메인 윈도우
    │
    ├── core/                        # 핵심 로직
    │   ├── sam_wrapper.py           # SAM3 모델 래퍼
    │   ├── support_sam.py           # SAMService (비동기)
    │   ├── feature_analyzer.py      # 3D Feature 추출
    │   ├── tf_optimizer.py          # TF 최적화 엔진
    │   └── support_optimization.py  # 최적화 워커
    │
    ├── gui/
    │   ├── data/                    # 데이터 로더
    │   │   ├── base_loader.py       # 추상 베이스 클래스
    │   │   ├── nifti_loader.py      # NIfTI 로더
    │   │   ├── npy_loader.py        # NumPy 로더
    │   │   ├── raw_loader.py        # Raw 로더
    │   │   ├── volume_loader.py     # 통합 로더 (전략 패턴)
    │   │   └── volume_processor.py  # 전처리
    │   │
    │   ├── dialogs/                 # 다이얼로그
    │   │   ├── raw_data_dialog.py   # Raw 파라미터 입력
    │   │   └── verification_dialog.py # MPR 뷰어
    │   │
    │   ├── panel/                   # UI 패널
    │   │   ├── base_panel.py        # 베이스 클래스
    │   │   ├── file_panel.py        # 파일 로드
    │   │   ├── tf_panel.py          # TF 편집
    │   │   ├── rendering_panel.py   # 렌더링 제어
    │   │   ├── optimization_panel.py # 최적화 제어
    │   │   └── clipping_panel.py    # Clipping UI
    │   │
    │   ├── rendering/               # 렌더링 매니저
    │   │   ├── camera_controller.py # 카메라 상태
    │   │   ├── lighting_manager.py  # 조명 제어
    │   │   ├── clipping_manager.py  # Clipping 평면
    │   │   └── screenshot_manager.py # 스크린샷
    │   │
    │   └── widget/                  # 커스텀 위젯
    │       ├── renderer_widget.py   # VTK 렌더러
    │       ├── transfer_function_widget.py # TF 편집기
    │       └── light_sphere_widget.py # 조명 방향 구
    │
    └── __pycache__/
```

### 주요 모듈 설명

#### 🔵 Core Modules

| 파일 | 클래스/함수 | 역할 |
|------|------------|------|
| `sam_wrapper.py` | `SAMWrapper` | SAM3 모델 로드 및 추론 (Point/Text) |
| `support_sam.py` | `SAMService`, `_SAMWorker` | QThread 기반 비동기 SAM 실행 |
| `feature_analyzer.py` | `FeatureAnalyzer` | Ray-casting으로 3D intensity profile 추출 |
| `tf_optimizer.py` | `TFOptimizer` | Tent 파라미터 최적화 (scipy.optimize) |
| `support_optimization.py` | `OptimizationWorker` | 백그라운드 최적화 루프 (QThread) |

#### 🟢 GUI Panels

| 파일 | 클래스 | 주요 기능 |
|------|--------|----------|
| `file_panel.py` | `FilePanel` | 볼륨 로드, 파일 정보 표시 |
| `tf_panel.py` | `TransferFunctionPanel` | TF 편집, Shading, Clipping 제어 |
| `rendering_panel.py` | `RenderingPanel` | Zoom, Sampling, Camera 저장/복원 |
| `optimization_panel.py` | `OptimizationPanel` | SAM 입력/결과, 최적화 실행 |

#### 🟡 Rendering System

| 파일 | 클래스 | 역할 |
|------|--------|------|
| `renderer_widget.py` | `VTKVolumeRenderer` | 메인 렌더러 (GPU Ray-casting) |
|  | `VTKPointOverlay` | Native 2D 포인트 오버레이 |
| `camera_controller.py` | `CameraController` | 카메라 상태 저장/복원, Zoom |
| `lighting_manager.py` | `LightingManager` | Key/Fill Light, Shading 속성 |
| `clipping_manager.py` | `VolumeClippingManager` | 6방향 Clipping 평면 관리 |
| `screenshot_manager.py` | `ScreenshotManager` | 렌더링 이미지 저장 |

---


## 🔧 확장 가능 기능

### ✅ 구현 완료
- [x] Multi-format 볼륨 로더 (NIfTI, NumPy, Raw)
- [x] Point/Text SAM 통합
- [x] 3D Feature Analysis (Ray-casting)
- [x] Background Optimization (QThread)
- [x] Native 2D Overlay (OS 독립적)
- [x] 카메라/TF 상태 저장/복원
- [x] 고급 렌더링 제어 (Shading, Clipping, Lighting)

### 🛠️ 추천 개선사항
- [ ] **GPU 가속 Ray-casting**: CUDA 커널로 Feature Extraction 병렬화 (100x 속도 향상)
- [ ] **Multi-ROI 최적화**: 여러 영역 동시 선택 및 가중치 조절
- [ ] **Loss Function 개선**: 
  - Perceptual Loss (LPIPS) 추가
  - Histogram Matching 기반 색상 일관성
  - Depth-aware Loss (깊이 정보 활용)
- [ ] **TF History**: Undo/Redo 스택 구현
- [ ] **Batch Processing**: 
  - 여러 데이터셋 자동 최적화
  - 스크립트 기반 파이프라인
- [ ] **Real-time Preview**: 
  - 최적화 중 중간 결과 실시간 표시
  - Loss 그래프 시각화
- [ ] **Export Options**:
  - 렌더링 동영상 생성 (360° 회전)
  - 3D Mesh Export (Marching Cubes)
  - Publication-ready 이미지 (고해상도, 스케일바)
- [ ] **Machine Learning Integration**:
  - TF 예측 모델 학습 (One-shot TF generation)
  - Reinforcement Learning 기반 최적화

---

## 📦 리소스 디렉토리

```
resources/
├── Camera/          # 저장된 카메라 뷰 (JSON)
│   ├── calix_camera_1.json
│   ├── human_brain.json
│   └── ...
│
├── Points/          # 수동 선택 포인트 (JSON)
│   └── points_YYYYMMDD_HHMMSS.json
│
├── TFs/             # Transfer Function 프리셋 (JSON)
│   ├── calix.json
│   ├── cardiac_1.json
│   └── ...
│
├── Rendered_Image/  # 렌더링 스크린샷 (PNG)
│   └── render_YYYYMMDD_HHMMSS.png
│
└── Volume_Data/     # 샘플 데이터셋
    ├── CALIX.nii
    └── tooth_103x94x161_uint8.nii
```

---

## ⚠️ 주의사항

### 1. 메모리 사용량
- **대용량 볼륨** (>512³): 16GB+ RAM 권장
- **GPU 메모리**: VTK Ray-casting은 볼륨을 GPU에 로드 (8GB+ VRAM 권장)
- **최적화 중**: 각 iteration마다 렌더링 수행 → 메모리 누수 주의

### 2. SAM 모델
- **첫 실행 시**: `facebook/sam3` (~2GB) 자동 다운로드
- **HuggingFace 토큰**: `sam_wrapper.py`에 하드코딩된 토큰 확인 필요
- **GPU 필수**: CPU 추론 시 매우 느림 (분 단위)

### 3. Thread Safety
- `OptimizationWorker`는 QMutex/QWaitCondition으로 동기화 완료
- 렌더링은 반드시 메인 스레드에서 수행 (VTK 제약)
- 최적화 중 UI 조작 금지 (데이터 불일치 가능)

### 4. 플랫폼별 이슈
- **Windows**: VTK GPU 드라이버 충돌 가능 → 최신 그래픽 드라이버 설치
- **Linux**: Mesa 라이브러리 필요 → `libgl1-mesa-glx` 설치
- **macOS**: M1/M2에서 VTK 호환성 이슈 → Rosetta 모드 실행 권장


---

## 📚 참고 자료

### 논문
- [SAM (Segment Anything)](https://arxiv.org/abs/2304.02643) - Meta AI, 2023

### 라이브러리 문서
- [VTK Documentation](https://vtk.org/documentation/)
- [PyQt6 Reference](https://www.riverbankcomputing.com/static/Docs/PyQt6/)
- [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/)

---

</div>