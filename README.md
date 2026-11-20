# MR WYSIWYG

## 실행

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python main.py
```

SAM은 HuggingFace `facebook/sam3`(접근 승인 필요)를 쓴다. 사용 신청 후 `huggingface-cli login`.
데이터셋 zip은 repo 루트에서 풀면 `resources/` 아래 제자리에 들어간다.

## Clustering 사용법

1. **데이터셋 로드**: TF 패널 **Dataset Presets**에서 데이터셋을 누른다(볼륨·INR·TF·카메라·SAM 포인트).
   Features의 INR·Coords를 쓰려면 INR feature가 로드돼 있어야 한다.
2. **ROI 지정** (SAM 패널): **Enable Set Mode**를 켜고
   - **Positive** / **Negative**: 화면을 클릭해 점을 찍는다.
   - **Line**: 드래그로 선을 그리면 선을 따라 positive 점 4개로 바뀐다(새로 그으면 이전 점은 지워짐).
   - **💾 Save Pts** / **📥 Load Pts**: 점 저장·불러오기.
3. **🔍 Check SAM**: 현재 화면으로 SAM mask를 만들어 보여준다. TF나 카메라를 바꾸면 다시 눌러야 한다.
4. **🧩 Run Clustering**: Stage A → Stage B를 돌리고, 결과 ROI를 local class(**Local N**)로 등록한다.
   - Stage A: mask 픽셀(stride 5)마다 ray를 쏴 샘플을 만들고 K-means로 나눈 뒤,
     점수(평균 visibility + 작은 depth 가중)가 가장 큰 cluster를 best로 고른다.
   - Stage B: best 샘플 / 그 밖의 voxel로 Random Forest를 학습해 볼륨 전체 voxel을 판정한다(P(best) > 0.5 → ROI).
   - 진행 로그와 cluster별 통계는 콘솔에 나온다.
5. **결과 보기** (WYSIWYG 패널 > Debug):
   - **Show K-means**: Stage A 샘플을 cluster 색으로(빨강 = best). 아래 cluster 칸에서 cluster별 표시·색 변경.
   - **Show Final Clusters**: Stage B 결과(ROI + 나머지 cluster).
   - **Show RF samples**: RF 학습셋 위치(빨강 = best, 파랑 = not-best).
   - **Z-Slice Viewer...**: 볼륨 단면 위에 K-means / Final / RF P(best)를 겹쳐 본다.
6. TF 패널의 편집 대상을 **Local N**으로 바꾸면 그 ROI의 TF만 편집된다. ROI 삭제는 **Delete**.

ROI를 여러 개 만들 수 있다(나중 ROI가 겹치는 voxel을 가져감).

## Clustering 설정 (WYSIWYG 패널 > Clustering)

Run Clustering을 누를 때 반영된다.

| 항목 | 설명 |
|---|---|
| k | K-means cluster 수 (기본값 6, tooth는 4) |
| seed | K-means 초기화 seed |
| restarts | K-means 재시작 횟수(샘플 5만 개로 돌려 SSE가 가장 낮은 해에서 시작) |
| Stage B stride floor | Stage B 투과율 계산용 ray 간격(픽셀) 하한 |
| Stage B RF: INR만 | RF를 INR embedding + coords만으로 학습 |
| trees / 마진 / P(best) ≥ | RF tree 수 / best 주변 이만큼(voxel)은 not-best로 뽑지 않음 / ROI 판정 추가 하한 |
| Features | INR (8D), Coords (3D) |
| Local | Visibility, Depth, Intensity, Opacity (Visibility·Depth는 현재 카메라 기준) |
| Visibility 값 | feature 열에 넣을 값: V(opacity × 투과율) 또는 T(투과율). best 점수는 항상 V |
| V/O min-max | Visibility·Opacity 열을 z-score 대신 min-max로 정규화 |
| best 공간 분리 | best 샘플 중 공간적으로 떨어진 덩어리를 best에서 뺌(한 번에 대상 하나를 찍는 전제) |

## 결과 파일

- Run Clustering마다 `resources/Experiments/stage_a_dump/`에 저장된다.
  - `stage_a_<시각>.npz`: 샘플 좌표·feature·cluster 라벨·best, UI 설정
  - `stage_b_<시각>.npz`: 최종 ROI mask, RF P(best)
- 덤프 보기: `python scripts\compare_stage_a_slices.py <stage_a_*.npz>`
- 결과 공유 zip(단면 뷰어 + cluster별 feature heatmap): `python scripts\export_result_package.py <stage_a_*.npz ...> --names <이름 ...>`
