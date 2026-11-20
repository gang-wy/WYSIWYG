"""Stage A 덤프에서 두 샘플 집단(A, B)이 feature 공간에서 얼마나 떨어져 있는지, 그리고 '이상적인 feature'였다면
Stage A가 A만 best로 잡았을지 검증한다. 데이터셋에 무관하다(집단은 voxel mask/GT 라벨로 받는다).

사용 예:
    # GT 라벨 A만 지정 → B = 이 실행의 best cluster에 잘못 들어온 샘플(A 밖)
    .venv\\Scripts\\python.exe experiments\\analyze_group_separation.py <stage_a_*.npz> --dataset tooth --a-label 3
    # GT 라벨 두 개
    .venv\\Scripts\\python.exe experiments\\analyze_group_separation.py <stage_a_*.npz> --gt resources/Volume_Data/3Dircadb2.1_GT.nii --a-label 5 --b-label 2
    # mask 파일 두 개 (BRAINIX: experiments/make_brainix_tumor_scalp_masks.py로 생성)
    .venv\\Scripts\\python.exe experiments\\analyze_group_separation.py <stage_a_*.npz> --a-mask tumor.nii --b-mask scalp.nii

입력 덤프: main.py가 Stage A 실행 때 저장하는 resources/Experiments/stage_a_dump/stage_a_*.npz
  normalized (N, D)  K-means에 실제로 들어간 행렬(열마다 z-score, layout에 열 범위)
  labels / centroids / best / seed  그 실행의 K-means 결과
  world_points (N, 3) 샘플 위치(mm) → voxel_spacing으로 나눠 반올림한 voxel의 mask 값으로 A/B 소속을 정한다
  visibility / depths  best 점수 계산용

검증 순서:
  [1] 집단 — A/B 샘플 수, 이 실행의 best cluster가 각 집단을 얼마나 포함했는지.
  [2] 거리 — 같은 정규화 공간에서 A·B 중심 거리, 각 집단 내부 퍼짐, 이 실행의 cluster 중심 간 거리.
  [3] K-means 기준(SSE) — "B를 A에서 떼어내는 이득"과 "기존 cluster 두 개를 합치는 비용"을 비교한다.
      K-means는 SSE(중심까지 거리 제곱합)를 줄이는 알고리즘이라, 이득이 가장 싼 합치기 비용보다 작으면
      cluster 하나를 A/B 분리에 쓰지 않는 쪽이 SSE가 더 작다.
  [4] 선형 구분 — feature 블록만으로 A/B를 가를 수 있는가(LDA, 5-fold 교차검증, 두 집단 같은 수 표본).
      [2]와 함께 보면 "정보가 없다"와 "정보는 있지만 거리가 가깝다"를 구분할 수 있다.
  [5] 상한 실험(오라클) — 정답(A/B)으로 feature를 조작한 뒤 파이프라인과 같은 K-means(_kmeans_gpu)·best 점수를 다시 돌린다.
      (a) feature 블록 안 A/B 구분 방향 w(LDA, 단위 벡터)를 α배 늘림 — 정보는 그대로, 거리 배치만 바꿈
      (b) B 샘플만 w 방향(A 반대쪽)으로 D만큼 이동 — 'B가 완전히 다른 재질'인 feature를 흉내
      성공 = best cluster가 A 샘플의 --win-a 이상을 포함하고 B 샘플은 --win-b 이하.
"""
import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.core.clustering.feature_clustering import _kmeans_gpu  # noqa: E402  (파이프라인과 같은 K-means)
from src.core.dataset_presets import DATASET_PRESETS  # noqa: E402

BEST_BETA1 = 1e-4   # 파이프라인 best 점수 S = 평균 visibility + beta1 × 평균 depth (feature_clustering.py)
MIN_GROUP = 20      # 이보다 작은 집단은 통계가 불안정해 중단


class Tee:
    """print를 화면과 --out 파일에 같이 쓴다."""
    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, *a, **kw):
        print(*a, **kw, flush=True)
        if self.f:
            print(*a, **{k: v for k, v in kw.items() if k != "flush"}, file=self.f, flush=True)


def load_volume_array(path):
    if path.lower().endswith(".npy"):
        arr = np.load(path)
    else:
        arr = np.asarray(nib.load(path).dataobj)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return arr


def find_gt(dataset):
    """run_gt_evaluation.py와 같은 규칙: 볼륨 폴더의 <볼륨 이름>_GT.* 또는 <dataset>_GT.*"""
    vol = os.path.join(ROOT, DATASET_PRESETS[dataset]["volume"])
    folder, stem = os.path.dirname(vol), os.path.basename(vol).split(".")[0]
    for pattern in (f"{stem}_GT.*", f"{dataset}_GT.*", f"{dataset.capitalize()}_GT.*"):
        hits = glob.glob(os.path.join(folder, pattern))
        if hits:
            return hits[0]
    return None


def label_mask(gt, label):
    """3D 라벨 볼륨이면 == label, 4D 채널 볼륨이면 channel label > 0."""
    return gt[..., int(label)] > 0 if gt.ndim == 4 else gt == int(label)


def sse_split_gain(A, B):
    """A∪B 한 cluster 대비 A, B 두 cluster로 나눌 때 줄어드는 SSE = nA·nB/(nA+nB)·‖중심A−중심B‖²."""
    na, nb = len(A), len(B)
    return na * nb / (na + nb) * float(((A.mean(0) - B.mean(0)) ** 2).sum())


def best_cluster(labels, k, vis, dep):
    s = [vis[labels == c].mean() + BEST_BETA1 * dep[labels == c].mean() if (labels == c).any() else -np.inf
         for c in range(k)]
    return int(np.argmax(s))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", help="stage_a_*.npz")
    ap.add_argument("--dataset", choices=sorted(DATASET_PRESETS), help="GT 자동 찾기용(--gt 대신)")
    ap.add_argument("--gt", help="GT 볼륨(.nii/.nii.gz/.npy) — 3D 라벨 또는 4D 채널")
    ap.add_argument("--a-label", type=int, help="GT에서 집단 A 라벨(채널)")
    ap.add_argument("--b-label", type=int, help="GT에서 집단 B 라벨(채널). 생략하고 --b-mask도 없으면 B = best 안의 A 밖 샘플")
    ap.add_argument("--a-mask", help="집단 A voxel mask(0이 아니면 A)")
    ap.add_argument("--b-mask", help="집단 B voxel mask")
    ap.add_argument("--dilate-a", type=int, default=0, help="A mask 팽창 voxel 수(보간 샘플이 경계에 걸리는 것 허용)")
    ap.add_argument("--block", default=None, help="거리/LDA/오라클에 쓸 feature 블록 이름(기본: inr로 시작하는 블록)")
    ap.add_argument("--seeds", type=int, default=5, help="오라클 K-means seed 개수(0부터)")
    ap.add_argument("--win-a", type=float, default=0.5)
    ap.add_argument("--win-b", type=float, default=0.2)
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--out", help="결과를 이 텍스트 파일에도 저장")
    args = ap.parse_args()
    out = Tee(args.out)

    d = np.load(args.dump)
    X = d["normalized"].astype(np.float64)
    labels, k, b0 = d["labels"], len(d["centroids"]), int(d["best"])
    layout = [(s.split(":")[0], int(s.split(":")[1]), int(s.split(":")[2])) for s in d["layout"]]
    vis, dep = d["visibility"], d["depths"]
    if args.block:
        i0, i1 = next((s, e) for n, s, e in layout if n == args.block)
        block = args.block
    else:
        block, i0, i1 = next(((n, s, e) for n, s, e in layout if n.startswith("inr")), ("전체", 0, X.shape[1]))
    out(f"덤프 {os.path.basename(args.dump)} | 샘플 {len(X):,} | 열 {X.shape[1]} {[f'{n}({e - s})' for n, s, e in layout]} | "
        f"k {k} | K-means seed {int(d['seed'])} | best {b0}")

    # ── [1] 집단
    gt = None
    if args.a_label is not None or args.b_label is not None:
        gt_path = args.gt or (find_gt(args.dataset) if args.dataset else None)
        if not gt_path:
            sys.exit("GT를 찾지 못했습니다. --gt 또는 --dataset을 지정하세요.")
        gt = load_volume_array(gt_path)
        out(f"GT {gt_path} shape {gt.shape}")
    if args.a_mask:
        mask_a = load_volume_array(args.a_mask) > 0
    elif args.a_label is not None:
        mask_a = label_mask(gt, args.a_label)
    else:
        sys.exit("집단 A가 필요합니다: --a-mask 또는 --a-label")
    if args.dilate_a:
        mask_a = ndimage.binary_dilation(mask_a, iterations=args.dilate_a)
    shape = np.array(mask_a.shape[:3])

    ijk_f = d["world_points"] / d["voxel_spacing"]
    inside = np.all((ijk_f > -0.5) & (ijk_f < shape - 0.5), axis=1)
    ijk = np.clip(np.round(ijk_f).astype(int), 0, shape - 1)
    at = lambda m: m[ijk[:, 0], ijk[:, 1], ijk[:, 2]] & inside
    s_a = at(mask_a)

    if args.b_mask:
        s_b, b_desc = at(load_volume_array(args.b_mask) > 0) & ~s_a, f"mask {os.path.basename(args.b_mask)}"
    elif args.b_label is not None:
        s_b, b_desc = at(label_mask(gt, args.b_label)) & ~s_a, f"GT 라벨 {args.b_label}"
    else:
        s_b, b_desc = (labels == b0) & ~s_a, f"이 실행 best cluster {b0} 안의 A 밖 샘플"
    na, nb = int(s_a.sum()), int(s_b.sum())
    out(f"\n[1] 집단 (샘플 위치 → 반올림 voxel의 mask 값, 볼륨 안 샘플 {inside.mean() * 100:.1f}%)")
    out(f"    A: {na:,}개 ({na / len(X) * 100:.1f}%)")
    out(f"    B: {nb:,}개 ({nb / len(X) * 100:.1f}%) — {b_desc}")
    out(f"    이 실행의 best cluster {b0}: A의 {np.mean(labels[s_a] == b0) * 100:.1f}%, B의 {np.mean(labels[s_b] == b0) * 100:.1f}% 포함")
    if min(na, nb) < MIN_GROUP:
        sys.exit(f"집단이 너무 작습니다(최소 {MIN_GROUP}). 라벨/mask를 확인하세요.")

    # ── [2] 거리
    out(f"\n[2] 거리 (K-means 입력과 같은 정규화 공간, 유클리드)")
    out(f"    {'공간':<16}{'A-B 중심':>10}{'A 내부 퍼짐':>12}{'B 내부 퍼짐':>12}{'전체 퍼짐':>10}   cluster 중심 간 (최소/중앙/최대)")
    spaces = [(f"{block} {i1 - i0}D", slice(i0, i1))]
    if (i0, i1) != (0, X.shape[1]):
        spaces.append((f"전체 {X.shape[1]}D", slice(None)))
    for name, cols in spaces:
        Z = X[:, cols]
        spread = lambda P: float(np.median(np.linalg.norm(P - P.mean(0), axis=1)))
        C = np.array([Z[labels == c].mean(0) for c in range(k) if (labels == c).any()])
        cd = [np.linalg.norm(C[a] - C[b]) for a in range(len(C)) for b in range(a + 1, len(C))]
        out(f"    {name:<16}{np.linalg.norm(Z[s_a].mean(0) - Z[s_b].mean(0)):>10.2f}{spread(Z[s_a]):>12.2f}"
            f"{spread(Z[s_b]):>12.2f}{spread(Z):>10.2f}   {min(cd):.2f} / {np.median(cd):.2f} / {max(cd):.2f}")
    out("    퍼짐 = 집단 중심까지 거리의 중앙값. cluster 중심 = 이 실행 K-means 결과의 cluster별 평균.")

    # ── [3] SSE
    gain = sse_split_gain(X[s_a], X[s_b])
    merges = sorted((sse_split_gain(X[labels == a], X[labels == b]), a, b)
                    for a in range(k) for b in range(a + 1, k) if (labels == a).any() and (labels == b).any())
    out(f"\n[3] K-means 기준 (SSE, 전체 {X.shape[1]}D)")
    out(f"    B를 A에서 떼어낼 때 줄어드는 SSE: {gain:,.0f}")
    out(f"    기존 cluster 두 개를 합칠 때 늘어나는 SSE: 최소 {merges[0][0]:,.0f} (cluster {merges[0][1]}+{merges[0][2]}), "
        f"중앙 {np.median([m[0] for m in merges]):,.0f}")
    out(f"    → 비 {gain / merges[0][0]:.3f}. " + ("1보다 작음: K-means는 cluster 하나를 A/B 분리에 쓰지 않는 쪽이 SSE가 더 작다."
                                             if gain < merges[0][0] else "1 이상: SSE 기준으로는 A/B 분리가 이득이다."))

    # ── [4] 선형 구분
    rng = np.random.default_rng(0)
    m = min(na, nb)
    idx = np.r_[rng.choice(np.flatnonzero(s_a), m, replace=False), rng.choice(np.flatnonzero(s_b), m, replace=False)]
    y = np.r_[np.zeros(m), np.ones(m)]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    out(f"\n[4] 선형 구분 (LDA, 5-fold 교차검증, 각 {m:,}개, 우연 수준 50%)")
    tests = [(f"{block} {i1 - i0}D", X[idx, i0:i1])]
    if "intensities" in d.files:
        tests.append(("intensity만", d["intensities"][idx][:, None]))
    for name, Z in tests:
        out(f"    {name:<16} 정확도 {cross_val_score(LinearDiscriminantAnalysis(), Z, y, cv=cv).mean() * 100:.1f}%")

    if args.no_oracle:
        return

    # ── [5] 오라클
    Xi = X[:, i0:i1]
    lda = LinearDiscriminantAnalysis().fit(np.r_[Xi[s_a], Xi[s_b]], np.r_[np.zeros(na), np.ones(nb)])
    w = lda.scalings_[:, 0] / np.linalg.norm(lda.scalings_[:, 0])
    p = Xi @ w - (Xi @ w).mean()
    out(f"\n[5] 상한 실험 — w = {block} 블록 안 A/B LDA 방향(단위 벡터), 성공 = best가 A ≥{args.win_a:.0%}, B ≤{args.win_b:.0%}")
    out(f"    w 위 투영: 전체 샘플 1~99 백분위 {np.percentile(p, 1):.2f} ~ {np.percentile(p, 99):.2f} | "
        f"A 중앙 {np.median(p[s_a]):.2f} | B 중앙 {np.median(p[s_b]):.2f}")

    def run(Xmod, tag):
        wins = 0
        for seed in range(args.seeds):
            lab = np.asarray(_kmeans_gpu(Xmod, k, seed=seed)[0])
            b = best_cluster(lab, k, vis, dep)
            fa, fb = np.mean(lab[s_a] == b), np.mean(lab[s_b] == b)
            ok = fa >= args.win_a and fb <= args.win_b
            wins += ok
            out(f"    {tag} seed {seed}: best 크기 {np.sum(lab == b):>8,} | A {fa * 100:5.1f}% | B {fb * 100:5.1f}%"
                + ("  ← 성공" if ok else ""))
        out(f"    {tag} 성공 {wins}/{args.seeds}")

    out("\n  (a) w 방향 거리를 α배")
    for alpha in (1, 2, 4, 8, 16):
        Xm = X.copy()
        Xm[:, i0:i1] += (alpha - 1) * p[:, None] * w[None, :]
        run(Xm, f"α={alpha:<2}")

    out("\n  (b) B 샘플만 w 방향(A 반대쪽)으로 D만큼 이동")
    sgn = np.sign(np.median(p[s_b]) - np.median(p[s_a])) or 1.0
    for D in (0, 1, 2, 4, 8, 16):
        Xm = X.copy()
        Xm[np.ix_(s_b, np.arange(i0, i1))] += sgn * D * w[None, :]
        run(Xm, f"D={D:<2}")


if __name__ == "__main__":
    main()
