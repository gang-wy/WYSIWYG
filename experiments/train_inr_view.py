"""NeuVolEx INR을 다시 학습하면서 view head(현재 시점·TF의 voxel별 투과율 T와 opacity alpha 예측)를 붙인다.

feature 개선 실험용: 손으로 만든 렌더링 열(V/T, depth, opacity) 대신 렌더링 정보를 담도록 적응시킨 INR embedding이
Stage A/B 분할(GT Dice)을 좋게 하는지 본다. --view-weight 0이면 렌더링 손실 없는 재학습(대조군)이다.

모델·학습 설정은 INR-DVR GUI의 neuvolex 학습과 같다(baseline_se_fm_mh, feature 8, hash 12×4, MLP 64×4,
100 epoch, lr 1e-3, 구조 패치 5³). INR-DVR 코드는 import만 하고 고치지 않는다 — train()은 손실을 밖에서
더할 수 없어 학습 루프를 이 파일로 옮기고 view 손실 한 줄만 더했다.

정답(T, alpha, ok)은 scripts/run_gt_evaluation.py --export-view-targets로 만든다(Stage B와 같은 TF·시점·코드).

실행 (INR-DVR의 .venv — torch + tiny-cuda-nn 필요, WYSIWYG 프로젝트 루트에서):
    <INR-DVR>\\.venv\\Scripts\\python.exe experiments\\train_inr_view.py ^
        --targets <tooth_view_targets.npz> --view-weight 10 --out <tooth_view10.nii>
"""
import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INR_DVR_ROOT = r"C:\Users\vomlab\Downloads\INR-DVR"   # INR-DVR 저장소 위치(PC마다 바꿀 것)
sys.path.insert(0, INR_DVR_ROOT)
from core.tcnn_feature_extractor import INRFeatureExtractor  # noqa: E402

FEATURE_DIM = 8


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volume", default="resources/Volume_Data/tooth.nii")
    p.add_argument("--targets", required=True, help="run_gt_evaluation.py --export-view-targets 결과 .npz")
    p.add_argument("--view-weight", type=float, required=True, help="view 손실 가중치 λ (0 = 대조군)")
    p.add_argument("--out", required=True, help="출력 feature .nii (X, Y, Z, 8 + 좌표 3)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha-balance", choices=["alpha", "voxel"],
                   help="view 손실에서 α>0(보임)·α=0(투명) voxel을 반반 반영: alpha = α 항만, voxel = T·α 둘 다")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    # INR-DVR VolumeProcessor와 같은 전처리: 전체 min-max로 [0, 1]
    volume = np.asarray(nib.load(args.volume).dataobj, dtype=np.float32)
    if volume.ndim == 4 and volume.shape[3] == 1:   # CALIX.nii는 (X, Y, Z, 1)
        volume = volume[..., 0]
    volume = (volume - volume.min()) / max(float(volume.max() - volume.min()), 1e-12)

    targets = np.load(args.targets)
    assert targets["T"].shape == volume.shape, (targets["T"].shape, volume.shape)
    # 좌표 순서는 extractor와 같은 (x, y, z) 'ij' flatten
    view_target = torch.from_numpy(np.stack([targets["T"].ravel(), targets["alpha"].ravel()], 1)).to(device)
    view_ok = torch.from_numpy(targets["ok"].ravel()).to(device)

    ext = INRFeatureExtractor(mode="baseline_se_fm_mh", device=device, structural_neighborhood_size=5)
    ext.set_volume(volume)
    ext.create_model(num_levels=12, features_per_level=4, feature_dim=FEATURE_DIM, mlp_neurons=64, mlp_layers=4)
    model = ext.model
    # view head: 최종 feature(8) → (T, alpha). INR-DVR head와 같은 크기의 작은 MLP
    view_head = nn.Sequential(nn.Linear(FEATURE_DIM, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
                              nn.Linear(32, 2), nn.Sigmoid()).to(device)

    # ── INRFeatureExtractor.train()과 같은 정답·최적화 설정 ──
    target_volume = ext.volume_data_tensor                      # (D, H, W)
    target_intensity = target_volume.permute(2, 1, 0).flatten()
    gz, gy, gx = torch.gradient(target_volume)
    target_grad = torch.sqrt(gx.permute(2, 1, 0).flatten() ** 2 + gy.permute(2, 1, 0).flatten() ** 2
                             + gz.permute(2, 1, 0).flatten() ** 2)
    pool = nn.AvgPool3d(3, stride=1, padding=1)
    tv = target_volume.unsqueeze(0).unsqueeze(0)
    local_mean = pool(tv).squeeze()
    local_std = torch.sqrt(torch.clamp(pool(tv ** 2).squeeze() - local_mean ** 2, min=1e-8))
    target_lmean = local_mean.permute(2, 1, 0).flatten()
    target_lstd = local_std.permute(2, 1, 0).flatten()

    params = list(model.parameters()) + list(view_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device=device)
    n = len(ext.coordinates)
    batch = 2 ** 18

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        sums = np.zeros(3)
        nb = 0
        for i in range(0, n, batch):
            optimizer.zero_grad()
            idx = perm[i:i + batch]
            coords = ext.coordinates[idx]
            with torch.amp.autocast(device_type=device):
                out = model(coords, structural_patches=ext._get_neighborhood_patches(coords))
                loss_base = (F.mse_loss(out["intensity"].squeeze(), target_intensity[idx])
                             + 1.0 * F.mse_loss(out["gradient"].squeeze(), target_grad[idx])
                             + 0.5 * (F.mse_loss(out["local_mean"].squeeze(), target_lmean[idx])
                                      + F.mse_loss(out["local_std"].squeeze(), target_lstd[idx])))
                ok = view_ok[idx]
                pred_view = view_head(out["features"].float())
                p, y = pred_view[ok], view_target[idx][ok]
                if not ok.any():
                    loss_view = pred_view.sum() * 0.0
                elif args.alpha_balance:
                    # 투명 voxel이 대부분이라 그냥 평균하면 "전부 투명"으로 맞춰도 손실이 작다
                    # 두 그룹이 손실의 반씩 차지하도록 voxel 가중치를 준다 (한쪽 그룹이 없으면 그 가중치는 안 쓰임)
                    vis = y[:, 1] > 0
                    w = torch.where(vis, 0.5 / vis.float().mean(), 0.5 / (~vis).float().mean())
                    err = (p - y) ** 2
                    if args.alpha_balance == "alpha":
                        err = torch.stack([err[:, 0], w * err[:, 1]], 1)
                    else:
                        err = w[:, None] * err
                    loss_view = err.mean()   # 두 열 평균 → 기존 MSE와 같은 스케일(λ 비교 가능)
                else:
                    loss_view = F.mse_loss(p, y)
                loss = loss_base + args.view_weight * loss_view
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            sums += [loss.item(), loss_base.item(), loss_view.item()]
            nb += 1
        scheduler.step()
        if epoch == 0 or (epoch + 1) % 10 == 0:
            s = sums / nb
            print(f"epoch {epoch + 1:3d}  total {s[0]:.3e}  base {s[1]:.3e}  view {s[2]:.3e}", flush=True)
    print(f"학습 {time.time() - t0:.1f}초 (λ={args.view_weight})")

    # feature: 채널별 min-max [0, 1] + 정규화 좌표 3채널 (*_8d_neuvolex.nii와 같은 형식)
    _, _, feat = ext.generate_outputs()
    X, Y, Z = volume.shape
    gx_, gy_, gz_ = np.meshgrid(np.linspace(0, 1, X), np.linspace(0, 1, Y), np.linspace(0, 1, Z), indexing="ij")
    out_vol = np.concatenate([feat, np.stack([gx_, gy_, gz_], -1).astype(np.float32)], axis=-1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    nib.save(nib.Nifti1Image(out_vol.astype(np.float32), np.eye(4)), args.out)

    print(f"저장: {args.out}  shape {out_vol.shape}")

    # embedding(8D)에 T가 얼마나 들어 있는지: ok voxel에서 T를 8D 선형회귀로 설명하는 R² (B·C 비교용)
    ok = targets["ok"].ravel()
    A = np.hstack([feat.reshape(-1, FEATURE_DIM)[ok], np.ones((int(ok.sum()), 1), np.float32)])
    T = targets["T"].ravel()[ok]
    coef, *_ = np.linalg.lstsq(A, T, rcond=None)
    r2 = 1.0 - np.var(T - A @ coef) / max(np.var(T), 1e-12)
    print(f"T 선형 R² (8D embedding → T): {r2:.3f}")


if __name__ == "__main__":
    main()
