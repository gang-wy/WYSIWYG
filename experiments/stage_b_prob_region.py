"""Stage B 덤프(stage_b_*.npz)의 RF P(best)를 상자 영역 안에서 요약한다.

상자 좌표는 Z-Slice Viewer 아래 줄(마우스 위치 voxel)에서 읽은 voxel index.

사용:
    .venv\\Scripts\\python.exe experiments\\stage_b_prob_region.py --x 100:160 --y 40:120 --z 30:90
    옵션: --dump <npz>        생략하면 resources/Experiments/stage_a_dump의 가장 최근 stage_b_*.npz
          --volume <nii>      주면 --min-value 이상인 voxel만 센다(공기 제외용)
          --min-value <값>    볼륨 원래 값 기준
"""
import argparse
import glob
import os

import numpy as np


def parse_range(text, size):
    a, b = (text or ":").split(":")
    return slice(int(a) if a else 0, int(b) if b else size)


def summarize(p, mask):
    """p: 영역 안 P(best) (판정 불가 -1 포함), mask: 같은 영역의 최종 best mask. Returns: 출력 줄 목록"""
    valid = p >= 0
    v = p[valid].astype(np.float64)
    lines = [f"voxel {p.size:,} / 판정 불가 {int((~valid).sum()):,} / 최종 best {int(mask.sum()):,}"]
    if v.size == 0:
        return lines + ["판정된 voxel 없음"]
    q = np.percentile(v, [10, 50, 90])
    lines.append(f"P(best) p10 {q[0]:.3f} / median {q[1]:.3f} / p90 {q[2]:.3f} / max {v.max():.3f}")
    lines.append(f"> 0.5: {int((v > 0.5).sum()):,} ({(v > 0.5).mean() * 100:.1f}%)")
    hist, edges = np.histogram(v, bins=10, range=(0.0, 1.0))
    for h, lo in zip(hist, edges[:-1]):
        lines.append(f"  {lo:.1f}~{lo + 0.1:.1f}  {h:>9,}  {h / v.size * 100:5.1f}%")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump")
    ap.add_argument("--x")
    ap.add_argument("--y")
    ap.add_argument("--z")
    ap.add_argument("--volume")
    ap.add_argument("--min-value", type=float)
    args = ap.parse_args()

    path = args.dump or max(glob.glob("resources/Experiments/stage_a_dump/stage_b_*.npz"), key=os.path.getmtime)
    d = np.load(path)
    if "p_best" not in d.files:
        raise SystemExit(f"{path}: p_best 없음 (P(best) 저장 이전 덤프)")
    p, mask = d["p_best"], d["mask"]
    box = tuple(parse_range(r, n) for r, n in zip((args.x, args.y, args.z), p.shape))
    p, mask = p[box], mask[box]
    if args.volume is not None:
        import nibabel as nib
        vol = np.asanyarray(nib.load(args.volume).dataobj)[box]
        keep = vol >= (args.min_value if args.min_value is not None else -np.inf)
        p, mask = p[keep], mask[keep]
    print(f"{os.path.basename(path)}  상자 {[(s.start, s.stop) for s in box]}")
    print("\n".join(summarize(p, mask)))


if __name__ == "__main__":
    # 자체 확인: 판정 불가(-1)는 분포에서 빠지고 0.5 초과 비율이 맞는지
    out = summarize(np.array([-1, 0.2, 0.6, 0.8], dtype=np.float16), np.array([0, 0, 1, 1], bool))
    assert out[0].startswith("voxel 4 / 판정 불가 1 / 최종 best 2") and "66.7%" in out[2], out
    main()
