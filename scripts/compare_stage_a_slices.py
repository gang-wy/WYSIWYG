"""Stage A 덤프 여러 개의 K-means 결과를 같은 Z-slice에 나란히 띄우는 창(Z 슬라이더 하나로 같이 움직임).
Z-Slice Debug Viewer의 "K-means" source와 같은 그림(점 = Stage A 샘플, slice 두께 ±0.5 voxel, 같은 기본 색)이다.
INR/intensity 짝을 자동으로 찾아 띄우려면 experiments/pair_stage_a_dumps.py를 쓴다.

사용:
    .venv\\Scripts\\python.exe scripts\\compare_stage_a_slices.py <stage_a_A.npz> <stage_a_B.npz> [--best-only]
    볼륨은 덤프에 기록된 volume_path로 읽는다(없는 옛 덤프는 --dataset 또는 --volume).

창 조작: 슬라이더 / ←→ 1칸 / ↑↓ 10칸 / 왼쪽 아래 X·Y·Z 버튼(또는 x·y·z 키) 자르는 축(기본 Z)
        / b best만↔전체 / n·p 다음·이전 비교(여러 개일 때)
cluster 색: best = 고정색, 나머지는 기본 팔레트. 실행마다 cluster 번호가 달라 best 외의 색은 서로 대응하지 않는다.
"""
import argparse
import json
import os
import sys

import numpy as np
import matplotlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.core.dataset_presets import DATASET_PRESETS  # noqa: E402
from src.gui.data.volume_loader import VolumeLoader  # noqa: E402
from src.gui.widget.renderer_widget import cluster_colors  # noqa: E402

AXIS_NAMES = "XYZ"


def feature_kind(layout):
    """덤프 layout → ('INR' | 'intensity' | 첫 그룹 이름, 나머지 그룹 튜플)."""
    names = [str(s).split(":")[0] for s in layout]
    if any(n.startswith("inr") for n in names):
        kind = "INR"
    elif "intensity" in names:
        kind = "intensity"
    else:
        kind = names[0] if names else "?"
    rest = tuple(sorted(n for n in names if not n.startswith("inr") and n != "intensity"))   # 열 순서와 무관하게 비교
    return kind, rest


def auto_name(layout):
    kind, rest = feature_kind(layout)
    return kind + (f" (+{', '.join(rest)})" if rest else "")


def dump_volume_source(d):
    """덤프에 기록된 (볼륨 경로, raw 파라미터) — 옛 덤프는 (None, None)."""
    if "volume_path" not in d.files or not str(d["volume_path"]):
        return None, None
    raw = json.loads(str(d["raw_params"])) if "raw_params" in d.files else {}
    if raw and "shape" in raw:
        raw["shape"] = tuple(raw["shape"])
        raw["voxel_spacing"] = tuple(raw.get("voxel_spacing", (1.0, 1.0, 1.0)))
    path = str(d["volume_path"])
    return (path if os.path.isabs(path) else os.path.join(ROOT, path)), (raw or None)


def preset_for_volume(vpath):
    """볼륨 경로가 프리셋 볼륨과 같으면 프리셋 이름."""
    if not vpath:
        return None
    norm = os.path.normcase(os.path.abspath(vpath))
    for key, preset in DATASET_PRESETS.items():
        if os.path.normcase(os.path.abspath(os.path.join(ROOT, preset["volume"]))) == norm:
            return key
    return None


def load_volume(dataset=None, volume=None, raw=None):
    if dataset:
        preset = DATASET_PRESETS[dataset]
        return VolumeLoader().load(os.path.join(ROOT, preset["volume"]), preset.get("raw"))
    if volume:
        return VolumeLoader().load(volume, raw)
    raise ValueError("볼륨을 알 수 없습니다(--dataset 또는 --volume).")


def model_to_world_of(path):
    d = np.load(path)
    if "model_to_world" in d.files:
        return d["model_to_world"].astype(np.float64)
    b_path = os.path.join(os.path.dirname(path), os.path.basename(path).replace("stage_a_", "stage_b_"))
    if os.path.exists(b_path) and "model_to_world" in np.load(b_path).files:
        return np.load(b_path)["model_to_world"].astype(np.float64)
    return np.eye(4)


def load_run(path, spacing, name=None):
    d = np.load(path)
    pts = d["world_points"].astype(np.float64)
    inv = np.linalg.inv(model_to_world_of(path))
    pts = (np.hstack([pts, np.ones((len(pts), 1))]) @ inv.T)[:, :3]          # world → 볼륨 mm
    labels, best = np.asarray(d["labels"]), int(d["best"])
    layout = [str(s) for s in d["layout"]]
    # "best 공간 분리"를 켰으면 best_cluster_mask = 분리 후 남은 best, detached_best_mask = 떼어낸 샘플(라벨은 그대로 best)
    best_mask = d["best_cluster_mask"].astype(bool) if "best_cluster_mask" in d.files else labels == best
    detached = d["detached_best_mask"].astype(bool) if "detached_best_mask" in d.files else np.zeros(len(labels), bool)
    return {
        "file": os.path.basename(path), "name": name or auto_name(layout),
        "vox": pts / np.asarray(spacing, dtype=np.float64), "labels": labels, "best": best,
        "best_mask": best_mask, "detached": detached,
        "k": len(d["centroids"]), "seed": int(d["seed"]), "best_n": int(best_mask.sum()), "detached_n": int(detached.sum()),
        "layout": ", ".join(s.split(":")[0] for s in layout),
    }


def slice_range(vol, runs, axis):
    idx = np.concatenate([np.round(r["vox"][:, axis]) for r in runs])
    return int(max(0, idx.min())), int(min(vol.shape[axis] - 1, idx.max()))


def slice_points(r, axis, z):
    """run r에서 axis 방향 슬라이스 z(±0.5 voxel)의 점 번호. 축마다 한 번 정렬해 두고 이분 탐색으로 꺼낸다."""
    cache = r.setdefault("_slice_index", {})
    if axis not in cache:
        key = np.rint(r["vox"][:, axis]).astype(np.int64)
        order = np.argsort(key, kind="stable")
        cache[axis] = (order, key[order])
    order, keys = cache[axis]
    return order[np.searchsorted(keys, z, "left"):np.searchsorted(keys, z, "right")]


class SlicePanels:
    """패널 artist(배경 이미지·점·제목)를 축이 바뀔 때만 새로 만들고, 슬라이스 이동은 데이터만 바꾼다."""

    def __init__(self, axs, vol, runs, axis, spacing):
        self.vol, self.runs, self.axis = vol, runs, axis
        self.p, self.q = [a for a in range(3) if a != axis]
        vmin, vmax = float(vol.min()), float(vol.max())
        self.artists = []
        for ax, r in zip(axs, runs):
            ax.clear()
            im = ax.imshow(np.zeros((vol.shape[self.q], vol.shape[self.p])), cmap="gray", origin="lower",
                           vmin=vmin, vmax=vmax)
            # 점: cluster마다 색 하나인 marker 선(점마다 색을 주는 scatter보다 훨씬 빨리 그려짐)
            lines = {}
            for c in np.unique(r["labels"]):
                color = cluster_colors(np.array([c]), r["best"])[0] / 255.0
                lines[int(c)] = ax.plot([], [], "o", ms=2, mec="none", alpha=0.7, color=color,
                                        zorder=3 if c == r["best"] else 2)[0]
            # best 공간 분리로 떼어낸 샘플: 흰색 ×(팔레트에 없는 표시)
            lines["detached"] = ax.plot([], [], "x", ms=3, mew=0.8, color="white", alpha=0.9, zorder=4)[0]
            title = ax.set_title("", fontsize=9)
            ax.set_xlim(-0.5, vol.shape[self.p] - 0.5); ax.set_ylim(-0.5, vol.shape[self.q] - 0.5)
            ax.set_aspect(spacing[self.q] / spacing[self.p])     # voxel 간격이 축마다 다르면 실제 비율로
            ax.set_xlabel(AXIS_NAMES[self.p], fontsize=8); ax.set_ylabel(AXIS_NAMES[self.q], fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            self.artists.append((im, lines, title))

    def update(self, z, best_only):
        """axis에 수직인 단면 z. 가로·세로 = 나머지 두 축(번호 순서), 원점 왼쪽 아래 — Z 단면은 가로 X, 세로 Y."""
        index = [slice(None)] * 3
        index[self.axis] = z
        sl = self.vol[tuple(index)].T                # 복사 없는 view(np.take는 매번 복사해 느림)
        for (im, lines, title), r in zip(self.artists, self.runs):
            im.set_data(sl)
            idx = slice_points(r, self.axis, z)
            lab, det = r["labels"][idx], r["detached"][idx]
            for c, line in lines.items():
                if c == "detached":
                    sel = idx[det]
                elif c == r["best"]:
                    sel = idx[r["best_mask"][idx]]
                else:
                    sel = idx[(lab == c) & ~det] if not best_only else idx[:0]
                line.set_data(r["vox"][sel, self.p], r["vox"][sel, self.q])
            title.set_text(f"{r['name']}  {AXIS_NAMES[self.axis].lower()}={z}\nk={r['k']} seed={r['seed']} "
                           f"best={r['best']} ({r['best_n']:,}개"
                           + (f", 떼어냄 {r['detached_n']:,}개 = 흰 ×" if r["detached_n"] else "") + ")")


def show_window(items, best_only=False):
    """비교 창. items = [(제목, 로더)], 로더() → (vol, spacing, runs). n/p로 넘길 때 그때 로드한다."""
    matplotlib.use("QtAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons, Slider
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    for k in ("keymap.back", "keymap.forward", "keymap.pan", "keymap.xscale", "keymap.yscale"):
        plt.rcParams[k] = []                         # ←/→, p, x/y 키를 이 창 조작에 쓴다

    cache = {}
    st = {"i": 0, "best_only": best_only, "axis": 2, "panels": None}

    def current():
        if st["i"] not in cache:
            cache[st["i"]] = items[st["i"]][1]()
        return cache[st["i"]]

    n = len(current()[2])
    fig, axs = plt.subplots(1, n, figsize=(5.2 * n, 5.8), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.12, wspace=0.08)
    slider = Slider(fig.add_axes([0.16, 0.03, 0.72, 0.035]), "z", 0, 1, valinit=0, valstep=1)
    radio = RadioButtons(fig.add_axes([0.01, 0.01, 0.07, 0.09]), ("X", "Y", "Z"), active=2)   # 자르는 축

    def set_suptitle():
        head = (f"[{st['i'] + 1}/{len(items)}] " if len(items) > 1 else "") + items[st["i"]][0]
        fig.suptitle(f"{head}\n" + ("best만 표시" if st["best_only"] else "전체 cluster 표시")
                     + "   (←/→ 1칸, ↑/↓ 10칸, 왼쪽 아래 X/Y/Z 또는 x/y/z 키로 자르는 축, b 전환"
                     + (", n/p 다음·이전" if len(items) > 1 else "") + ")", fontsize=9)

    def set_range():
        """축이나 비교 대상이 바뀌면 패널을 새로 만들고 슬라이더 범위를 맞춘다."""
        vol, spacing, runs = current()
        st["panels"] = SlicePanels(axs[0], vol, runs, st["axis"], spacing)
        lo, hi = slice_range(vol, runs, st["axis"])
        slider.valmin, slider.valmax = lo, hi
        slider.ax.set_xlim(lo, hi)
        slider.label.set_text(AXIS_NAMES[st["axis"]].lower())
        set_suptitle()
        slider.set_val((lo + hi) // 2)
        redraw()

    def redraw(_=None):
        st["panels"].update(int(slider.val), st["best_only"])
        fig.canvas.draw_idle()

    def set_axis(axis):
        if axis != st["axis"]:
            st["axis"] = axis
            set_range()

    def on_key(event):
        step = {"left": -1, "right": 1, "down": -10, "up": 10}.get(event.key)
        if step:
            slider.set_val(int(np.clip(slider.val + step, slider.valmin, slider.valmax)))
        elif event.key == "b":
            st["best_only"] = not st["best_only"]
            set_suptitle()
            redraw()
        elif event.key in ("x", "y", "z"):
            radio.set_active("xyz".index(event.key))   # → set_axis
        elif event.key in ("n", "p") and len(items) > 1:
            st["i"] = (st["i"] + (1 if event.key == "n" else -1)) % len(items)
            set_range()

    radio.on_clicked(lambda label: set_axis("XYZ".index(label)))

    slider.on_changed(lambda _: redraw() if st["panels"] is not None else None)
    fig.canvas.mpl_connect("key_press_event", on_key)
    set_range()
    plt.show()


def open_dumps(dumps, names=None, dataset=None, volume=None, best_only=False, title=None):
    """덤프들을 읽어 비교 창을 연다(pair_stage_a_dumps.py가 만든 짝 파일도 이 함수를 부른다)."""
    if dataset or volume:
        vol, spacing = load_volume(dataset, volume)
    else:
        vpath, raw = dump_volume_source(np.load(dumps[0]))
        if vpath is None:
            sys.exit("이 덤프에는 볼륨 경로가 없습니다(옛 덤프). --dataset 또는 --volume을 지정하세요.")
        vol, spacing = load_volume(volume=vpath, raw=raw)

    names = names or [None] * len(dumps)
    runs = [load_run(p, spacing, n) for p, n in zip(dumps, names)]
    for r in runs:
        print(f"{r['file']}: {r['name']} | 샘플 {len(r['labels']):,} | k {r['k']} seed {r['seed']} "
              f"best {r['best']} ({r['best_n']:,}개)")
    if len({len(r["labels"]) for r in runs}) > 1:
        print("⚠️ 덤프마다 샘플 수가 다릅니다 — SAM mask/ROI가 달라 같은 샘플 비교가 아닙니다.")
    show_window([(title or " vs ".join(r["file"] for r in runs), lambda: (vol, spacing, runs))], best_only)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--dataset", choices=sorted(DATASET_PRESETS), help="볼륨 경로가 없는 옛 덤프용")
    ap.add_argument("--volume", help="--dataset 대신 볼륨 경로")
    ap.add_argument("--names", nargs="+", help="패널 제목(기본: 덤프의 feature 구성)")
    ap.add_argument("--best-only", action="store_true", help="best만 표시한 상태로 시작")
    ap.add_argument("--title", help="창 위 제목")
    args = ap.parse_args()
    open_dumps(args.dumps, args.names, args.dataset, args.volume, args.best_only, args.title)


if __name__ == "__main__":
    main()
