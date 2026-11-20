"""ROI 결과 뷰어 — 압축을 푼 폴더에서 open_viewer.bat 더블클릭(또는 python viewer.py). numpy·matplotlib만 필요.
(scripts/export_result_package.py가 이 파일을 viewer.py로 복사한다. 저장소 코드를 import하지 말 것.)

화면
  왼쪽         볼륨 단면 + 그 단면(±0.5 voxel)의 Stage A ray 샘플, 색 = K-means cluster(빨강 = best)
               주황 선 = 최종 ROI(Stage B RF가 P(best) > 0.5로 판정한 voxel) 경계, 흰 × = best 공간 분리로 떼어낸 best 샘플
  오른쪽 위    ROI를 고른 렌더링 화면(노란 선 = SAM mask, 파랑·빨강 점 = positive·negative 클릭)
  오른쪽 아래  cluster × feature 채널 평균(K-means 입력 = 정규화 값). 마우스를 올리면 원래 값,
               행을 누르면 그 cluster만 표시(다시 누르면 전체)
조작
  슬라이더 / ←→ 1칸 / ↑↓ 10칸         단면 이동
  x·y·z 키 또는 왼쪽 아래 버튼        자르는 축
  휠 확대·축소(커서 기준) / r          원래 크기
  b best만 / m 최종 ROI 켜고 끄기
  n·p 키 또는 버튼                    다음·이전 ROI
"""
import glob
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

HERE = os.path.dirname(os.path.abspath(__file__))
AXES = "XYZ"
# 발산형: 파랑(음) — 회색(0) — 주황(양). 빨강은 best cluster 색이라 쓰지 않는다
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_orange", ["#1f5fbf", "#e6e6e6", "#d9730d"])

_fonts = {f.name for f in matplotlib.font_manager.fontManager.ttflist}   # 없는 글꼴 이름은 경고만 쌓여서 거른다
plt.rcParams["font.family"] = [f for f in ("Malgun Gothic", "AppleGothic", "NanumGothic") if f in _fonts] or ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
for _k in ("keymap.back", "keymap.forward", "keymap.pan", "keymap.home", "keymap.xscale", "keymap.yscale"):
    plt.rcParams[_k] = []      # ←→, p, r, x/y 키는 이 창 조작에 쓴다


def find_rois(root):
    return sorted(p for p in glob.glob(os.path.join(root, "data", "*", "*.npz"))
                  if os.path.basename(p) != "volume.npz")


def load_roi(path):
    d = dict(np.load(path, allow_pickle=False))
    r = {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}
    r["vox"] = r["vox"].astype(np.float32) / float(r["coord_scale"])
    r["labels"] = r["labels"].astype(np.int64)
    r["folder"], r["_index"] = os.path.dirname(path), {}
    return r


def slice_points(r, axis, z):
    """axis 방향 단면 z(±0.5 voxel)의 샘플 번호. 축마다 한 번 정렬해 두고 이분 탐색으로 꺼낸다."""
    if axis not in r["_index"]:
        key = np.rint(r["vox"][:, axis]).astype(np.int64)
        order = np.argsort(key, kind="stable")
        r["_index"][axis] = (order, key[order])
    order, keys = r["_index"][axis]
    return order[np.searchsorted(keys, z, "left"):np.searchsorted(keys, z, "right")]


def edge_rgba(mask2d):
    """2D mask의 경계 voxel만 주황인 RGBA(나머지 투명). contour보다 훨씬 빠르고 voxel 단위 경계 그대로다.
    주황 = cluster 팔레트에 없고 CT의 흰 뼈와도 구분되는 색."""
    m = np.asarray(mask2d, bool)
    edge = m.copy()
    edge[1:-1, 1:-1] &= ~(m[1:-1, 1:-1] & m[:-2, 1:-1] & m[2:, 1:-1] & m[1:-1, :-2] & m[1:-1, 2:])
    rgba = np.zeros(m.shape + (4,), np.float32)
    rgba[edge] = (1.0, 0.5, 0.0, 1.0)
    return rgba


class Viewer:
    def __init__(self, paths):
        self.paths, self.i, self.axis = paths, 0, 2
        self.best_only, self.show_mask, self.solo = True, False, None     # 시작: best만, 최종 ROI 끔
        self.volumes, self.roi, self.bg, self.last_cell = {}, None, None, None
        self.fig = plt.figure(figsize=(16, 8.6))
        f = self.fig
        self.ax_s = f.add_axes([0.03, 0.15, 0.50, 0.72])
        self.ax_r = f.add_axes([0.58, 0.52, 0.40, 0.36])
        self.ax_sw = f.add_axes([0.665, 0.10, 0.012, 0.33])     # cluster 색 칸(행 이름 옆)
        self.ax_h = f.add_axes([0.68, 0.10, 0.27, 0.33])
        self.ax_cb = f.add_axes([0.955, 0.10, 0.008, 0.33])
        self.slider = Slider(f.add_axes([0.12, 0.07, 0.38, 0.03]), "z", 0, 1, valinit=0, valstep=1)
        self.radio = RadioButtons(f.add_axes([0.03, 0.01, 0.05, 0.10]), AXES, active=2)
        self.slider.drawon = False      # 단면 이동 때 창 전체를 다시 그리지 않는다(redraw가 단면·슬라이더만 blit)
        self.check = CheckButtons(f.add_axes([0.12, 0.005, 0.15, 0.055]), ["best만 (b)", "최종 ROI (m)"],
                                  [self.best_only, self.show_mask])
        self.b_prev = Button(f.add_axes([0.30, 0.012, 0.09, 0.04]), "◀ 이전 ROI (p)")
        self.b_next = Button(f.add_axes([0.40, 0.012, 0.09, 0.04]), "다음 ROI (n) ▶")
        self.status = f.text(0.68, 0.475, "", fontsize=8, va="bottom", animated=True)

        self.slider.on_changed(lambda _: self.redraw())
        self.radio.on_clicked(lambda lab: self.set_axis(AXES.index(lab)))
        self.check.on_clicked(self.on_check)
        self.b_prev.on_clicked(lambda _: self.open(self.i - 1))
        self.b_next.on_clicked(lambda _: self.open(self.i + 1))
        f.canvas.mpl_connect("key_press_event", self.on_key)
        f.canvas.mpl_connect("scroll_event", self.on_scroll)
        f.canvas.mpl_connect("button_press_event", self.on_click)
        f.canvas.mpl_connect("motion_notify_event", self.on_move)
        f.canvas.mpl_connect("draw_event", self.on_draw)
        self.open(0)

    # ---------- ROI 열기 ----------
    def volume(self, r):
        if r["folder"] not in self.volumes:
            v = np.load(os.path.join(r["folder"], "volume.npz"))
            self.volumes[r["folder"]] = (v["volume"], v["spacing"].astype(float))
        return self.volumes[r["folder"]]

    def open(self, i):
        self.i = i % len(self.paths)
        r = self.roi = load_roi(self.paths[self.i])
        r["final_n"] = int(r["stage_b_mask"].sum()) if "stage_b_mask" in r else None
        self.vol, self.spacing = self.volume(r)
        self.solo = None
        self.draw_render()
        self.draw_heatmap()
        self.build_slice()

    def draw_render(self):
        r, ax = self.roi, self.ax_r
        ax.clear(); ax.set_xticks([]); ax.set_yticks([])
        base = os.path.join(r["folder"], r["name"])
        if not r["has_render"]:
            ax.text(0.5, 0.5, "렌더링 화면 없음", ha="center", va="center", transform=ax.transAxes)
            return
        ax.imshow(plt.imread(base + "_render.png"))
        if r["has_mask"]:
            m = plt.imread(base + "_mask.png")
            m = m[..., 0] if m.ndim == 3 else m
            if m.max() > m.min():
                ax.contour(m, levels=[(m.max() + m.min()) / 2], colors="yellow", linewidths=1.2)
        pts, types = np.asarray(r["sam_points"]).reshape(-1, 2), np.asarray(r["sam_types"]).astype(str)
        for t, color in (("positive", "#2b6cff"), ("negative", "#ff3030")):
            sel = np.char.startswith(np.char.lower(types), t[:3]) if len(types) else np.zeros(0, bool)
            if sel.any():
                ax.plot(pts[sel, 0], pts[sel, 1], "o", ms=7, mfc=color, mec="white", mew=1.2)
        ax.set_title("ROI를 고른 화면 (노란 선 = SAM mask)", fontsize=9)

    def draw_heatmap(self):
        r = self.roi
        z, k, names = np.asarray(r["heat_z"]), int(r["k"]), [str(c) for c in r["channels"]]
        lim = max(float(np.nanmax(np.abs(z))), 1e-6)
        for ax in (self.ax_h, self.ax_sw, self.ax_cb):
            ax.clear()
        im = self.ax_h.imshow(z, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        self.ax_h.set_xticks(range(len(names)))
        self.ax_h.set_xticklabels(names, rotation=90, fontsize=8)
        self.ax_h.set_yticks([])
        self.ax_h.set_xticks(np.arange(-0.5, len(names)), minor=True)
        self.ax_h.set_yticks(np.arange(-0.5, k), minor=True)
        self.ax_h.grid(which="minor", color="white", linewidth=1.5)        # 칸 사이 틈
        self.ax_h.tick_params(which="minor", length=0)
        for c in range(k):
            for j in range(len(names)):
                if np.isfinite(z[c, j]):
                    self.ax_h.text(j, c, f"{z[c, j]:.1f}", ha="center", va="center", fontsize=6.5,
                                   color="white" if abs(z[c, j]) > 0.6 * lim else "#222222")
        self.ax_h.set_title("cluster별 채널 평균 (K-means 입력 = 정규화 값)", fontsize=9)
        self.fig.colorbar(im, cax=self.ax_cb)
        self.ax_cb.tick_params(labelsize=7)

        n = np.asarray(r["counts"], float)
        self.ax_sw.imshow(np.asarray(r["colors"])[:, None, :], aspect="auto")
        self.ax_sw.set_xticks([])
        labels = [("★ best " if c == r["best"] else "") + f"c{c}  {n[c] / n.sum():.1%}  I {r['mean_intensity'][c]:.2f}"
                  for c in range(k)]
        self.ax_sw.set_yticks(range(k))
        self.ax_sw.set_yticklabels(labels, fontsize=8)
        self.solo_box = self.ax_h.add_patch(Rectangle((-0.5, -0.5), len(names), 1, fill=False, ec="black",
                                                      lw=2, visible=False))
        self.status.set_text("행 이름 = cluster · 샘플 비율 · 평균 intensity(정규화 전)   행 클릭 = 그 cluster만 표시")

    def build_slice(self):
        """축이나 ROI가 바뀌면 단면 artist를 새로 만든다(단면 이동은 데이터만 바꿈)."""
        ax, r, vol = self.ax_s, self.roi, self.vol
        self.p, self.q = [a for a in range(3) if a != self.axis]
        ax.clear(); self.bg = None
        shape2d = (vol.shape[self.q], vol.shape[self.p])
        # animated artist = 창 전체 그리기에서 빠지고 blit으로만 그려짐(단면 이동이 빠른 이유)
        self.im = ax.imshow(np.zeros(shape2d, np.uint8), cmap="gray", origin="lower", vmin=0, vmax=255,
                            interpolation="nearest", animated=True)
        self.no_edge = np.zeros(shape2d + (4,), np.float32)
        self.edge = ax.imshow(self.no_edge, origin="lower", interpolation="nearest", zorder=5, animated=True)
        self.lines = {}
        for c in range(int(r["k"])):
            color = np.asarray(r["colors"][c]) / 255.0
            self.lines[c] = ax.plot([], [], "o", ms=2.2, mec="none", alpha=0.75, color=color,
                                    zorder=3 if c == r["best"] else 2, animated=True)[0]
        self.lines["detached"] = ax.plot([], [], "x", ms=3, mew=0.8, color="white", zorder=4, animated=True)[0]
        self.full = ((-0.5, vol.shape[self.p] - 0.5), (-0.5, vol.shape[self.q] - 0.5))
        ax.set_xlim(*self.full[0]); ax.set_ylim(*self.full[1])
        ax.set_aspect(self.spacing[self.q] / self.spacing[self.p])
        ax.set_xlabel(AXES[self.p], fontsize=8); ax.set_ylabel(AXES[self.q], fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        self.title = ax.set_title("", fontsize=9)
        self.title.set_animated(True)
        n_slices = vol.shape[self.axis]
        self.slider.valmin, self.slider.valmax = 0, n_slices - 1
        self.slider.ax.set_xlim(0, n_slices - 1)
        self.slider.label.set_text(AXES[self.axis].lower())
        best = r["vox"][r["best_mask"], self.axis]           # best 샘플 가운데 단면에서 시작
        start = int(np.clip(np.rint(np.median(best)) if len(best) else n_slices // 2, 0, n_slices - 1))
        self.fig.suptitle(f"[{self.i + 1}/{len(self.paths)}]  {r['dataset']} · {r['name']}", fontsize=12)
        if int(self.slider.val) == start:
            self.redraw()
        else:
            self.slider.set_val(start)                        # → redraw

    # ---------- 그리기 ----------
    def redraw(self):
        if self.roi is None or not hasattr(self, "im"):
            return
        r, z = self.roi, int(self.slider.val)
        index = [slice(None)] * 3
        index[self.axis] = z
        self.im.set_data(self.vol[tuple(index)].T)
        idx = slice_points(r, self.axis, z)
        lab, det = r["labels"][idx], r["detached"][idx]
        for c, line in self.lines.items():
            if c == "detached":
                sel = idx[det] if self.solo in (None, r["best"]) else idx[:0]
            elif self.solo is not None and c != self.solo:
                sel = idx[:0]
            elif c == r["best"]:
                sel = idx[r["best_mask"][idx]]
            else:
                sel = idx[(lab == c) & ~det] if not self.best_only or self.solo == c else idx[:0]
            line.set_data(r["vox"][sel, self.p], r["vox"][sel, self.q])
        mask = r.get("stage_b_mask")
        self.edge.set_data(edge_rgba(mask[tuple(index)].T) if self.show_mask and mask is not None else self.no_edge)
        final = f"{r['final_n']:,} voxel" if r["final_n"] is not None else "없음"
        self.title.set_text(
            f"{AXES[self.axis].lower()} = {z}   |   k {r['k']} · seed {r['seed']} · 재시작 {r['n_init']}   |   "
            f"best c{r['best']} 샘플 {int(r['best_mask'].sum()):,}개"
            + (f" (떼어냄 {int(r['detached'].sum()):,} = 흰 ×)" if r["detached"].any() else "")
            + f"   |   최종 ROI {final} (주황 선)"
            + ("\n[best만 표시]" if self.best_only else "") + (f"\n[c{self.solo}만 표시]" if self.solo is not None else ""))
        self.blit()

    def draw_animated(self):
        arts = [self.im, self.edge, self.title, self.status, *self.lines.values()]
        for a in sorted(arts, key=lambda a: a.get_zorder()):
            self.fig.draw_artist(a)
        self.fig.draw_artist(self.slider.ax)     # drawon=False라 손잡이·값 표시는 여기서 갱신

    def on_draw(self, _):
        """창 전체를 그린 직후(처음·창 크기 변경·heatmap 변경): 단면 빼고 그린 배경을 저장하고 단면을 올린다."""
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.draw_animated()

    def blit(self):
        c = self.fig.canvas
        if self.bg is None:          # 배경이 낡음 → 전체 그리기(on_draw가 배경을 새로 저장)
            c.draw_idle()
            return
        c.restore_region(self.bg)
        self.draw_animated()
        c.blit(self.fig.bbox)

    # ---------- 조작 ----------
    def set_axis(self, axis):
        if axis != self.axis:
            self.axis = axis
            self.build_slice()

    def on_check(self, label):
        self.best_only, self.show_mask = self.check.get_status()
        self.bg = None               # 체크 표시가 바뀌었으니 전체 그리기
        self.redraw()

    def set_solo(self, c):
        self.solo = None if c == self.solo else c
        self.solo_box.set_visible(self.solo is not None)
        if self.solo is not None:
            self.solo_box.set_y(self.solo - 0.5)
        self.bg = None               # heatmap 강조 상자가 바뀌었으니 전체 그리기
        self.redraw()

    def on_key(self, e):
        step = {"left": -1, "right": 1, "down": -10, "up": 10}.get(e.key)
        if step:
            self.slider.set_val(int(np.clip(self.slider.val + step, self.slider.valmin, self.slider.valmax)))
        elif e.key in ("x", "y", "z"):
            self.radio.set_active("xyz".index(e.key))
        elif e.key in ("b", "m"):
            self.check.set_active("bm".index(e.key))
        elif e.key in ("n", "p"):
            self.open(self.i + (1 if e.key == "n" else -1))
        elif e.key == "r":
            self.ax_s.set_xlim(*self.full[0]); self.ax_s.set_ylim(*self.full[1])
            self.blit()

    def on_scroll(self, e):
        if e.inaxes is not self.ax_s:
            return
        f = 1 / 1.25 if e.button == "up" else 1.25
        (x0, x1), (y0, y1) = self.ax_s.get_xlim(), self.ax_s.get_ylim()
        self.ax_s.set_xlim(e.xdata + (x0 - e.xdata) * f, e.xdata + (x1 - e.xdata) * f)
        self.ax_s.set_ylim(e.ydata + (y0 - e.ydata) * f, e.ydata + (y1 - e.ydata) * f)
        self.blit()

    def heat_cell(self, e):
        if e.inaxes not in (self.ax_h, self.ax_sw) or e.xdata is None:
            return None
        c = int(np.rint(e.ydata))
        j = int(np.rint(e.xdata)) if e.inaxes is self.ax_h else None
        return (c, j) if 0 <= c < int(self.roi["k"]) else None

    def on_click(self, e):
        cell = self.heat_cell(e)
        if cell is not None and e.button == 1:
            self.set_solo(cell[0])

    def on_move(self, e):
        cell = self.heat_cell(e)
        if cell is None or cell[1] is None or not 0 <= cell[1] < len(self.roi["channels"]) or cell == self.last_cell:
            return
        self.last_cell = c, j = cell
        raw = self.roi["heat_raw"][c, j]
        self.status.set_text(f"c{c} · {self.roi['channels'][j]}: 정규화 평균 {self.roi['heat_z'][c, j]:+.2f}"
                             + (f"   원래 값 평균 {raw:.4g}" if np.isfinite(raw) else "   (INR 잠재값 — 원래 값 없음)"))
        self.blit()


def main():
    paths = find_rois(HERE)
    if not paths:
        raise SystemExit(f"{os.path.join(HERE, 'data')}에 ROI 파일이 없습니다.")
    print(f"ROI {len(paths)}개: " + ", ".join(os.path.relpath(p, os.path.join(HERE, 'data')) for p in paths))
    Viewer(paths)
    plt.show()


if __name__ == "__main__":
    main()
