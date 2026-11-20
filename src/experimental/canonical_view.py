"""깊이 구간(depth_slabs) 결과를 보여주는 창: ray × 깊이 opacity 지도 + canonical 곡선 + 잘린 구간."""
import numpy as np
import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QDialog, QVBoxLayout

SLAB_COLORS = ["#fbbc04", "#34a853", "#a142f4", "#00acc1", "#ff6d01", "#f06292", "#8d6e63", "#1a73e8"]


class CanonicalView(QDialog):
    """main_window가 하나 만들어 재사용한다(Run 깊이 구간마다 update)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("깊이 구간 — canonical")
        self.resize(1100, 720)
        matplotlib.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
        self.fig = Figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def update_result(self, out, opacity, valid, title=""):
        """out: depth_slabs.run 결과, opacity/valid: (R, N)."""
        can, slabs = out["canonical"], out["slabs"]
        bins = can["bin"]
        R = bins.shape[0]
        B = len(can["op_mean"])
        # ray × bin opacity 지도 (ray는 처음 opacity>0인 깊이 순)
        s = np.zeros((R, B))
        c = np.zeros((R, B))
        rr = np.repeat(np.arange(R), bins.shape[1]).reshape(bins.shape)
        np.add.at(s, (rr[valid], bins[valid]), opacity[valid])
        np.add.at(c, (rr[valid], bins[valid]), 1)
        grid = s / np.maximum(c, 1)
        hit = grid > 0
        first = np.where(hit.any(axis=1), np.argmax(hit, axis=1), B)
        order = np.argsort(first, kind="stable")
        nz = np.flatnonzero(can["smooth"] > 1e-4)
        lo, hi = (max(0, nz.min() - 10), min(B - 1, nz.max() + 10)) if len(nz) else (0, B - 1)
        best_slab = out["clusters"][0]["slab"] if out["clusters"] else None
        thr, depth = out["params"]

        self.fig.clear()
        ax0, ax1 = self.fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
        img = grid[order][:, lo:hi + 1]
        vmax = np.percentile(img[img > 0], 99) if (img > 0).any() else 1.0
        ax0.imshow(img, aspect="auto", cmap="magma", vmax=vmax, extent=(lo - 0.5, hi + 0.5, R, 0))
        ax0.set_ylabel("ray (처음 닿는 깊이 순)")
        ax0.set_title(f"{title}  ray × 시선축 깊이 opacity", fontsize=10, loc="left")

        x = np.arange(lo, hi + 1)
        ax1.plot(x, can["op_mean"][lo:hi + 1], color="#9aa0a6", lw=0.8, label="opacity 평균")
        ax1.plot(x, can["smooth"][lo:hi + 1], color="#1a73e8", lw=1.5, label="opacity 평균 (smoothing, 자르기 기준)")
        ax1.axhline(thr, color="#c5221f", lw=0.8, ls="--", label=f"빈 곳 기준 {thr:g}")
        for si, (a, b) in enumerate(slabs):
            col = SLAB_COLORS[si % len(SLAB_COLORS)]
            for ax in (ax0, ax1):
                ax.axvspan(a - 0.5, b - 0.5, color=col, alpha=0.18 if ax is ax1 else 0.0, lw=0)
                ax.axvline(a - 0.5, color=col, lw=1.2)
                ax.axvline(b - 0.5, color=col, lw=1.2)
            star = " ★best" if si == best_slab else ""
            ax1.text((a + b) / 2, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 0, f"구간 {si}{star}",
                     ha="center", va="top", fontsize=9, color=col)
        ax1.set_xlabel(f"시선축 깊이 (bin = 샘플 간격 {can['step']:.2f})")
        ax1.set_title(f"canonical — 구간 {len(slabs)}개 (빈 곳 ≤ {thr:g}, 골 깊이 ≥ {depth:g})", fontsize=10, loc="left")
        ax1.legend(fontsize=8, loc="upper right")
        self.fig.tight_layout()
        self.canvas.draw_idle()
