# phase_plot.py
import numpy as np
from functions_parameters.universal_parameters import threshold
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors, cm
from matplotlib.path import Path
from dataclasses import dataclass

# ------------------------- config -------------------------

@dataclass
class PlotConfig:
    """
    Paper-wide, fixed plotting configuration.

    Set these ONCE (or pass when calling) so all figures are comparable.
    """
    # --- nematic color scale (face colors) ---
    threshold: float = threshold                  # symmetric/nematic split
    nem_vmin: float = threshold                  # lower bound for color scale (usually == threshold)
    nem_vmax: float = 1.2                 # upper bound for color scale (FIXED across figures)
    cmap_name: str = "plasma"                # base cmap for nematicity
    cmap_lo: float = 0.5                    # truncate dark end (0..1)
    cmap_hi: float = 0.95                    # truncate light end (0..1)
    under_color: str = "#A6CEE3"             # face color for values < nem_vmin (symmetric)

    # --- dot size from energy (area, pt^2) ---
    size_min: float = 49                # smallest marker area (≈ 6 pt diameter)
    size_max: float = 250                # largest marker area (≈ 14 pt diameter)
    energy_min: float = 1E-8                 # FIXED global min |E| (set for the whole paper)
    energy_max: float = 0.2               # FIXED global max |E| (set for the whole paper)

    # --- ring (magnetism) thickness from f_measure (points) ---
    # thickness grows only OUTWARD around the marker (uniform circular ring)
    m_min: float = threshold                       # FIXED global min of magnetization metric
    m_max: float = 0.5                      # FIXED global max of magnetization metric
    t_min: float = 1.6                     # thinnest ring (points)
    t_max: float = 5                       # thickest ring (points)
    ring_shape: str = "o"                    # outside ring shape: 'o' circle, 's' square, 'D' diamond

    # --- ring classes (edge colors) ---
    color_fm: str = "#8C6D1F"              # dark navy for FM
    color_ferri_afm: str = "#B4B4B4"         # dark brown for Ferri/AFM
    color_masking: str = '#FFFFFF'         # white for masking


    # --- outline for base markers (helps on printers) ---
    base_outline_color: str = "black"
    base_outline_lw: float = 0.5

    # --- colorbar ---
    cbar_extend_min: bool = True
    cbar_extendfrac: float = 0.04            # triangle size; use extendrect=True for flat cap
    cbar_extendrect: bool = False
    cbar_shrink: float = 0.9
    cbar_aspect: int = 30
    cbar_pad: float = 0.01

# --------------------- colormap helpers --------------------

def truncated_cmap(name: str, lo: float, hi: float, under: str | None):
    """Slice a Matplotlib cmap to [lo..hi] and optionally set an 'under' color."""
    base = cm.get_cmap(name)
    new = mcolors.ListedColormap(base(np.linspace(lo, hi, 256)), name=f"{name}_trunc")
    return new.with_extremes(under=under) if under is not None else new

def make_face_cmap_norm(cfg: PlotConfig):
    """Return (cmap, norm) for nematic face colors with a fixed global range."""
    cmap = truncated_cmap(cfg.cmap_name, cfg.cmap_lo, cfg.cmap_hi, cfg.under_color)
    norm = mcolors.Normalize(vmin=cfg.nem_vmin, vmax=cfg.nem_vmax)
    return cmap, norm

# ---------------------- size / thickness -------------------

def size_from_energy_fixed(E, cfg: PlotConfig):
    """
    Map |energy| -> marker area (pt^2) using a FIXED global range [energy_min, energy_max].
    Values outside the range are clipped.
    """
    E = np.asarray(E, float)
    t = 0.0 if cfg.energy_max == cfg.energy_min else np.clip((E - cfg.energy_min) / (cfg.energy_max - cfg.energy_min), 0, 1)
    return cfg.size_min + t * (cfg.size_max - cfg.size_min)

def thickness_pts_from_m_fixed(m, cfg: PlotConfig):
    """
    Map f_measure (or f_max_measure) -> outward ring thickness in POINTS using a FIXED range
    [m_min, m_max]. Below cfg.threshold no ring is drawn.
    """
    m = np.asarray(m, float)
    t = 0.0 if cfg.m_max == cfg.m_min else np.clip((m - cfg.m_min) / (cfg.m_max - cfg.m_min), 0, 1)
    thickness = cfg.t_min + t * (cfg.t_max - cfg.t_min)
    return np.where(m >= cfg.threshold, thickness, 0.0)

def ring_sizes_from_fill(s_fill, thick_pts):
    """
    Convert base fill size (pt^2) and outward thickness (pt) to:
      s_outer  - area for the outer ring circle (stroke centered on r_outer)
      s_inner  - area for the inner masking circle (exact fill radius)
    The visible ring width becomes exactly 'thick_pts' after masking.
    """
    r_fill = np.sqrt(np.asarray(s_fill, float) / np.pi)     # points
    t      = np.asarray(thick_pts, float)
    r_outer = r_fill + t
    s_outer = np.pi * r_outer**2
    s_inner = np.pi * r_fill**2
    return s_outer, s_inner

# ---------------------- shape markers ---------------------

def rect_marker(w=1.8, h=0.8):
    """Rectangle Path marker (same as your current one)."""
    w2, h2 = w / 2, h / 2
    verts = [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2), (-w2, -h2)]
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    return Path(verts, codes)

# ---------------------- main plotting ---------------------

def plot_phase_map(
    fig,
    ax: plt.Axes,
    v1_mesh: np.ndarray,
    v2_mesh: np.ndarray,
    sorted_ground_state_e: np.ndarray,   # [..., 0] used for |E|
    nematic_measure: np.ndarray,         # face color scalar
    ts_check: np.ndarray,                # 0: 1×1, 1: 1×2 stripe, 2: 2×2
    f_phase: np.ndarray,                 # 0: none, 1: FM, 2: Ferri/AFM
    f_strength: np.ndarray,              # f_measure or f_max_measure (choose upstream)
    cfg: PlotConfig = PlotConfig(),
    draw_badge_2x2: bool = False
):
    """
    Plot your phase diagram with:
      - face color = fixed-range sequential cmap of nematic_measure
      - dot size   = fixed-range mapping of |ground state energy|
      - uniform outside ring (circle) = magnetic class (color) + thickness from f_strength
      - base shape: circle (1×1), rectangle (stripe), 'P' (2×2), no edges
    """
    # flatten once
    x = v1_mesh.ravel()
    y = v2_mesh.ravel()
    E = np.abs(sorted_ground_state_e[..., 0]).ravel()
    phi = nematic_measure.ravel()
    ts = ts_check.ravel()
    fp = f_phase.ravel()
    fm = f_strength.ravel()

    # masks for shapes
    is_11 = (ts == 0)
    is_12 = (ts == 1)
    is_22 = (ts == 2)

    # fixed-range face mapping
    cmap, norm = make_face_cmap_norm(cfg)
    face_rgba = cmap(norm(phi))                    # (N,4)

    # fixed-range marker sizes from energy
    s_fill = size_from_energy_fixed(E, cfg)

    # fixed-range ring thickness from magnetism
    thick = thickness_pts_from_m_fixed(fm, cfg)
    has_ring = thick > 0
    s_outer, s_inner = ring_sizes_from_fill(s_fill, thick)

    # ring colors by magnetic class (1-D list!)
    ring_color = np.full(x.size, "none", dtype=object)
    ring_color[fp == 1] = cfg.color_fm
    ring_color[fp == 2] = cfg.color_ferri_afm
    ring_col_sel = np.asarray(ring_color, dtype=object)[has_ring].tolist()


    # ---------- Layer 1: OUTSIDE RING (stroke-only circle bigger than fill) ----------
    # (we keep ring shape circular for uniform perception; change marker in scatter below for other shapes)
    # ax.scatter(x[has_ring], y[has_ring], s=s_outer[has_ring],
    #            facecolors="none", edgecolors=ring_col_sel,
    #            linewidths=2 * thick[has_ring], marker=cfg.ring_shape, zorder=2.0)
    ax.scatter(x[has_ring], y[has_ring], s=s_outer[has_ring],
               c=ring_col_sel, marker=cfg.ring_shape, edgecolors=cfg.base_outline_color,
                    linewidths=cfg.base_outline_lw, zorder=2.0)

    # ---------- Layer 2: INNER MASK CIRCLE (exact fill radius, same face RGBA) -------
    ax.scatter(x[has_ring], y[has_ring], s=s_inner[has_ring],
               facecolors=cfg.color_masking, edgecolors=cfg.base_outline_color,
                    linewidths=cfg.base_outline_lw,
               marker="o", zorder=2.2)

    # ---------- Layer 3: BASE FILL (shape-coded), no edges ---------------------------
    # circles (1×1)
    sc = ax.scatter(x[is_11], y[is_11], s=s_fill[is_11], c=phi[is_11],
                    cmap=cmap, norm=norm, edgecolors=cfg.base_outline_color,
                    linewidths=cfg.base_outline_lw, marker="o", zorder=3)

    # rectangles (stripe)
    ax.scatter(x[is_12], y[is_12], s=s_fill[is_12], c=phi[is_12],
               cmap=cmap, norm=norm, edgecolors=cfg.base_outline_color,
               linewidths=cfg.base_outline_lw, marker=rect_marker(), zorder=3)

    # 'P' marker (2×2)
    ax.scatter(x[is_22], y[is_22], s=s_fill[is_22],c=phi[is_22],
               cmap=cmap, norm=norm, edgecolors=cfg.base_outline_color,
               linewidths=cfg.base_outline_lw, marker="P", zorder=3)

    # # optional tiny badge for 2×2 (uncomment if you want the 'x' overlay)
    # if draw_badge_2x2 and np.any(is_22):
    #     ax.scatter(x[is_22], y[is_22], s=s_fill[is_22] * 0.28,
    #                c="k", marker="x", linewidths=0.9, zorder=4)

    # colorbar (tied to sc)
    if cfg.cbar_extend_min:
        cb = fig.colorbar(sc, ax=ax,
                          extend="min", extendfrac=cfg.cbar_extendfrac,
                          shrink=cfg.cbar_shrink, pad=cfg.cbar_pad, aspect=cfg.cbar_aspect)
    else:
        cb = fig.colorbar(sc, ax=ax, shrink=cfg.cbar_shrink, pad=cfg.cbar_pad, aspect=cfg.cbar_aspect)
    if cfg.cbar_extendrect:
        cb.extend = "min"  # needed for some backends
        cb.solids.set_edgecolor("face")
    cb.set_label("nematic measure")

    # ax.set_xlabel("v1")
    # ax.set_ylabel("v2")
    return sc  # return the mappable if the caller wants more control
