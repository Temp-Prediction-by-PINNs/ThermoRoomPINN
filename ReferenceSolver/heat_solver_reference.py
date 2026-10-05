"""
Solveur de reference : equation de la chaleur 2D instationnaire dans une piece.

Resout dT/dt = alpha * (d2T/dx2 + d2T/dy2) par differences finies explicites
(schema FTCS), avec :
- domaine carre Omega = [0, Lx] x [0, Ly]
- condition initiale : objet chaud a T_obj, reste de la piece a T_amb
- conditions aux limites de Dirichlet : murs a T_amb pour tout t > 0

Sert de solution de reference pour valider le PINN parametrique (comparaison
MSE / erreur L2). Genere aussi une animation (mp4 si ffmpeg est installe,
sinon gif).

MODIF vs version precedente : le champ initial peut maintenant etre soit une
simple ELLIPSE (meme famille geometrique que le PINN parametrique, sans
harmoniques), soit une forme IRREGULIERE (ellipse + harmoniques, comme avant)
-- controle par SHAPE_MODE ci-dessous. Les parametres de forme reellement
utilises (centre, rayons, angle) sont maintenant sauvegardes dans le .npz,
pour que export_pinn_predictions.py puisse donner exactement la meme
configuration au PINN.

Utilisation :
    python heat_solver_reference.py

Dependances : numpy, matplotlib (ffmpeg optionnel, pour un .mp4 plutot qu'un .gif)
"""

import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# --------------------------------------------------------------------------
# Parametres physiques et numeriques
# --------------------------------------------------------------------------
Lx, Ly = 1.0, 1.0          # dimensions de la piece (m)
NX, NY = 60, 60             # nombre de points de grille par axe
ALPHA = 2e-5                 # diffusivite thermique de l'air (m^2/s)

T_AMB = 20.0                 # temperature ambiante / murs (C)
T_OBJ = 80.0                 # temperature de l'objet chaud (C)

T_MAX = 600.0                # duree simulee (s) -- alignee sur T_MAX_PHYS du PINN
N_SNAPSHOTS = 12
PRINT_INTERVAL_S = 100.0

FRAME_DT = 10.0
VIDEO_FPS = 15

CMAP = "turbo"

OUTPUT_NPZ = "reference_solution.npz"
OUTPUT_PNG = "reference_snapshots.png"
OUTPUT_MP4 = "reference_diffusion.mp4"
OUTPUT_GIF = "reference_diffusion.gif"

# ------------------------------------------------------------------------
# SHAPE_MODE :
#   "ellipse"   -> objet = simple ellipse (centre, rayons, rotation), MEME
#                  famille geometrique que le PINN parametrique actuel.
#                  A utiliser pour toute comparaison quantitative PINN/reference.
#   "irregular" -> ellipse + harmoniques de deformation (forme "organique",
#                  comme dans les toutes premieres versions du solveur).
#                  Le PINN actuel n'a PAS ete entraine sur cette famille plus
#                  large -> a n'utiliser que pour des tests qualitatifs de
#                  robustesse/generalisation, pas pour une comparaison chiffree.
# ------------------------------------------------------------------------
SHAPE_MODE = "ellipse"

OBJ_MIN_RADIUS = 0.08
OBJ_MAX_RADIUS = 0.18
OBJ_MIN_ASPECT = 0.6
OBJ_MAX_ASPECT = 1.6

RANDOM_SEED = None   # None = different a chaque execution ; entier = reproductible

# --------------------------------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------------------------------
def build_grid():
    x = np.linspace(0, Lx, NX)
    y = np.linspace(0, Ly, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, dx, dy, X, Y


def initial_condition(X, Y):
    """
    Champ initial avec un objet chaud de position/taille/orientation
    aleatoires (ellipse), et -- si SHAPE_MODE="irregular" -- une deformation
    supplementaire du contour par harmoniques.

    Retourne (T, shape_params) ou shape_params est un dict avec au moins
    center_x, center_y, radius_x, radius_y, angle -- les memes conventions
    que sample_shape_params() dans pinn_heat2d.py, pour pouvoir donner
    exactement la meme config au PINN lors de la comparaison.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    max_radius = OBJ_MAX_RADIUS * OBJ_MAX_ASPECT
    center_x = rng.uniform(max_radius, Lx - max_radius)
    center_y = rng.uniform(max_radius, Ly - max_radius)

    base_radius = rng.uniform(OBJ_MIN_RADIUS, OBJ_MAX_RADIUS)
    aspect = rng.uniform(OBJ_MIN_ASPECT, OBJ_MAX_ASPECT)
    radius_x = base_radius * np.sqrt(aspect)
    radius_y = base_radius / np.sqrt(aspect)

    angle = rng.uniform(0, 2 * np.pi)

    dx = X - center_x
    dy = Y - center_y
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy

    r_ellipse = np.sqrt((xr / radius_x) ** 2 + (yr / radius_y) ** 2)

    if SHAPE_MODE == "irregular":
        theta = np.arctan2(yr, xr)
        n_harmonics = rng.integers(2, 5)
        deformation = np.ones_like(theta)
        for _ in range(n_harmonics):
            harmonic = rng.integers(2, 7)
            amplitude = rng.uniform(0.05, 0.20)
            phase = rng.uniform(0, 2 * np.pi)
            deformation += amplitude * np.cos(harmonic * theta + phase)
    else:  # "ellipse" -- meme famille que le PINN parametrique
        deformation = np.ones_like(r_ellipse)

    inside = r_ellipse <= deformation

    T = np.full_like(X, T_AMB)
    T[inside] = T_OBJ

    shape_params = {
        "center_x": center_x, "center_y": center_y,
        "radius_x": radius_x, "radius_y": radius_y,
        "angle": angle, "shape_mode": SHAPE_MODE,
    }
    return T, shape_params


def apply_dirichlet_bc(T):
    T[0, :] = T_AMB
    T[-1, :] = T_AMB
    T[:, 0] = T_AMB
    T[:, -1] = T_AMB
    return T


def stable_dt(dx, dy, safety=0.9):
    dt_max = 1.0 / (2 * ALPHA * (1 / dx ** 2 + 1 / dy ** 2))
    return safety * dt_max


def step(T, dx, dy, dt):
    lap = (
        (T[2:, 1:-1] - 2 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2
        + (T[1:-1, 2:] - 2 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
    )
    T_new = T.copy()
    T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + dt * ALPHA * lap
    apply_dirichlet_bc(T_new)
    return T_new


def run_simulation():
    x, y, dx, dy, X, Y = build_grid()
    dt = stable_dt(dx, dy)
    n_steps = int(np.ceil(T_MAX / dt))

    T, shape_params = initial_condition(X, Y)
    apply_dirichlet_bc(T)

    print(f"[info] Objet ({shape_params['shape_mode']}) : "
          f"centre=({shape_params['center_x']:.3f},{shape_params['center_y']:.3f}), "
          f"rayons=({shape_params['radius_x']:.3f},{shape_params['radius_y']:.3f}), "
          f"angle={shape_params['angle']:.3f} rad")

    frame_every = max(1, round(FRAME_DT / dt))
    print_every = max(1, round(PRINT_INTERVAL_S / dt))

    frames = []
    frame_times = []

    for n in range(n_steps + 1):
        take_frame = (n % frame_every == 0) or (n == n_steps)
        if take_frame:
            frames.append(T.copy())
            frame_times.append(n * dt)
        if (n % print_every == 0) or (n == n_steps):
            print(f"t = {n * dt:7.1f} s  (pas {n}/{n_steps})  "
                  f"T_max = {T.max():.1f} C  [{len(frames)} frames capturees]")
        if n < n_steps:
            T = step(T, dx, dy, dt)

    print(f"Grille {NX}x{NY}, dt = {dt:.3f} s, {n_steps} pas de temps, "
          f"t_max reel = {n_steps * dt:.1f} s, {len(frames)} frames "
          f"(1 toutes les {FRAME_DT} s)")

    return x, y, np.array(frames), np.array(frame_times), shape_params


def save_and_plot(x, y, frames, times, shape_params):
    np.savez(
        OUTPUT_NPZ, x=x, y=y, T=frames, t=times,
        center_x=shape_params["center_x"], center_y=shape_params["center_y"],
        radius_x=shape_params["radius_x"], radius_y=shape_params["radius_y"],
        angle=shape_params["angle"], shape_mode=shape_params["shape_mode"],
    )
    print(f"Solution de reference sauvegardee dans {OUTPUT_NPZ} "
          f"(shape T = {frames.shape}, shape_mode={shape_params['shape_mode']})")

    idx = np.linspace(0, len(times) - 1, N_SNAPSHOTS, dtype=int)
    snap_frames = frames[idx]
    snap_times = times[idx]

    n = len(snap_times)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.2), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, T, t in zip(axes, snap_frames, snap_times):
        im = ax.imshow(
            T.T, origin="lower", extent=[0, Lx, 0, Ly],
            cmap=CMAP, vmin=T_AMB, vmax=T_OBJ,
        )
        ax.set_title(f"t = {t:.0f} s")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Temperature (C)")
    fig.suptitle(f"Solution de reference ({shape_params['shape_mode']}) - diffusion thermique 2D")
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee dans {OUTPUT_PNG}")


def build_video(x, y, frames, times):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(
        frames[0].T, origin="lower", extent=[0, Lx, 0, Ly],
        cmap=CMAP, vmin=T_AMB, vmax=T_OBJ,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, label="Temperature (C)")
    title = ax.set_title(f"t = {times[0]:.0f} s")
    fig.tight_layout()

    def update(i):
        im.set_data(frames[i].T)
        title.set_text(f"t = {times[i]:.0f} s")
        return im, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / VIDEO_FPS, blit=False
    )

    saved = False
    if shutil.which("ffmpeg") is not None:
        candidates = [
            ("libx264", ["-pix_fmt", "yuv420p", "-crf", "20"]),
            ("mpeg4", ["-pix_fmt", "yuv420p"]),
        ]
        for codec, extra_args in candidates:
            try:
                writer = animation.FFMpegWriter(
                    fps=VIDEO_FPS, codec=codec, extra_args=extra_args
                )
                anim.save(OUTPUT_MP4, writer=writer)
                print(f"Video sauvegardee dans {OUTPUT_MP4} (codec {codec})")
                saved = True
                break
            except Exception as e:
                print(f"Codec '{codec}' indisponible ({type(e).__name__}), on essaie une alternative...")

    if not saved:
        anim.save(OUTPUT_GIF, writer=animation.PillowWriter(fps=VIDEO_FPS))
        print(f"Aucun codec mp4 compatible trouve -> animation sauvegardee en GIF dans {OUTPUT_GIF}")

    plt.close(fig)


if __name__ == "__main__":
    x, y, frames, times, shape_params = run_simulation()
    save_and_plot(x, y, frames, times, shape_params)
    build_video(x, y, frames, times)
