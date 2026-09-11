"""
Solveur de reference : equation de la chaleur 2D instationnaire dans une piece.

Resout dT/dt = alpha * (d2T/dx2 + d2T/dy2) par differences finies explicites
(schema FTCS), avec :
- domaine carre Omega = [0, Lx] x [0, Ly]
- condition initiale : objet chaud (disque) a T_obj, reste de la piece a T_amb
- conditions aux limites de Dirichlet : murs a T_amb pour tout t > 0

Sert de solution de reference pour valider le PINN (comparaison MSE / erreur L2).
Genere aussi une animation (mp4 si ffmpeg est installe, sinon gif) montrant la
diffusion de la chaleur, une frame toutes les FRAME_DT secondes simulees.

Utilisation :
    python heat_solver_reference.py

Dependances : numpy, matplotlib (ffmpeg optionnel, pour un .mp4 plutot qu'un .gif)
"""

import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")  # pas d'affichage interactif requis ; retire cette ligne
                        # si tu veux plt.show() en local avec un backend GUI
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
OBJ_CENTER = (0.5, 0.5)      # centre de l'objet (m)
OBJ_RADIUS = 0.15            # rayon du disque chaud (m)

T_MAX = 1800.0               # duree simulee (s)
N_SNAPSHOTS = 12              # nombre d'instantanes affiches sur la figure PNG
PRINT_INTERVAL_S = 100.0      # frequence des print de progression (s simulees)

FRAME_DT = 10.0               # une frame de video toutes les FRAME_DT secondes simulees
VIDEO_FPS = 15                 # images par seconde a la lecture

CMAP = "turbo"                 # palette type camera thermique : bleu (froid) -> rouge (chaud)
                                # (contrairement a 'inferno', le bas de l'echelle n'est pas noir)

OUTPUT_NPZ = "reference_solution.npz"
OUTPUT_PNG = "reference_snapshots.png"
OUTPUT_MP4 = "reference_diffusion.mp4"
OUTPUT_GIF = "reference_diffusion.gif"


def build_grid():
    x = np.linspace(0, Lx, NX)
    y = np.linspace(0, Ly, NY)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="ij")
    return x, y, dx, dy, X, Y


def initial_condition(X, Y):
    """Champ initial : T_obj dans le disque, T_amb ailleurs."""
    T = np.full_like(X, T_AMB)
    dist2 = (X - OBJ_CENTER[0]) ** 2 + (Y - OBJ_CENTER[1]) ** 2
    T[dist2 <= OBJ_RADIUS ** 2] = T_OBJ
    return T


def apply_dirichlet_bc(T):
    """Impose T = T_amb sur les 4 murs (en place)."""
    T[0, :] = T_AMB
    T[-1, :] = T_AMB
    T[:, 0] = T_AMB
    T[:, -1] = T_AMB
    return T


def stable_dt(dx, dy, safety=0.9):
    """Condition de stabilite du schema explicite (CFL diffusif 2D)."""
    dt_max = 1.0 / (2 * ALPHA * (1 / dx ** 2 + 1 / dy ** 2))
    return safety * dt_max


def step(T, dx, dy, dt):
    """Un pas de temps explicite (FTCS) sur les points interieurs."""
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

    T = initial_condition(X, Y)
    apply_dirichlet_bc(T)

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

    return x, y, np.array(frames), np.array(frame_times)


def save_and_plot(x, y, frames, times):
    np.savez(OUTPUT_NPZ, x=x, y=y, T=frames, t=times)
    print(f"Solution de reference sauvegardee dans {OUTPUT_NPZ} "
          f"(shape T = {frames.shape})")

    # sous-echantillonnage pour la figure statique (N_SNAPSHOTS panneaux)
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
    fig.suptitle("Solution de reference - diffusion thermique 2D")
    fig.savefig(OUTPUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Figure sauvegardee dans {OUTPUT_PNG}")


def build_video(x, y, frames, times):
    """Anime les frames et sauvegarde en mp4 (ffmpeg) ou, a defaut, en gif."""
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
        # on essaie plusieurs codecs : certains builds de ffmpeg (notamment
        # celui installe par conda) n'ont pas libx264 (licence GPL absente),
        # mpeg4 est lui present dans a peu pres tous les builds
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
                print(f"Video sauvegardee dans {OUTPUT_MP4} (codec {codec}, "
                      f"{len(frames)} frames, {VIDEO_FPS} fps, "
                      f"1 frame / {FRAME_DT:.0f} s simulees)")
                saved = True
                break
            except Exception as e:
                print(f"Codec '{codec}' indisponible sur cette machine "
                      f"({type(e).__name__}), on essaie une alternative...")

    if not saved:
        anim.save(OUTPUT_GIF, writer=animation.PillowWriter(fps=VIDEO_FPS))
        print(f"Aucun codec mp4 compatible trouve -> animation sauvegardee "
              f"en GIF dans {OUTPUT_GIF} a la place "
              f"({len(frames)} frames, {VIDEO_FPS} fps)")

    plt.close(fig)


if __name__ == "__main__":
    x, y, frames, times = run_simulation()
    save_and_plot(x, y, frames, times)
    build_video(x, y, frames, times)