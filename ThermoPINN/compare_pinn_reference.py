"""
Comparaison PINN vs solution de reference (differences finies)
==================================================================

Charge pinn_predictions.npz et reference_solution.npz (meme grille, memes
instants -- generes respectivement par export_pinn_predictions.py et
heat_solver_reference.py) et calcule :

  - MSE globale (sur tous les points x,y,t confondus)
  - Erreur L2 relative globale : ||T_pinn - T_ref||_2 / ||T_ref||_2
  - MSE et erreur L2 relative par instant t (pour voir si l'erreur augmente
    avec le temps -> extrapolation, accumulation d'erreur, etc.)
  - Cartes d'erreur spatiale |T_pinn - T_ref| a plusieurs instants

Genere :
  - comparison_metrics.txt   (resume chiffre)
  - comparison_error_over_time.png
  - comparison_error_maps.png (cartes d'erreur + T_pinn + T_ref cote a cote)

Usage :
    python compare_pinn_reference.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PINN_NPZ = "pinn_predictions.npz"
REF_NPZ = "../ReferenceSolver/reference_solution.npz"

T_AMB = 20.0
T_OBJ = 80.0
CMAP_TEMP = "turbo"
CMAP_ERR = "magma"

N_SNAPSHOTS = 6  # nombre d'instants affiches sur la figure de cartes d'erreur


def load_and_check():
    pinn = np.load(PINN_NPZ)
    ref = np.load(REF_NPZ)

    x_pinn, y_pinn, t_pinn, T_pinn = pinn["x"], pinn["y"], pinn["t"], pinn["T"]
    x_ref, y_ref, t_ref, T_ref = ref["x"], ref["y"], ref["t"], ref["T"]

    # Verifications de coherence -- si ca ne matche pas, la comparaison
    # point-a-point n'a pas de sens (mauvaise grille, mauvais instants...)
    assert np.allclose(x_pinn, x_ref), "Grilles x differentes entre PINN et reference !"
    assert np.allclose(y_pinn, y_ref), "Grilles y differentes entre PINN et reference !"
    assert np.allclose(t_pinn, t_ref), "Instants t differents entre PINN et reference !"
    assert T_pinn.shape == T_ref.shape, (
        f"Formes differentes : PINN {T_pinn.shape} vs reference {T_ref.shape}"
    )

    return x_ref, y_ref, t_ref, T_pinn, T_ref


def compute_global_metrics(T_pinn: np.ndarray, T_ref: np.ndarray):
    """MSE et erreur L2 relative sur l'ensemble (tous instants, tous points)."""
    diff = T_pinn - T_ref
    mse = np.mean(diff ** 2)
    l2_rel = np.linalg.norm(diff) / np.linalg.norm(T_ref)
    max_abs_err = np.max(np.abs(diff))
    return mse, l2_rel, max_abs_err


def compute_per_frame_metrics(T_pinn: np.ndarray, T_ref: np.ndarray):
    """MSE et erreur L2 relative a chaque instant (axe 0 = temps)."""
    n_frames = T_pinn.shape[0]
    mse_t = np.zeros(n_frames)
    l2_rel_t = np.zeros(n_frames)

    for i in range(n_frames):
        diff = T_pinn[i] - T_ref[i]
        mse_t[i] = np.mean(diff ** 2)
        ref_norm = np.linalg.norm(T_ref[i])
        l2_rel_t[i] = np.linalg.norm(diff) / ref_norm if ref_norm > 0 else np.nan

    return mse_t, l2_rel_t


def plot_error_over_time(t: np.ndarray, mse_t: np.ndarray, l2_rel_t: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(t, mse_t, marker="o", ms=3)
    axes[0].set_xlabel("t (s)")
    axes[0].set_ylabel("MSE (C^2)")
    axes[0].set_title("MSE par instant")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, l2_rel_t, marker="o", ms=3, color="tab:orange")
    axes[1].set_xlabel("t (s)")
    axes[1].set_ylabel("Erreur L2 relative")
    axes[1].set_title("Erreur L2 relative par instant")
    axes[1].grid(alpha=0.3)

    fig.suptitle("Evolution de l'erreur PINN vs reference au cours du temps")
    fig.tight_layout()
    fig.savefig("./Result/comparison_error_over_time.png", dpi=150)
    plt.close(fig)
    print("Figure sauvegardee : comparison_error_over_time.png")


def plot_error_maps(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                     T_pinn: np.ndarray, T_ref: np.ndarray):
    n_frames = T_pinn.shape[0]
    idx = np.linspace(0, n_frames - 1, min(N_SNAPSHOTS, n_frames), dtype=int)

    n = len(idx)
    fig, axes = plt.subplots(3, n, figsize=(3 * n, 9), constrained_layout=True)
    if n == 1:
        axes = axes.reshape(3, 1)

    extent = [x[0], x[-1], y[0], y[-1]]

    for col, i in enumerate(idx):
        err_map = np.abs(T_pinn[i] - T_ref[i])

        im_ref = axes[0, col].imshow(T_ref[i].T, origin="lower", extent=extent,
                                      cmap=CMAP_TEMP, vmin=T_AMB, vmax=T_OBJ)
        axes[0, col].set_title(f"Reference | t={t[i]:.0f}s")

        im_pinn = axes[1, col].imshow(T_pinn[i].T, origin="lower", extent=extent,
                                       cmap=CMAP_TEMP, vmin=T_AMB, vmax=T_OBJ)
        axes[1, col].set_title(f"PINN | t={t[i]:.0f}s")

        im_err = axes[2, col].imshow(err_map.T, origin="lower", extent=extent,
                                      cmap=CMAP_ERR, vmin=0, vmax=err_map.max())
        axes[2, col].set_title(f"|erreur| | t={t[i]:.0f}s")

        for row in range(3):
            axes[row, col].set_xlabel("x (m)")
            if col == 0:
                axes[row, col].set_ylabel("y (m)")

    fig.colorbar(im_ref, ax=axes[0, :], shrink=0.8, label="T (C)")
    fig.colorbar(im_pinn, ax=axes[1, :], shrink=0.8, label="T (C)")
    fig.colorbar(im_err, ax=axes[2, :], shrink=0.8, label="|T_pinn - T_ref| (C)")

    fig.suptitle("Comparaison reference / PINN / carte d'erreur")
    fig.savefig("./Result/comparison_error_maps.png", dpi=150)
    plt.close(fig)
    print("Figure sauvegardee : comparison_error_maps.png")


def main():
    x, y, t, T_pinn, T_ref = load_and_check()

    mse, l2_rel, max_abs_err = compute_global_metrics(T_pinn, T_ref)
    mse_t, l2_rel_t = compute_per_frame_metrics(T_pinn, T_ref)

    summary = (
        f"Comparaison PINN vs solution de reference\n"
        f"===========================================\n"
        f"Nombre d'instants compares : {len(t)}\n"
        f"Grille spatiale             : {x.shape[0]} x {y.shape[0]}\n\n"
        f"MSE globale                 : {mse:.4f} C^2\n"
        f"RMSE globale                : {np.sqrt(mse):.4f} C\n"
        f"Erreur L2 relative globale  : {l2_rel:.4%}\n"
        f"Erreur absolue max          : {max_abs_err:.4f} C\n\n"
        f"MSE au 1er instant (t={t[0]:.1f}s)   : {mse_t[0]:.4f} C^2\n"
        f"MSE au dernier instant (t={t[-1]:.1f}s) : {mse_t[-1]:.4f} C^2\n"
        f"Erreur L2 rel. au 1er instant    : {l2_rel_t[0]:.4%}\n"
        f"Erreur L2 rel. au dernier instant: {l2_rel_t[-1]:.4%}\n"
    )

    print("\n" + summary)
    with open("./Result/comparison_metrics.txt", "w") as f:
        f.write(summary)
    print("Resume sauvegarde dans comparison_metrics.txt")

    plot_error_over_time(t, mse_t, l2_rel_t)
    plot_error_maps(x, y, t, T_pinn, T_ref)


if __name__ == "__main__":
    main()
