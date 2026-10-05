"""
Export des predictions du PINN parametrique entraine, au meme format .npz
que le solveur de reference (x, y, T, t), pour permettre une comparaison
directe (MSE, erreur L2 relative) point par point.

MODIF vs version precedente : le PINN prend maintenant en entree les
parametres de forme de l'objet (centre, rayons, angle) en plus de (x*,y*,t*).
Ce script lit ces parametres directement dans reference_solution.npz (via
les champs center_x, center_y, radius_x, radius_y, angle ajoutes par
heat_solver_reference.py) pour etre certain d'evaluer le PINN EXACTEMENT sur
la meme configuration d'objet que celle simulee par le solveur de reference.

IMPORTANT :
  - Si reference_solution.npz a ete genere avec SHAPE_MODE="irregular"
    (harmoniques), la comparaison ne sera PAS valide : le PINN n'a ete
    entraine que sur la famille "ellipse" (sans harmoniques). Ce script
    verifie ce champ et previent si besoin.
  - Les instants t du fichier de reference doivent rester <= a l'horizon
    d'entrainement du PINN (T_MAX_PHYS dans pinn_heat2d.py).

Usage :
    python export_pinn_predictions.py
"""

import numpy as np
import torch

from pinn_heat2d import PINN, LX, LY, ALPHA, T_AMB, T_OBJ, T_STAR_MAX, DEVICE

REFERENCE_NPZ = "../ReferenceSolver/reference_solution.npz"
OUTPUT_NPZ = "pinn_predictions.npz"
MODEL_PATH = "pinn_heat2d.pt"


def load_model() -> PINN:
    model = PINN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


def build_shape_params(cx: float, cy: float, rx: float, ry: float, angle: float,
                        n_points: int):
    """Construit les tenseurs de forme (constants, broadcastes a n_points)
    dans le meme format que sample_shape_params() de pinn_heat2d.py :
    (cx, cy, rx, ry, cos_a, sin_a), chacun de shape (n_points, 1)."""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    ones = torch.ones(n_points, 1, dtype=torch.float32, device=DEVICE)
    return (
        ones * cx, ones * cy, ones * rx, ones * ry,
        ones * cos_a, ones * sin_a,
    )


def predict_frame(model: PINN, x_grid_star: np.ndarray, y_grid_star: np.ndarray,
                   t_star: float, shape_params) -> np.ndarray:
    """Predit T* (puis reconverti en C) sur toute la grille, a un instant t*
    donne, pour la configuration d'objet fixee dans shape_params."""
    shape = x_grid_star.shape
    n_points = x_grid_star.size

    x_flat = torch.tensor(x_grid_star.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    y_flat = torch.tensor(y_grid_star.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    t_flat = torch.full_like(x_flat, t_star)

    with torch.no_grad():
        T_star_pred = model(x_flat, y_flat, t_flat, shape_params)

    T_star_pred = T_star_pred.cpu().numpy().reshape(shape)
    T_phys_pred = T_AMB + (T_OBJ - T_AMB) * T_star_pred
    return T_phys_pred


def main():
    ref = np.load(REFERENCE_NPZ, allow_pickle=True)
    x_phys = ref["x"]
    y_phys = ref["y"]
    t_phys = ref["t"]

    shape_mode = str(ref["shape_mode"]) if "shape_mode" in ref else "inconnu"
    if shape_mode != "ellipse":
        print(f"[ATTENTION] reference_solution.npz a ete genere avec "
              f"shape_mode='{shape_mode}', mais le PINN n'est entraine que "
              f"sur la famille 'ellipse' (sans harmoniques). La comparaison "
              f"risque d'etre invalide -- relance heat_solver_reference.py "
              f"avec SHAPE_MODE='ellipse' pour une comparaison quantitative valide.")

    cx = float(ref["center_x"])
    cy = float(ref["center_y"])
    rx = float(ref["radius_x"])
    ry = float(ref["radius_y"])
    angle = float(ref["angle"])

    print(f"Grille de reference : {x_phys.shape[0]} x {y_phys.shape[0]}, "
          f"{t_phys.shape[0]} instants (de {t_phys[0]:.1f}s a {t_phys[-1]:.1f}s)")
    print(f"Objet : centre=({cx:.3f},{cy:.3f}), rayons=({rx:.3f},{ry:.3f}), "
          f"angle={angle:.3f} rad")

    X_phys, Y_phys = np.meshgrid(x_phys, y_phys, indexing="ij")
    X_star = X_phys / LX
    Y_star = Y_phys / LY
    n_points = X_star.size

    # NB : LX=LY=1.0 dans pinn_heat2d.py -> les coordonnees physiques (m) et
    # adimensionnees sont numeriquement identiques ici, mais on adimensionne
    # quand meme explicitement pour rester correct si LX/LY changent un jour.
    shape_params = build_shape_params(cx / LX, cy / LY, rx / LX, ry / LY, angle, n_points)

    t_star_max_ref = ALPHA * t_phys[-1] / (LX ** 2)
    if t_star_max_ref > T_STAR_MAX:
        print(f"[ATTENTION] Le dernier instant de reference (t={t_phys[-1]:.1f}s, "
              f"t*={t_star_max_ref:.4f}) depasse l'horizon d'entrainement du PINN "
              f"(t*_max={T_STAR_MAX:.4f}). Extrapolation au-dela, potentiellement peu fiable.")

    model = load_model()

    frames_pred = []
    for i, t_val in enumerate(t_phys):
        t_star = ALPHA * t_val / (LX ** 2)
        T_frame = predict_frame(model, X_star, Y_star, t_star, shape_params)
        frames_pred.append(T_frame)
        if i % max(1, len(t_phys) // 10) == 0:
            print(f"  instant {i+1}/{len(t_phys)} (t={t_val:.1f}s) : "
                  f"T_min={T_frame.min():.1f}C, T_max={T_frame.max():.1f}C")

    frames_pred = np.array(frames_pred)

    np.savez(OUTPUT_NPZ, x=x_phys, y=y_phys, T=frames_pred, t=t_phys,
             center_x=cx, center_y=cy, radius_x=rx, radius_y=ry, angle=angle)
    print(f"\nPredictions PINN sauvegardees dans {OUTPUT_NPZ} "
          f"(shape T = {frames_pred.shape})")


if __name__ == "__main__":
    main()
