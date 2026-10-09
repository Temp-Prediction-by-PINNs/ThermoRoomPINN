"""
Etape 5 -- Demonstrateur interactif (Gradio)
=============================================

Interface permettant de :
  - faire varier les parametres de l'objet chaud (centre, rayon moyen, aspect,
    orientation) ;
  - visualiser la carte de chaleur predite par le PINN via un slider temporel t ;
  - comparer a la solution de reference (differences finies, FTCS) recalculee
    a la volee pour la MEME configuration d'objet ;
  - afficher la carte d'erreur |T_pinn - T_ref| et les metriques (MSE, L2 relative).

Ce fichier ne modifie rien aux etapes precedentes : il REUTILISE
  - pinn_heat2d.py              (architecture PINN, constantes)
  - export_pinn_predictions.py  (build_shape_params, predict_frame)
  - heat_solver_reference.py    (build_grid, stable_dt, step, constantes)

Les sliders sont exprimes avec les memes parametres que ceux tires a
l'entrainement (rayon moyen + aspect -> rx, ry), de sorte que l'objet reste
toujours dans la plage sur laquelle le PINN a ete entraine.

Usage (depuis le dossier ThermoPINN/, pinn_heat2d.pt doit exister) :
    python app_gradio.py
"""

import os
import sys

# Cluster : empeche le proxy d'intercepter l'auto-verification de Gradio sur localhost
for _v in ("NO_PROXY", "no_proxy"):
    os.environ[_v] = ",".join(filter(None, [os.environ.get(_v), "127.0.0.1,localhost,0.0.0.0"]))

import numpy as np
import torch
import gradio as gr
from matplotlib.figure import Figure

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "ReferenceSolver"))

import heat_solver_reference as ref_solver
from pinn_heat2d import (PINN, DEVICE, LX, LY, ALPHA, T_AMB, T_OBJ,
                         OBJ_MIN_RADIUS, OBJ_MAX_RADIUS,
                         OBJ_MIN_ASPECT, OBJ_MAX_ASPECT)
from export_pinn_predictions import build_shape_params, predict_frame

MODEL_PATH = os.path.join(HERE, "pinn_heat2d.pt")
CMAP_TEMP = "turbo"
CMAP_ERR = "magma"

# Memes marges que le generateur de formes (objet toujours loin des murs)
MAX_EXTENT = OBJ_MAX_RADIUS * OBJ_MAX_ASPECT
CENTER_MIN, CENTER_MAX = MAX_EXTENT, LX - MAX_EXTENT


# ----------------------------------------------------------------------
# Chargement du modele (une seule fois)
# ----------------------------------------------------------------------
def _load_model() -> PINN:
    model = PINN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


MODEL = _load_model()


# ----------------------------------------------------------------------
# Calcul : reference (differences finies) + PINN pour une config d'objet
# ----------------------------------------------------------------------
def shape_from_sliders(cx, cy, base_radius, aspect, angle_deg):
    """Meme parametrisation que sample_shape_params() de pinn_heat2d.py."""
    rx = base_radius * np.sqrt(aspect)
    ry = base_radius / np.sqrt(aspect)
    return cx, cy, rx, ry, np.deg2rad(angle_deg)


def run_reference(cx, cy, rx, ry, angle):
    """Solveur FTCS de reference (reutilise step/stable_dt du solveur) avec
    la meme logique de capture de frames que run_simulation()."""
    x, y, dx, dy, X, Y = ref_solver.build_grid()

    cos_a, sin_a = np.cos(angle), np.sin(angle)
    xr = cos_a * (X - cx) + sin_a * (Y - cy)
    yr = -sin_a * (X - cx) + cos_a * (Y - cy)
    inside = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2) <= 1.0

    T = np.full_like(X, T_AMB)
    T[inside] = T_OBJ
    ref_solver.apply_dirichlet_bc(T)

    dt = ref_solver.stable_dt(dx, dy)
    n_steps = int(np.ceil(ref_solver.T_MAX / dt))
    frame_every = max(1, round(ref_solver.FRAME_DT / dt))

    frames, times = [], []
    for n in range(n_steps + 1):
        if n % frame_every == 0 or n == n_steps:
            frames.append(T.copy())
            times.append(n * dt)
        if n < n_steps:
            T = ref_solver.step(T, dx, dy, dt)
    return x, y, np.array(frames), np.array(times)


def simulate(cx, cy, base_radius, aspect, angle_deg):
    """Calcule T_ref et T_pinn sur tous les instants, + metriques par instant."""
    cx, cy, rx, ry, angle = shape_from_sliders(cx, cy, base_radius, aspect, angle_deg)
    x, y, T_ref, times = run_reference(cx, cy, rx, ry, angle)

    X_star, Y_star = np.meshgrid(x / LX, y / LY, indexing="ij")
    shape_params = build_shape_params(cx / LX, cy / LY, rx / LX, ry / LY,
                                      angle, X_star.size)
    T_pinn = np.array([
        predict_frame(MODEL, X_star, Y_star, ALPHA * t / (LX ** 2), shape_params)
        for t in times
    ])

    diff = T_pinn - T_ref
    mse_t = np.mean(diff ** 2, axis=(1, 2))
    l2_t = np.linalg.norm(diff.reshape(len(times), -1), axis=1) / \
        np.linalg.norm(T_ref.reshape(len(times), -1), axis=1)
    return {"x": x, "y": y, "t": times, "T_ref": T_ref, "T_pinn": T_pinn,
            "mse_t": mse_t, "l2_t": l2_t, "rx": rx, "ry": ry}


# ----------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------
def nearest_index(times, t):
    return int(np.argmin(np.abs(times - t)))


def make_maps_figure(sim, i):
    x, y = sim["x"], sim["y"]
    extent = [x[0], x[-1], y[0], y[-1]]
    T_ref, T_pinn = sim["T_ref"][i], sim["T_pinn"][i]
    err = np.abs(T_pinn - T_ref)

    fig = Figure(figsize=(13, 4.2), constrained_layout=True)
    axes = fig.subplots(1, 3)

    im0 = axes[0].imshow(T_ref.T, origin="lower", extent=extent,
                         cmap=CMAP_TEMP, vmin=T_AMB, vmax=T_OBJ)
    axes[0].set_title(f"Reference (differences finies) | t={sim['t'][i]:.0f}s")
    im1 = axes[1].imshow(T_pinn.T, origin="lower", extent=extent,
                         cmap=CMAP_TEMP, vmin=T_AMB, vmax=T_OBJ)
    axes[1].set_title(f"PINN | t={sim['t'][i]:.0f}s")
    im2 = axes[2].imshow(err.T, origin="lower", extent=extent,
                         cmap=CMAP_ERR, vmin=0, vmax=max(err.max(), 1e-6))
    axes[2].set_title("|T_pinn - T_ref|")

    for ax in axes:
        ax.set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="T (C)")
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="T (C)")
    fig.colorbar(im2, ax=axes[2], shrink=0.85, label="Erreur (C)")
    return fig


def make_error_figure(sim, i):
    t = sim["t"]
    fig = Figure(figsize=(11, 3.4), constrained_layout=True)
    ax_mse, ax_l2 = fig.subplots(1, 2)

    ax_mse.plot(t, sim["mse_t"], marker="o", ms=3)
    ax_mse.set_ylabel("MSE (C^2)")
    ax_mse.set_title("MSE par instant")
    ax_l2.plot(t, sim["l2_t"], marker="o", ms=3, color="tab:orange")
    ax_l2.set_ylabel("Erreur L2 relative")
    ax_l2.set_title("Erreur L2 relative par instant")

    for ax in (ax_mse, ax_l2):
        ax.axvline(t[i], color="k", ls="--", lw=1)
        ax.set_xlabel("t (s)")
        ax.grid(alpha=0.3)
    return fig


def metrics_text(sim, i):
    diff = sim["T_pinn"] - sim["T_ref"]
    mse = np.mean(diff ** 2)
    l2 = np.linalg.norm(diff) / np.linalg.norm(sim["T_ref"])
    return (
        f"**Objet** : rx = {sim['rx']:.3f} m, ry = {sim['ry']:.3f} m\n\n"
        f"**A t = {sim['t'][i]:.0f} s** : MSE = {sim['mse_t'][i]:.4f} C^2 "
        f"| L2 relative = {sim['l2_t'][i]:.2%}\n\n"
        f"**Global (tous instants)** : MSE = {mse:.4f} C^2 "
        f"| RMSE = {np.sqrt(mse):.3f} C | L2 relative = {l2:.2%} "
        f"| erreur max = {np.max(np.abs(diff)):.2f} C"
    )


def render(sim, t):
    i = nearest_index(sim["t"], t)
    return make_maps_figure(sim, i), make_error_figure(sim, i), metrics_text(sim, i)


# ----------------------------------------------------------------------
# Callbacks Gradio
# ----------------------------------------------------------------------
def on_simulate(cx, cy, base_radius, aspect, angle_deg, t):
    sim = simulate(cx, cy, base_radius, aspect, angle_deg)
    maps, errs, txt = render(sim, t)
    return sim, maps, errs, txt


def on_time_change(sim, t):
    if sim is None:
        return gr.skip(), gr.skip(), gr.skip()
    return render(sim, t)


def build_ui():
    with gr.Blocks(title="ThermoRoomPINN - Demonstrateur") as demo:
        gr.Markdown(
            "# ThermoRoomPINN - Diffusion thermique 2D par PINN\n"
            "Choisissez les parametres de l'objet chaud (80 C) dans une piece "
            "de 1 m x 1 m a 20 C (murs a 20 C), puis cliquez sur **Simuler**. "
            "Le slider temporel permet ensuite de parcourir l'evolution "
            "predite par le PINN, comparee a la solution de reference."
        )

        with gr.Row():
            with gr.Column(scale=1):
                cx = gr.Slider(CENTER_MIN, CENTER_MAX, value=0.5, step=0.01,
                               label="Centre x (m)")
                cy = gr.Slider(CENTER_MIN, CENTER_MAX, value=0.5, step=0.01,
                               label="Centre y (m)")
                radius = gr.Slider(OBJ_MIN_RADIUS, OBJ_MAX_RADIUS, value=0.15,
                                   step=0.005, label="Rayon moyen (m)")
                aspect = gr.Slider(OBJ_MIN_ASPECT, OBJ_MAX_ASPECT, value=1.0,
                                   step=0.05,
                                   label="Aspect rx/ry (1 = disque)")
                angle = gr.Slider(0, 360, value=0, step=5,
                                  label="Orientation (degres)")
                run_btn = gr.Button("Simuler", variant="primary")
            with gr.Column(scale=3):
                t_slider = gr.Slider(0, ref_solver.T_MAX, value=0,
                                     step=ref_solver.FRAME_DT,
                                     label="Temps t (s)")
                maps_plot = gr.Plot(label="Cartes de chaleur et d'erreur")
                metrics_md = gr.Markdown()

        err_plot = gr.Plot(label="Evolution de l'erreur")
        state = gr.State(None)

        run_inputs = [cx, cy, radius, aspect, angle, t_slider]
        run_outputs = [state, maps_plot, err_plot, metrics_md]
        run_btn.click(on_simulate, run_inputs, run_outputs)
        t_slider.change(on_time_change, [state, t_slider],
                        [maps_plot, err_plot, metrics_md], show_progress="hidden")
        demo.load(on_simulate, run_inputs, run_outputs)
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")