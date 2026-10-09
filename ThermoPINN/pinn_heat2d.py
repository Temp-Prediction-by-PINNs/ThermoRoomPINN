"""
PINN - Diffusion thermique 2D instationnaire dans une piece
=============================================================

Version PARAMETRIQUE : le reseau ne predit plus la diffusion pour un seul
objet fixe (disque centre), mais pour toute une FAMILLE d'objets chauds
(ellipses de position, taille et orientation variables), en prenant ces
parametres de forme directement en entree du reseau, en plus de (x*,y*,t*).

Correspond a la meme famille geometrique que le generateur aleatoire de
heat_solver_reference.py (centre, rayons rx/ry, rotation) -- SANS les
harmoniques de deformation du contour pour l'instant (etape ulterieure).

Equation adimensionnee resolue (voir dérivation precedente) :

    dT*/dt* - (d2T*/dx*2 + d2T*/dy*2) = 0   sur [0,1]^2 x ]0, t*_max]

avec :
    x* = x/Lx, y* = y/Ly            (espace dans [0,1])
    T* = (T - T_amb)/(T_obj - T_amb) (temperature dans [0,1])
    t* = alpha * t / L^2             (temps adimensionne)

Parametres de forme (tires aleatoirement a l'entrainement, comme x*,y*,t*) :
    (cx, cy)   : centre de l'ellipse chaude, dans [0,1]^2
    (rx, ry)   : demi-axes de l'ellipse, dans [OBJ_MIN_RADIUS, OBJ_MAX_RADIUS] approx.
    angle      : rotation de l'ellipse (represente par (cos, sin) pour eviter
                 la discontinuite en 0/2*pi)

Architecture "hard-constraint" (IC et BC imposees EXACTEMENT, par construction) :

    T*_theta(x*,y*,t*,shape) = T*_IC(x*,y*,shape) + t* * D(x*,y*) * N_theta(x*,y*,t*,shape)

- T*_IC depend maintenant de `shape` (l'ellipse peut etre n'importe ou,
  n'importe quelle taille/orientation dans la plage entrainee).
- D(x*,y*) (enveloppe nulle sur les murs) NE depend PAS de shape : les murs
  sont toujours au meme endroit quel que soit l'objet.
- A t*=0 : T* = T*_IC exactement, quel que soit N_theta -> IC satisfaite,
  pour N'IMPORTE QUELLE configuration d'objet dans la plage entrainee.
- Sur les murs, pour tout t* : T* = 0 exactement (T*_IC quasi-nul loin des
  murs par construction du generateur + D=0) -> BC satisfaite.

Auteur: (à compléter par toi)
"""

import math
import torch
import torch.nn as nn
import numpy as np

# ----------------------------------------------------------------------
# 0. Configuration / parametres physiques
# ----------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)

# Parametres physiques (SI) - servent seulement a definir t*_max et la geometrie
LX, LY = 1.0, 1.0          # dimensions de la piece (m)
ALPHA = 2e-5                # diffusivite thermique de l'air (m^2/s)
T_AMB = 20.0                 # temperature ambiante (°C)
T_OBJ = 80.0                 # temperature de l'objet chaud (°C)
T_MAX_PHYS = 600.0           # horizon temporel physique (s)

T_STAR_MAX = ALPHA * T_MAX_PHYS / (LX ** 2)
print(f"[info] t*_max = {T_STAR_MAX:.4f}")

# ----------------------------------------------------------------------
# Plage des parametres de forme -- MEME distribution que le generateur
# aleatoire de heat_solver_reference.py (centre/rayons/aspect), pour rester
# coherent avec la solution de reference utilisee pour la validation.
# ----------------------------------------------------------------------
OBJ_MIN_RADIUS = 0.08
OBJ_MAX_RADIUS = 0.18
OBJ_MIN_ASPECT = 0.6
OBJ_MAX_ASPECT = 1.6

IC_SMOOTH_WIDTH_ABS = 0.01
# Largeur de transition lissee, en distance ABSOLUE (unites x*,y*), fixe
# quelle que soit la taille/forme de l'objet -- meme valeur que la version a
# objet fixe qui fonctionnait bien (rayon 0.15, largeur 0.01). Voir
# hard_ic_field() pour la justification detaillee de ce choix (vs un
# lissage normalise par rx/ry, qui devient instable pour les petits objets).
# NB : un step discontinu aurait une derivee nulle partout sous autograd
# (voir discussion precedente) -> solution triviale N_theta=0 qui satisfait
# le residu sans jamais apprendre la diffusion. D'ou ce lissage.


def sample_shape_params(n: int):
    """Tire n configurations d'objet aleatoires (centre, demi-axes, angle),
    memes lois que initial_condition() de heat_solver_reference.py (sans les
    harmoniques de deformation, gardees pour une etape ulterieure)."""
    base_radius = torch.rand(n, 1) * (OBJ_MAX_RADIUS - OBJ_MIN_RADIUS) + OBJ_MIN_RADIUS
    aspect = torch.rand(n, 1) * (OBJ_MAX_ASPECT - OBJ_MIN_ASPECT) + OBJ_MIN_ASPECT

    rx = base_radius * torch.sqrt(aspect)
    ry = base_radius / torch.sqrt(aspect)

    angle = torch.rand(n, 1) * 2 * math.pi
    cos_a = torch.cos(angle)
    sin_a = torch.sin(angle)

    # meme marge que le solveur de reference : on garde l'objet loin des murs
    max_extent = OBJ_MAX_RADIUS * OBJ_MAX_ASPECT
    cx = torch.rand(n, 1) * (LX - 2 * max_extent) + max_extent
    cy = torch.rand(n, 1) * (LY - 2 * max_extent) + max_extent

    return (cx.to(DEVICE), cy.to(DEVICE), rx.to(DEVICE), ry.to(DEVICE),
            cos_a.to(DEVICE), sin_a.to(DEVICE))


def hard_ic_field(x_star: torch.Tensor, y_star: torch.Tensor,
                   cx: torch.Tensor, cy: torch.Tensor,
                   rx: torch.Tensor, ry: torch.Tensor,
                   cos_a: torch.Tensor, sin_a: torch.Tensor) -> torch.Tensor:
    """T*_IC(x*,y*, shape) : champ indicateur lisse de l'ellipse chaude
    definie par (cx,cy,rx,ry,angle). ~1 dedans, ~0 dehors.

    IMPORTANT : le lissage est fait en DISTANCE ABSOLUE (unites x*,y*), pas
    en rayon elliptique normalise. Avec un lissage normalise (largeur
    proportionnelle a rx/ry), la largeur ABSOLUE de la transition devient
    minuscule pour les petits objets (jusqu'a OBJ_MIN_RADIUS=0.08) -> derivees
    secondes qui explosent localement -> loss/gradients instables, RAR qui
    ne cible plus que ces points extremes, entrainement bloque (observe :
    loss Adam plafonnant a ~1e5, L-BFGS qui s'arrete apres 4 iterations).
    On approxime la distance absolue au bord par (1-r_ellipse)*min(rx,ry) et
    on lisse avec une largeur FIXE (IC_SMOOTH_WIDTH_ABS), independante de la
    taille de l'objet -> raideur bornee quelle que soit la config tiree."""
    dx = x_star - cx
    dy = y_star - cy

    # rotation dans le repere de l'ellipse
    xr = cos_a * dx + sin_a * dy
    yr = -sin_a * dx + cos_a * dy

    r_ellipse = torch.sqrt((xr / rx) ** 2 + (yr / ry) ** 2 + 1e-12)
    min_radius = torch.minimum(rx, ry)
    dist_abs = (1.0 - r_ellipse) * min_radius
    return torch.sigmoid(dist_abs / IC_SMOOTH_WIDTH_ABS)


def boundary_envelope(x_star: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    """D(x*,y*) : s'annule EXACTEMENT sur les 4 bords de [0,1]^2, positif a
    l'interieur. Ne depend PAS de la forme de l'objet : les murs sont
    toujours au meme endroit."""
    return 16.0 * x_star * (1.0 - x_star) * y_star * (1.0 - y_star)


# ----------------------------------------------------------------------
# 1. Echantillonnage des points de collocation (residu + forme aleatoire)
# ----------------------------------------------------------------------

def sample_residual_points(n_res: int):
    """Points internes (x*,y*,t*) + une configuration d'objet aleatoire par
    point, pour entrainer le reseau sur toute la famille de formes/positions
    d'un coup (domain randomization)."""
    x = torch.rand(n_res, 1, requires_grad=True)
    y = torch.rand(n_res, 1, requires_grad=True)
    t = (torch.rand(n_res, 1) * T_STAR_MAX).requires_grad_(True)
    cx, cy, rx, ry, cos_a, sin_a = sample_shape_params(n_res)
    return x.to(DEVICE), y.to(DEVICE), t.to(DEVICE), cx, cy, rx, ry, cos_a, sin_a


# ---- Residual Adaptive Resampling (RAR) -- etape 3 du sujet -----------

RAR_ENABLED = True
RAR_POOL_SIZE = 80000
RAR_N_SELECT = 8000
RAR_INTERVAL = 500


def sample_candidate_pool(n_candidates: int):
    return sample_residual_points(n_candidates)


def rar_select_points(model: "PINN", n_candidates: int, n_select: int):
    """Evalue le residu sur un pool de candidats (points + formes aleatoires)
    et retourne les n_select plus difficiles."""
    pts = sample_candidate_pool(n_candidates)
    x, y, t, cx, cy, rx, ry, cos_a, sin_a = pts
    res = pde_residual(model, x, y, t, (cx, cy, rx, ry, cos_a, sin_a))
    res_abs = res.detach().abs().squeeze(-1)

    topk = torch.topk(res_abs, k=min(n_select, n_candidates)).indices

    x_sel = x[topk].detach().clone().requires_grad_(True)
    y_sel = y[topk].detach().clone().requires_grad_(True)
    t_sel = t[topk].detach().clone().requires_grad_(True)
    cx_sel = cx[topk].detach().clone()
    cy_sel = cy[topk].detach().clone()
    rx_sel = rx[topk].detach().clone()
    ry_sel = ry[topk].detach().clone()
    cos_sel = cos_a[topk].detach().clone()
    sin_sel = sin_a[topk].detach().clone()

    selected = (x_sel, y_sel, t_sel, cx_sel, cy_sel, rx_sel, ry_sel, cos_sel, sin_sel)
    return selected, res_abs[topk].mean().item(), res_abs.mean().item()


def concat_points(pts_a, pts_b):
    """Concatene deux jeux de points (tuples de 9 tenseurs alignes)."""
    return tuple(torch.cat([a, b], dim=0) for a, b in zip(pts_a, pts_b))


# ----------------------------------------------------------------------
# 2. Architecture du reseau T_theta(x*, y*, t*, shape) -> T*
# ----------------------------------------------------------------------

class PINN(nn.Module):
    """Reseau hard-constraint parametrique :
    T*_theta = T*_IC(x*,y*,shape) + t* * D(x*,y*) * N_theta(x*,y*,t*,shape)

    N_theta prend en entree (x*,y*,t*,cx,cy,rx,ry,cos_a,sin_a) -- 9 features.
    """

    N_INPUTS = 9

    def __init__(self, hidden_dim: int = 96, n_hidden_layers: int = 7):
        super().__init__()
        layers = [nn.Linear(self.N_INPUTS, hidden_dim), nn.Tanh()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)

    def raw_output(self, x, y, t, shape_params):
        cx, cy, rx, ry, cos_a, sin_a = shape_params
        inp = torch.cat([x, y, t, cx, cy, rx, ry, cos_a, sin_a], dim=1)
        return self.net(inp)

    def forward(self, x, y, t, shape_params):
        cx, cy, rx, ry, cos_a, sin_a = shape_params
        N_theta = self.raw_output(x, y, t, shape_params)
        T_ic = hard_ic_field(x, y, cx, cy, rx, ry, cos_a, sin_a)
        D = boundary_envelope(x, y)
        return T_ic + t * D * N_theta


# ----------------------------------------------------------------------
# 3. Residu de l'EDP via autograd
# ----------------------------------------------------------------------

def pde_residual(model: PINN, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                  shape_params) -> torch.Tensor:
    """Calcule dT*/dt* - (d2T*/dx*2 + d2T*/dy*2). Les derivees sont prises
    uniquement par rapport a x,y,t -- les parametres de forme sont des
    "constantes" du point de vue de l'EDP (ils ne varient pas dans le temps
    ni dans l'espace pour un point donne)."""
    T = model(x, y, t, shape_params)

    dT_dt = torch.autograd.grad(
        T, t, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]

    dT_dx = torch.autograd.grad(
        T, x, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    d2T_dx2 = torch.autograd.grad(
        dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True
    )[0]

    dT_dy = torch.autograd.grad(
        T, y, grad_outputs=torch.ones_like(T), create_graph=True
    )[0]
    d2T_dy2 = torch.autograd.grad(
        dT_dy, y, grad_outputs=torch.ones_like(dT_dy), create_graph=True
    )[0]

    residual = dT_dt - (d2T_dx2 + d2T_dy2)
    return residual


# ----------------------------------------------------------------------
# 4. Fonction de perte : L = L_res uniquement (IC/BC = hard-constraints)
# ----------------------------------------------------------------------

def compute_losses(model: PINN, res_pts):
    x, y, t, cx, cy, rx, ry, cos_a, sin_a = res_pts
    shape_params = (cx, cy, rx, ry, cos_a, sin_a)
    res = pde_residual(model, x, y, t, shape_params)
    return torch.mean(res ** 2)


# ----------------------------------------------------------------------
# 5. Entrainement hybride : Adam puis L-BFGS
# ----------------------------------------------------------------------

def train(n_res=10000,
          adam_epochs=255000, adam_lr=1e-3,
          lbfgs_epochs=252000):
    model = PINN().to(DEVICE)

    hard_pts = None
    loss_history = []

    optimizer = torch.optim.Adam(model.parameters(), lr=adam_lr)
    for epoch in range(adam_epochs):
        if RAR_ENABLED and epoch % RAR_INTERVAL == 0 and epoch > 0:
            hard_pts, mean_hard, mean_pool = rar_select_points(
                model, RAR_POOL_SIZE, RAR_N_SELECT
            )
            print(f"[RAR]  epoch {epoch:5d} | residu moyen pool={mean_pool:.3e} "
                  f"| residu moyen points selectionnes={mean_hard:.3e}")

        res_pts = sample_residual_points(n_res)
        if hard_pts is not None:
            res_pts = concat_points(res_pts, hard_pts)

        optimizer.zero_grad()
        loss = compute_losses(model, res_pts)
        loss.backward()
        # Filet de securite : meme avec le lissage en distance absolue, un
        # point RAR ponctuellement extreme (objet tres allonge/petit) peut
        # produire un gradient tres eleve sur un batch donne -> on le
        # plafonne pour eviter qu'un seul pas Adam ne destabilise tout.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        loss_history.append(("adam", epoch, loss.item()))

        if epoch % 500 == 0:
            print(f"[Adam] epoch {epoch:5d} | loss_res={loss.item():.3e} "
                  f"| n_points={res_pts[0].shape[0]}")

    if RAR_ENABLED:
        hard_pts, mean_hard, mean_pool = rar_select_points(
            model, RAR_POOL_SIZE, RAR_N_SELECT
        )
        print(f"[RAR]  avant L-BFGS | residu moyen pool={mean_pool:.3e} "
              f"| residu moyen points selectionnes={mean_hard:.3e}")

    res_pts_fixed = sample_residual_points(n_res)
    if hard_pts is not None:
        res_pts_fixed = concat_points(res_pts_fixed, hard_pts)

    lbfgs_iter_count = [0]

    def closure():
        lbfgs.zero_grad()
        loss = compute_losses(model, res_pts_fixed)
        loss.backward()
        lbfgs_iter_count[0] += 1
        loss_history.append(("lbfgs", lbfgs_iter_count[0], loss.item()))
        if lbfgs_iter_count[0] % 200 == 0:
            print(f"[L-BFGS] iter {lbfgs_iter_count[0]:5d} | loss_res={loss.item():.3e}")
        return loss

    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_epochs,
                               history_size=50, line_search_fn="strong_wolfe")
    lbfgs.step(closure)
    final_loss = compute_losses(model, res_pts_fixed)
    print(f"[L-BFGS] loss_res finale={final_loss.item():.3e} "
          f"| n_points={res_pts_fixed[0].shape[0]}")
    print(f"[L-BFGS] iterations reellement effectuees : "
          f"{lbfgs_iter_count[0]} / {lbfgs_epochs} demandees")

    import csv
    with open("./Loss/loss_history.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "step", "loss_res"])
        writer.writerows(loss_history)
    print("Historique de loss sauvegarde dans loss_history.csv")

    return model


# ----------------------------------------------------------------------
# 6. Point d'entree
# ----------------------------------------------------------------------

if __name__ == "__main__":
    trained_model = train()
    torch.save(trained_model.state_dict(), "pinn_heat2d.pt")
    print("Modèle sauvegardé dans pinn_heat2d.pt")
