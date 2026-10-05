"""
Trace la courbe de convergence (loss_history.csv) pour juger objectivement
si le nombre d'epochs Adam/L-BFGS choisi est suffisant (plateau atteint)
ou s'il reste de la marge de progression.

Usage :
    python plot_loss_history.py
"""

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

phases, steps, losses = [], [], []
with open("./Loss/loss_history.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        phases.append(row["phase"])
        steps.append(int(row["step"]))
        losses.append(float(row["loss_res"]))

adam_losses = [l for p, l in zip(phases, losses) if p == "adam"]
lbfgs_losses = [l for p, l in zip(phases, losses) if p == "lbfgs"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].plot(adam_losses)
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch Adam")
axes[0].set_ylabel("loss_res (echelle log)")
axes[0].set_title(f"Phase Adam ({len(adam_losses)} epochs)")
axes[0].grid(alpha=0.3, which="both")

axes[1].plot(lbfgs_losses, color="tab:orange")
axes[1].set_yscale("log")
axes[1].set_xlabel("iteration L-BFGS")
axes[1].set_ylabel("loss_res (echelle log)")
axes[1].set_title(f"Phase L-BFGS ({len(lbfgs_losses)} iterations reellement effectuees)")
axes[1].grid(alpha=0.3, which="both")

fig.suptitle("Courbe de convergence - a regarder pour juger si plus d'epochs aiderait")
fig.tight_layout()
fig.savefig("./Loss/loss_convergence.png", dpi=150)
print("Figure sauvegardee : loss_convergence.png")

# Diagnostic simple : la loss a-t-elle encore une pente nette sur le dernier
# quart de chaque phase, ou est-elle deja a plat (plateau) ?
def diagnose(name, values):
    if len(values) < 20:
        print(f"{name} : trop peu de points pour diagnostiquer.")
        return
    quarter = len(values) // 4
    start_val = sum(values[-quarter*2:-quarter]) / quarter
    end_val = sum(values[-quarter:]) / quarter
    if start_val == 0:
        return
    rel_drop = (start_val - end_val) / start_val
    print(f"{name} : baisse relative de loss sur le dernier quart = {rel_drop:.1%}")
    if rel_drop < 0.05:
        print(f"  -> Plateau probable : plus d'epochs sur cette phase apporterait peu.")
    else:
        print(f"  -> Encore en baisse significative : plus d'epochs pourrait aider.")

print()
diagnose("Adam", adam_losses)
diagnose("L-BFGS", lbfgs_losses)
