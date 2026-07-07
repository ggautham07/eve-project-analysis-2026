from os import path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Cost function for the fit
def cost(d, test_data, ref_data):
    return np.sum((np.abs(ref_data - (test_data + d))))

# From Brückner et al.
genomic_dists = np.array([58, 82, 88, 149, 190, 595, 3327])
genomic_dists_log = np.log10(genomic_dists)

# Pass unique simulation tag (like "00_Rouse") to identify the simulated data
args = sys.argv
tag = args[1]
analyses_dir = f"./analyses/{tag}"

# Load data computed from simulations
with h5py.File(path.join(analyses_dir, "analysis.h5"), mode="a") as handler:
    tau = handler["Tau(s)"][:] * 800    # to include 800 as the length of each t.u. in time steps

# Experimental data given by David Brückner
tau_exp = [39.53656951044747103,
           72.07541654553817523,
           68.56142461161556412,
           100.3457588639866600,
           128.6890127669227297,
           190.2885358950472892,
           742.8575441880935841]

# Perform fitting with experimental data
optim_results = []
sigma = []
for _ in range(10):
    init_guess = np.random.randint(low=-1, high=1, size=1)
    result = minimize(fun=cost, x0=init_guess, args=(np.log10(tau_exp), np.log10(tau),), method="L-BFGS-B")
    optim_results.append(result)
    sigma.append(result["fun"])
result = optim_results[np.argmin(sigma)]
d_fit = np.squeeze(result["x"])     # Scaling factor
sigma = result["fun"]   # Deviation from experiment

slope_exp, intercept1 = np.polyfit(genomic_dists_log, np.log10(tau_exp), deg=1)     # Experiment scaling exponent
slope_sim, intercept2 = np.polyfit(genomic_dists_log, np.log10(tau), deg=1)     # Simulation scaling exponent   

# Parameters for plotting
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 48}
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#009ba5", "#008f4c", "#0085b6", "#004e9e", "#00177d", "#00053d"]

# Plot a manuscript-ready figure and save it
fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$s$    (kb)", fontdict=fd, labelpad=24)
ax.set_ylabel("$\\tau$    ($\mathregular{t.u.}$)", fontdict=fd, labelpad=24)
ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.scatter(genomic_dists, 10 ** (np.log10(tau_exp) + d_fit),
           s=480, marker="o", color=default_colours[7], edgecolor="#000",
           label=f"Experiment", zorder=10)
ax.scatter(genomic_dists, 10 ** np.log10(tau),
           s=300, marker="X", color=default_colours[1], edgecolor="#000",
           label=f"Simulation", zorder=11)
pt_l, pt_r = ax.get_xlim()
pt_range = np.linspace(pt_l, pt_r, 20)
ax.plot(pt_range, pt_range ** slope_sim * (10 ** intercept2),
        "--", linewidth=4, color=default_colours[1])
ax.plot(pt_range, ((pt_range ** slope_exp) * (10 ** intercept1)) * (10 ** d_fit),
        "--", linewidth=4, color=default_colours[7])
ax.plot(pt_range, pt_range ** 2 * (10 ** (intercept2-1.5)),
         linestyle=(0, (1, 1)), linewidth=3, color=default_colours[0]+"99", label="ideal chain")
ax.plot(pt_range, pt_range ** (5 / 3) * (10 ** (intercept2-1.5)),
         linestyle=(0, (3, 1, 1, 1)), linewidth=3, color=default_colours[2]+"99", label="crumpled chain")

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize="34", frameon=False, borderpad=0.5, ncols=1)
plt.tight_layout()
fig.savefig(path.join(analyses_dir, "Tau(s).svg"), format="svg")

# Display important results from fitting procedure
print(f"Scaling exponent from experiment: {slope_exp:.3f}")
print(f"Scaling exponent from simulation: {slope_sim:.3f}")