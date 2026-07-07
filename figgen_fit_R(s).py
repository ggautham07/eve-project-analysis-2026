from os import path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Cost function for the fit
def cost(d, test_data, ref_data):
    return np.sum((np.abs(ref_data - (test_data + d))))

# From Brückner et al.
genomic_dists = np.array([58, 82, 88, 149, 190, 595, 3327])
genomic_dists_log = np.log10(genomic_dists)

# Load processed experimental data
data = np.load("./resources/processed_data/exp_R(s)_open.npy", allow_pickle=True)
exp_data_log = np.log10(data)

# Pass unique simulation tag (like "00_Rouse") to identify the simulated data
args = sys.argv
tag = args[1]
dir_results = path.join(f"./analyses/{tag}")

# Load already computed R(s) distribution from corresponding analysis file
try:
    filename = "analysis.h5"
    with h5py.File(path.join(dir_results, filename)) as handler:
        sim_data_raw = handler["R(s)"][:]
    sim_data = np.mean(sim_data_raw, axis=0).T  # mean R(s) computation
    sim_data_log = np.log10(sim_data)
except FileNotFoundError:
    print("Could not find computed distances. Compute using compute_R(s).py")
    raise

# Perform fitting with experimental data
optim_results = []
sigma = []
for _ in range(10):     # Optimisation performed 10 times to find the best one
    init_guess = np.random.randint(low=-1, high=1, size=1)
    result = minimize(fun=cost, x0=init_guess, args=(exp_data_log, sim_data_log,), method="L-BFGS-B")
    optim_results.append(result)
    sigma.append(result["fun"])
result = optim_results[np.argmin(sigma)]
d_fit = np.squeeze(result["x"])     # Scaling factor
sigma = result["fun"]   # Deviation from experiment

slope_sim, intercept_sim = np.polyfit(genomic_dists_log[:5], sim_data_log[:5], deg=1)   # Simulation scaling exponent
slope_exp, intercept_exp = np.polyfit(genomic_dists_log[:5], exp_data_log[:5], deg=1)   # Experiment scaling exponent

# Parameters for plotting
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 40}

# Plot a manuscript-ready figure and save it
fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.scatter(genomic_dists, 10 ** (exp_data_log + d_fit), s=480, marker="o", color=default_colours[7], edgecolor="#000", label=f"Experiment")
ax.scatter(genomic_dists, 10 ** sim_data_log, s=300, marker="X", color=default_colours[1], edgecolor="#000", label=f"Simulation")
ax.plot(genomic_dists, genomic_dists ** slope_sim * (10 ** intercept_sim), ":", linewidth=3.5, color=default_colours[1])
ax.plot(genomic_dists, genomic_dists ** slope_exp * (10 ** (intercept_exp + d_fit)), ":", linewidth=3.5, color=default_colours[7])
for_slope = np.arange(genomic_dists[0], genomic_dists[4] + 100)
ax.plot(for_slope, np.sqrt(for_slope) * 3, "--", linewidth=2.5, color="#000")
ax.text(x=for_slope[len(for_slope) // 5], y=np.sqrt(for_slope)[len(for_slope) // 5] * 4, s="$s^{0.5}$", fontsize="30")
ax.set_xlabel("$s$   (kb)", fontdict=fd, labelpad=24)
ax.set_ylabel("$\\langle R \\rangle$   ($\mathregular{s.u.}$)", fontdict=fd, labelpad=24)
ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize="34", frameon=False, borderpad=0.5)
plt.tight_layout()
fig.savefig(path.join(dir_results, "R(s)_fit.svg"), format="svg")
plt.close(fig)

# Display important results from fitting procedure
print(f"Scaling exponent from simulation: {slope_sim:.3f}")
print(f"Scaling exponent from experiment: {slope_exp:.3f}")
print(f"Translation in log-log along R axis: {d_fit}")
print(f"Deviation from experiment: {sigma}")