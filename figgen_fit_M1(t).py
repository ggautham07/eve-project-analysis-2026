from os import path
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import container, lines, patches
from polychrom.hdf5_format import load_hdf5_file
from scipy.optimize import minimize

# Compute a polynomial of degree of the size of the coefficients array passed, setting bounds
def polymd(X, coeff, bounds=None):
    assert coeff.ndim == X.ndim
    # coefficients in order from 0 ... n
    assert coeff.ndim in [1, 2]
    powers = np.arange(coeff.shape[0])
    if coeff.ndim == 1:
        res = np.sum(coeff[..., np.newaxis] * (X ** powers[..., np.newaxis]), axis=0)
        if bounds is None:
            return res
        else:
            return res if (res <= bounds[1] and res >= bounds[0]) else -100
    vals = np.zeros_like(X)
    for i in range(coeff.shape[1]):
        vals[:, i] = np.sum(coeff[:, i, np.newaxis] * (X[:, i] ** powers[..., np.newaxis]), axis=0)
    if bounds is None:
        return vals
    vals[~np.logical_and(vals <= bounds[1], vals >= bounds[0])] = -100
    return vals

# Cost function for the fit
def cost(d, f, Tau, Phi, f_poly_coeff, f_bounds=None):
    dTau, dPhi = d
    return np.sum(np.abs((Phi + dPhi) - f(Tau + dTau, f_poly_coeff, f_bounds)))

# Reverse log given a base
def ilog(x, base):
    return base ** x

# Colours to be used for the plot
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]

genomic_dists = np.array([58, 82, 88, 149, 190, 595, 3327])

# Load experimental data
data = load_hdf5_file("./resources/processed_data/exp_M1(t)_open.h5")
idxs_logspaced = [0, 1, 2, 3, 6, 8, 11, 15, 20, 27, 36, 48]
dt_range_exp_log = np.log10(data["time_lag"][idxs_logspaced])
exp_data_blue = np.mean(data["one_locus_MSD_blue"][:, idxs_logspaced], axis=0)[np.newaxis, ...]
exp_data_green = data["one_locus_MSD_green"][:, idxs_logspaced]
exp_data_log = np.log10(np.concatenate((exp_data_blue, exp_data_green))).T

# Process arguments to get unique simulation tag
args = sys.argv
tag = args[1]
dir_results = path.join(f"./analyses/{tag}")

# Process arguments to get unique simulation tag
try:
    filename = "analysis.h5"
    with h5py.File(path.join(dir_results, filename)) as handler:
        dt_range_sim = handler["time"][:]
        sim_data = handler["M1(t)"][:]
except FileNotFoundError:
    print("Could not find computed MSDs. Compute using compute_Mx(t).py")
    raise

sim_data = np.concatenate((sim_data[..., 1], np.mean(sim_data[..., 0], axis=-1)[..., np.newaxis]), axis=-1)
dt_range_sim_log = np.log10(dt_range_sim[:])
sim_data_log = np.log10(sim_data[:, :])

# Fit MSD curves to first degree polynomial and extract coefficient
coeff = np.polyfit(dt_range_sim_log, sim_data_log, deg=1)
coeff = coeff[::-1, ...]

# Construct interpolated simulated MSD curves using the coefficients
dt_range_sim_log_concat = np.array([dt_range_sim_log for _ in range(coeff.shape[1])]).T
sim_data_log_fit = polymd(dt_range_sim_log_concat, coeff)

# Perform fitting with experimental data
dt_range_exp_log_concat = np.array([dt_range_exp_log for _ in range(coeff.shape[1])]).T
optim_results = []
sigma = []
for _ in range(10):
    init_guess = np.random.randint(low=-2, high=2, size=2)
    result = minimize(fun=cost,
                    x0=np.asarray(init_guess),
                    args=(polymd,
                          dt_range_exp_log_concat,
                          exp_data_log,
                          coeff,
                          (np.min(sim_data_log), np.max(sim_data_log))),
                    method="L-BFGS-B",)
    optim_results.append(result)
    sigma.append(result["fun"])
result = optim_results[np.argmin(sigma)]
dTau_fit, dPhi_fit = result["x"]    # Scaling factors along both axes
sigma = result["fun"]   # Deviation from experiment

# Parameters for plotting
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 40}
colours = [default_colours[0]] + gregor_colours

# Plot a manuscript-ready figure and save it
fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$t$   ($\mathregular{t.u.}$)", fontdict=fd, labelpad=24)
ax.set_ylabel("$MSD_1$   ($\mathregular{s.u.^2}$)", fontdict=fd, labelpad=24)
ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
for i in range(exp_data_log.shape[1]):
    ax.plot(10 ** (dt_range_exp_log + dTau_fit), 10 ** (exp_data_log[:, i] + dPhi_fit),
            "o-", markersize=15, linewidth=0.6,
            color=colours[i]+"99", markeredgecolor="#ff000080", markeredgewidth=0.85)
    ax.plot(10 ** dt_range_sim_log[:45], 10 ** sim_data_log[:45, i],
            linewidth=4, color=colours[i]+"99")
ax.plot(10 ** dt_range_sim_log[:20], (np.sqrt(10 ** (dt_range_sim_log[:20] + 0.5))),
        "--", linewidth=2.5, color="#000")
ax.text(x=10 ** dt_range_sim_log[1], y=(np.sqrt(10 ** (dt_range_sim_log[1] + 0.8))), s="$t^{0.5}$", fontsize="30")
handles0, _ = ax.get_legend_handles_labels()
handles0.append(lines.Line2D([],
                             [],
                             color="#0d0d0d",
                             linewidth=3,
                             label="Simulation"))
handles0.append(lines.Line2D([],
                             [],
                             marker="o",
                             color="#0d0d0d",
                             linewidth=0.6,
                             markerfacecolor="#000",
                             markeredgecolor="#ff000080",
                             markeredgewidth=0.85,
                             markersize=15,
                             label="Experiment"))
slope = handles0.pop(0)
handles0.append(slope)
handles1 = [patches.Patch(color=gregor_colours[i]+"98", label=f"{genomic_dists[i]} kb") \
            for i in range(len(genomic_dists))]
leg1 = ax.legend(handles=handles0, loc="lower right", fontsize="32", frameon=False, borderpad=1)
ax.add_artist(leg1)
ax.legend(handles=handles1, loc="upper left", fontsize="22", ncol=2, frameon=False, borderpad=1)
plt.tight_layout()
fig.savefig(path.join(dir_results, "M1(t)_fit.svg"), format="svg")
plt.close(fig)

# Display important results from fitting procedure
print(f"Translation in log-log along t axis: {dTau_fit}")
print(f"Translation in log-log along MSD axis: {dPhi_fit}")
print(f"Deviation from experiment: {sigma}")