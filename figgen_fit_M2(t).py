import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib import container, lines, patches
from polychrom.hdf5_format import load_hdf5_file
from os import path
from scipy.optimize import minimize
import sys
import json
from copy import copy
from glob import glob

# Compute a polynomial of degree of the size of the coefficients array passed
def polymd(X, coeff):
    assert coeff.ndim == X.ndim
    # coefficients in order from 0 ... n
    assert coeff.ndim in [1, 2]
    powers = np.arange(coeff.shape[0])
    if coeff.ndim == 1:
        return np.sum(coeff[..., np.newaxis] * (X ** powers[..., np.newaxis]), axis=0)
    vals = np.zeros_like(X)
    for i in range(coeff.shape[1]):
        vals[:, i] = np.sum(coeff[:, i, np.newaxis] * (X[:, i] ** powers[..., np.newaxis]), axis=0)
    return vals

# Cost function for the fit
def cost(d, f, Tau, Phi, f_poly_coeff):
    dTau, dPhi = d
    # return np.mean(np.abs(10 ** (Phi + dPhi) - (10 ** f(Tau + dTau, f_poly_coeff))))
    return np.sum(np.abs((Phi + dPhi) - f(Tau + dTau, f_poly_coeff)))
    # return np.mean(10 ** np.abs((Phi + dPhi) - f(Tau + dTau, f_poly_coeff)))
    # return np.mean(np.sqrt(np.abs((Phi + dPhi) - f(Tau + dTau, f_poly_coeff))))     # new error

def ilog(x, base):
    return base ** x

# Colours to be used for the plot
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]

linear_interlocus_dists = np.array([58, 82, 88, 149, 190, 595, 3327])

# # Timothy's processed experimental data
# dt_range_exp_log = np.log10(np.load("./resources/processed_data_from_tim/time.npy")[1:])
# exp_data_log = np.log10(np.load("./resources/processed_data_from_tim/MSD.npy")[:-1, 1:]).T

# Our processed experimental data
# data = load_hdf5_file("./resources/processed_data/exp_MSDs.h5")
# dt_range_exp_log = np.log10(data["time_lags"])
# exp_data_log = np.log10(data["MSDs"]).T
data = load_hdf5_file("./resources/processed_data/exp_M2(t)_open.h5")
dt_range_exp_log = np.log10(data["time_lags"])
exp_data_log = np.log10(data["two_locus_MSD"]).T

# time_lags = data["time_lag"]
# two_locus_MSD = data["two_locus_MSD"]
# np.logspace(start=np.min(time_lags, stop=np.max(time_lags)))

args = sys.argv
tag = args[1]
sim_dir = path.join(f"./analyses/{tag}")
# print("Found the following folders", *sim_dirs, sep="\n")
flag_mult = True if len(sim_dir) > 1 else False

dir_results = sim_dir
# eps = float(sim_dir[sim_dir.find("=")+1:])

try:
    filename = "analysis.h5"
    with h5py.File(path.join(sim_dir, filename)) as handler:
        print(handler.keys())
        # print(handler.keys())
        dt_range_sim = handler["time"][:]
        sim_data = handler["M2(t)"][:]
    # raw_data_sim = load_hdf5_file(path.join(dir_results, filename))
    # print("Found computed MSDs")
except FileNotFoundError:
    # print("Could not find computed MSDs. Compute them from the simulation trajectory using workflow_compute_MSDs.py")
    raise

params = json.load(open(path.join(f"./trajectories/{tag}", "parameters.json"), mode="r"))
N = params["polymer_length"]
bc = False
le = False
if "interaction_matrix" in list(params.keys()):
    try:
        eps = params["interaction_matrix"][1][1]
    except IndexError:
        eps = params["interaction_matrix"][0][0]
    bc = True
if path.exists(path.join(sim_dir[0], "parameters_loop_extrusion.json")):
    params = json.load(open(path.join(sim_dir[0], "parameters_loop_extrusion.json"), mode="r"))
    unload_prob = params["LEF_unload_prob"]
    load_prob = params["LEF_load_prob"]
    le = True   

dt_range_sim_log = np.log10(dt_range_sim[:-3])
sim_data_log = np.log10(sim_data[:-3, :])

coeff = np.polyfit(dt_range_sim_log, sim_data_log, deg=5)
coeff = coeff[::-1, ...]

dt_range_sim_log_concat = np.array([dt_range_sim_log for _ in range(coeff.shape[1])]).T
sim_data_log_fit = polymd(dt_range_sim_log_concat, coeff)

# plt.rcParams['font.family'] = ["serif", "sans-serif"]
# plt.rcParams['mathtext.fontset'] = "dejavuserif"
# fd = {"fontfamily": "serif", "fontsize": 12}

# plt.figure(figsize=(10, 8))
# title = "Two-locus MSD curve polynomial fits"
# if (not bc) and le:
#     title = title + f" | Loop extrusion, $N = {N}$, $p_U = {unload_prob}$, $p_L = {load_prob}$"
# elif bc and (not le):
#     title = title +  f" | Block copolymer, $N={4208}, \\epsilon={eps:.3f}$"
# elif bc and le:
#     title = title + f" | Block copolymer + loop extrusion, $N = {N}$\n\\epsilon={eps:.3f}, $p_U = {unload_prob}$, $p_L = {load_prob}$"
# else:
#     title = title + f" | Rouse model, $N={4208}$"
# plt.title(title, fontsize=16)
# for i in range(sim_data_log_fit.shape[1]):
#     plt.plot(dt_range_sim_log, sim_data_log[:, i], "o", markersize=8, label=f"Simulated data, {linear_interlocus_dists[i]} kb", color=gregor_colours[i])
#     plt.plot(dt_range_sim_log, sim_data_log_fit[:, i], "--", linewidth=2, label=f"Polynomial fit, {linear_interlocus_dists[i]} kb", color=gregor_colours[i])
# plt.xlabel("$t$", fontdict=fd)
# plt.ylabel("$f$", fontdict=fd)
# plt.legend()
# plt.savefig(path.join(sim_dir, "M2(t)_fit_polynomial.png"))
# plt.close()

dt_range_exp_log_concat = np.array([dt_range_exp_log for _ in range(coeff.shape[1])]).T
optim_results = []
sigma = []
for _ in range(10):
    init_guess = np.random.randint(low=-2, high=2, size=2)
    result = minimize(fun=cost,
                    x0=np.asarray(init_guess),
                    args=(polymd, dt_range_exp_log_concat, exp_data_log, coeff),
                    method="L-BFGS-B")
    optim_results.append(result)
    sigma.append(result["fun"])

result = optim_results[np.argmin(sigma)]
dTau_fit, dPhi_fit = result["x"]
sigma = result["fun"]

plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
# colour_base = "#121212"
# plt.rcParams['text.color'] = colour_base
# plt.rcParams['axes.labelcolor'] = colour_base
# plt.rcParams['xtick.color'] = colour_base
# plt.rcParams['ytick.color'] = colour_base
fd = {"fontfamily": "serif", "fontsize": 40}

fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$t$   ($\mathregular{t.u.}$)", fontdict=fd, labelpad=24)
ax.set_ylabel("$MSD_2$   ($\mathregular{s.u.^2}$)", fontdict=fd, labelpad=24)
for i in range(exp_data_log.shape[1]):
    ax.plot(10 ** (dt_range_exp_log + dTau_fit), 10 ** (exp_data_log[:, i] + dPhi_fit), "o-", markersize=15, linewidth=0.6, color=gregor_colours[i]+"99", markeredgecolor="#ff000080", markeredgewidth=0.85)
    # ax.plot(10 ** dt_range_sim_log, 10 ** sim_data_log_fit[:, i], color=gregor_colours[i] + "98", linewidth=3)
    ax.plot(10 ** dt_range_sim_log, 10 ** sim_data_log[:, i], color=gregor_colours[i]+"99", linewidth=4)
ax.plot(10 ** dt_range_sim_log[:12], (np.sqrt(10 ** (dt_range_sim_log[:12] + 1.5))), "--", linewidth=2.5, color="#000")
ax.text(x=10 ** dt_range_sim_log[1], y=(np.sqrt(10 ** (dt_range_sim_log[1] + 2))), s="$t^{0.5}$", fontsize="30")
# ax.text(x=10 ** dt_range_sim_log[5], y= 10 ** (np.min(sim_data_log) + 0.05), s=f"$\\Delta_\\tau = {dTau_fit:.3f}$, $\\Delta_\\phi = {dPhi_fit:.3f}$, $\\sigma = {sigma:.3f}$", fontdict={"fontsize": 18, "fontfamily": "serif", "fontstyle": "italic"})
# ax.grid(lw=0.3)
ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
handles0, _ = ax.get_legend_handles_labels()
handles0.append(lines.Line2D([], [], color="#0d0d0d", linewidth=3, label="Simulation"))
handles0.append(lines.Line2D([], [], marker="o", color="#0d0d0d", linewidth=0.6, markerfacecolor="#000", markeredgecolor="#ff000080", markeredgewidth=0.85, markersize=15, label="Experiment"))
slope = handles0.pop(0)
handles0.append(slope)
handles1 = [patches.Patch(color=gregor_colours[i]+"98", label=f"{linear_interlocus_dists[i]} kb") for i in range(len(linear_interlocus_dists))]
leg1 = ax.legend(handles=handles0, loc="lower right", fontsize="32", frameon=False, borderpad=1)
ax.add_artist(leg1)
ax.legend(handles=handles1, loc="upper left", fontsize="22", ncol=2, frameon=False, borderpad=1)
plt.tight_layout()
# fig.savefig("./results/open/block_copolymer_with_loopex_best-all_M2(t)_fit.svg", format="svg")
# fig.savefig("./results/open/block_copolymer_best-both_M2(t)_fit.svg", format="svg")
# fig.savefig("./results/open_off_state/block_copolymer_with_loopex_best-common_M2(t)_fit.svg", format="svg")
# fig.savefig("./results/open/loopex_best-dyn_M2(t)_fit.svg", format="svg")
# fig.savefig("./results/open/Rouse_M2(t)_fit.svg", format="svg")
# fig.savefig("./results/open/Rouse_excvol_M2(t)_fit.svg", format="svg")
fig.savefig(path.join(dir_results, "M2(t)_fit.svg"), format="svg")
plt.close(fig)

# print(f"Translation in log-log along t axis: {dTau_fit}")
# print(f"Translation in log-log along MSD axis: {dPhi_fit}")
# print(f"Deviation from experiment: {sigma}")

print(sigma)
print(str(dTau_fit) + "," + str(dPhi_fit))
sys.exit(0)