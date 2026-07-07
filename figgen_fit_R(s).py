import numpy as np
import matplotlib.pyplot as plt
from os import path
from polychrom.hdf5_format import load_hdf5_file
from scipy.optimize import minimize
import sys
import json
from glob import glob
import h5py

# Cost function for the fit
def cost(d, test_data, ref_data):
    return np.sum((np.abs(ref_data - (test_data + d))))
    # return np.mean(np.sqrt(np.abs(ref_data - (test_data + d))))

linear_interlocus_dists = np.array([58, 82, 88, 149, 190, 595, 3327])
linear_interlocus_dists_log = np.log10(linear_interlocus_dists)

# Our processed experimental data
# data = np.load("./resources/processed_data/exp_R(s).npy", allow_pickle=True)
data = np.load("./resources/processed_data/exp_R(s)_open.npy", allow_pickle=True)
# data_temp = np.copy(data)
# data[1] = data_temp[2]
# data[2] = data_temp[1]
exp_data_log = np.log10(data)

args = sys.argv
tag = args[1]
sim_dir = path.join(f"./analyses/{tag}")
# print("Found the following folders", *sim_dirs, sep="\n")
flag_mult = True if len(sim_dir) > 1 else False

dir_results = sim_dir
try:
    filename = "analysis.h5"
    with h5py.File(path.join(sim_dir, filename)) as handler:
        # print(handler.keys())
        sim_data_raw = handler["R(s)"][:]
    # sim_data_raw = load_hdf5_file(path.join(dir_results, filename))["R(s)"]
    # print("Found computed distances")
    sim_data = np.mean(sim_data_raw, axis=0).T
    sim_data_log = np.log10(sim_data)
except FileNotFoundError:
    print("Could not find computed distances. Compute them from the simulation trajectory using workflow_compute_R_scaling.py")
    raise

# params = json.load(open(path.join(f"./trajectories/{tag}", "parameters.json"), mode="r"))
# N = params["polymer_length"]
# bc = False
# le = False
# try:
#     eps = params["interaction_matrix"][1][1]
# except IndexError:
#     eps = params["interaction_matrix"][0][0]
#     bc = True
# if path.exists(path.join(f"./trajectories/{tag}", "parameters_loop_extrusion.json")):
#     params = json.load(open(path.join(f"./trajectories/{tag}", "parameters_loop_extrusion.json")), mode="r")
#     unload_prob = params["LEF_unload_prob"]
#     load_prob = params["LEF_load_prob"]
#     step_prob = params["LEF_step_prob"]
#     le = True

optim_results = []
sigma = []
for _ in range(10):
    init_guess = np.random.randint(low=-1, high=1, size=1)
    result = minimize(fun=cost, x0=init_guess, args=(exp_data_log, sim_data_log,), method="L-BFGS-B")
    optim_results.append(result)
    sigma.append(result["fun"])
result = optim_results[np.argmin(sigma)]
d_fit = np.squeeze(result["x"])
sigma = result["fun"]

slope1, intercept1 = np.polyfit(linear_interlocus_dists_log[:5], sim_data_log[:5], deg=1)
slope2, intercept2 = np.polyfit(linear_interlocus_dists_log[:5], exp_data_log[:5], deg=1)
# slope2, intercept2 = np.polyfit(linear_interlocus_dists_log[:], sim_data_log[:], deg=1)

default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]
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
ax.scatter(linear_interlocus_dists, 10 ** (exp_data_log + d_fit), s=480, marker="o", color=default_colours[7], edgecolor="#000", label=f"Experiment")
ax.scatter(linear_interlocus_dists, 10 ** sim_data_log, s=300, marker="X", color=default_colours[1], edgecolor="#000", label=f"Simulation")
ax.plot(linear_interlocus_dists, linear_interlocus_dists ** slope1 * (10 ** intercept1), ":", linewidth=3.5, color=default_colours[1])
ax.plot(linear_interlocus_dists, linear_interlocus_dists ** slope2 * (10 ** (intercept2 + d_fit)), ":", linewidth=3.5, color=default_colours[7])
for_slope = np.arange(linear_interlocus_dists[0], linear_interlocus_dists[4] + 100)
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
# ax.set_yticks([10, 20, 40, 80])
# ax.get_yaxis().get_major_formatter().labelOnlyBase = False
# formatter = ScalarFormatter()
# formatter.set_scientific(False)
# ax.yaxis.set_major_formatter(formatter)
# ax.yaxis.set_minor_formatter(formatter)
# ax.grid(lw=0.3)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], fontsize="34", frameon=False, borderpad=0.5)
plt.tight_layout()
# fig.savefig("./results/open/block_copolymer_with_loopex_best-all_R(s)_fit.svg", format="svg")
# fig.savefig("./results/open/block_copolymer_best-both_R(s)_fit.svg", format="svg")
# fig.savefig("./results/open_off_state/block_copolymer_with_loopex_best-common_R(s)_fit.svg", format="svg")
# fig.savefig("./results/open/loopex_best-str_R(s)_fit.svg", format="svg")
# fig.savefig("./results/open/Rouse_R(s)_fit.svg", format="svg")
# fig.savefig("./results/open/Rouse_excvol_R(s)_fit.svg", format="svg")
fig.savefig(path.join(dir_results, "R(s)_fit.svg"), format="svg")
plt.close(fig)
print(f"Scaling exponent from simulation: {slope1:.3f}")
print(f"Scaling exponent from experiment: {slope2:.3f}")
# # print(f"Translation in log-log along R axis: {d_fit}")
# # print(f"Deviation from experiment: {sigma}")

print(d_fit)
print(sigma)
sys.exit(0)