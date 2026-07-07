import numpy as np
from polychrom.hdf5_format import list_URIs, load_URI, load_hdf5_file
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import container, lines, patches
import sys
from os import path
from time import perf_counter
import h5py
from itertools import chain


def generate_bins(N, start=3, bins_per_order_magn=30):
    lstart = np.log10(start)
    lend = np.log10(N - 1) + 1e-6
    num = int(np.ceil((lend - lstart) * bins_per_order_magn))
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=max(num, 0)))
    if len(bins) > 0:
        assert bins[-1] == N - 1
    return bins

# Getting the monomer compartment blocks
monomer_classes_file_path = "./resources/processed_data/evec_region.csv"
raw_evec_data = pd.read_csv(monomer_classes_file_path, sep="\t")
compartments = (raw_evec_data["E1"].to_numpy() >= 0).astype(int)
monomer_classes = np.asarray(list(chain.from_iterable([[compartments[i] for _ in range(16)] \
                                                       for i in range(len(compartments))])))

switch_pos = np.nonzero(monomer_classes[1:] - monomer_classes[:-1])[0]
switch_pos = np.concatenate([[0], switch_pos])
lengths = switch_pos[1:] - switch_pos[:-1]

args = sys.argv
sim_dir = args[1]
try:
    start = int(args[2])
except IndexError:
    print("Start block not specified. Defaulting to start = 0")
    start = 0

# URIs = list_URIs(sim_dir)[start:]

# trajectory = np.empty((len(URIs), 4208, 3))
# for i, uri in enumerate(URIs):
#     trajectory[i] = load_URI(uri)["pos"]

try:
    st = perf_counter()
    # Extracting trajectories
    URIs = list_URIs(sim_dir)[start:]     # for eve locus simulations
    sample = load_URI(URIs[0])["pos"]
    N = sample.shape[0]
    Z = len(URIs)
    # For one axis, ML's method
    trajectory = np.empty((Z, N, 3))
    for z in range(Z):
        data = load_URI(URIs[z])["pos"]
        trajectory[z,:] = data[:]
    print(f"Successfully loaded trajectory data in {perf_counter() - st:.2f} seconds")
except Exception as err:
    raise err

st = perf_counter()

time_total = trajectory.shape[0]
dt_max = time_total // 50
dt_range = generate_bins(dt_max, start=2, bins_per_order_magn=30)
num_frames = time_total - dt_max

autocorr_all = []
autocorr_lim_all = []
tau = []
for m in range(len(lengths)):
    # Get autocorrelation curve
    comp = 0 if m % 2 == 0 else 1
    length = lengths[m]
    if length >= 250:
        idxs = (switch_pos[m], switch_pos[m+1])
        Rvecs = trajectory[:, idxs[1], :] - trajectory[:, idxs[0], :]
        R2_mean = np.square(np.mean(np.linalg.norm(Rvecs[::100], axis=1)))
        autocorr = np.empty_like(dt_range, dtype=float)
        for i, dt in enumerate(dt_range):
            autocorr_dt = np.empty(num_frames)
            for n in range(num_frames):
                autocorr_dt[n] = np.dot(Rvecs[n], Rvecs[n+dt])
            # print(np.mean(autocorr_dt))
            autocorr[i] = np.mean(autocorr_dt)
        autocorr /= R2_mean
        autocorr_all.append(autocorr)
        # Get autocorrelation decay limits
        vecdots_self = np.empty(num_frames)
        for n in range(num_frames):
            vecdots_self[n] = np.dot(Rvecs[n], Rvecs[n])
        autocorr_lim = (np.mean(vecdots_self) / R2_mean) / np.e
        autocorr_lim_all.append(autocorr_lim)
        # Get relaxation time
        tau.append(dt_range[np.argmax(autocorr < autocorr_lim)])
        print(f"Completed computation for loci pairs {lengths[m]} kb")

dir_saving = f"./results/latest/sims/{path.basename(sim_dir)}/"
with h5py.File(path.join(dir_saving, "autocorrelation.h5"), mode="a") as handler:
    if "time_autocorr" in handler.keys():
        del handler["time_autocorr"]
    handler.create_dataset(name="time_autocorr", data=np.asarray(dt_range)[:])
    if "C_R(t)" in handler.keys():
        del handler["C_R(t)"]
    handler.create_dataset(name="C_R(t)", data=np.asarray(autocorr_all)[:])
    if "C_R(t)_lim" in handler.keys():
        del handler["C_R(t)_lim"]
    handler.create_dataset(name="C_R(t)_lim", data=np.asarray(autocorr_lim_all)[:])
    if "Tau(s)" in handler.keys():
        del handler["Tau(s)"]
    handler.create_dataset(name="Tau(s)", data=np.asarray(tau)[:])

print(autocorr_all)
print(autocorr_lim_all)
print(tau)

# with h5py.File(path.join(sim_dir, "analysis.h5"), mode="a") as handler:
#     if "time_autocorr" in handler.keys():
#         del handler["time_autocorr"]
#     handler.create_dataset(name="time_autocorr", data=np.asarray(dt_range)[:])
#     if "C_R(t)" in handler.keys():
#         del handler["C_R(t)"]
#     handler.create_dataset(name="C_R(t)", data=np.asarray(autocorr_all)[:])
#     if "C_R(t)_lim" in handler.keys():
#         del handler["C_R(t)_lim"]
#     handler.create_dataset(name="C_R(t)_lim", data=np.asarray(autocorr_lim_all)[:])
#     if "Tau(s)" in handler.keys():
#         del handler["Tau(s)"]
#     handler.create_dataset(name="Tau(s)", data=np.asarray(tau)[:])

# print(f"Completed task in {perf_counter() - st:.2f} seconds. Plotting...")


# plt.rcParams['font.family'] = ["serif", "sans-serif"]
# plt.rcParams['mathtext.fontset'] = "dejavuserif"
# fd = {"fontfamily": "serif", "fontsize": 48}
# default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
# gregor_colours = ["#7ab200", "#009ba5", "#008f4c", "#0085b6", "#004e9e", "#00177d", "#00053d"]
# # eps = [0.20, 0.23, 0.26, 0.29, 0.32]
# # eps_minor = [0.18, 0.19, 0.21, 0.22, 0.24, 0.25, 0.27, 0.28, 0.30, 0.31]
# fig, ax = plt.subplots()
# fig.set_figwidth(14)
# fig.set_figheight(14)
# ax.set_xlabel("$\\Delta t$    ($\mathregular{t.u.}$)", fontdict=fd, labelpad=30)  #     ($\mathregular{k_B T}$)
# ax.set_ylabel("$\\langle C_R \\rangle$", fontdict=fd, labelpad=20)
# for i in range(len(autocorr_all)):
#     ax.plot(dt_range, autocorr_all[i], "-o", linewidth=0.8, markersize=12, color=gregor_colours[i]+"99")
#     ax.axhline(y=autocorr_lim_all[i], ls=":", lw=2, color=gregor_colours[i]+"99")
#     ax.axvline(x=tau[i], ls="--", lw=2, color=gregor_colours[i]+"99")
# ax.set_xscale("log")
# ax.set_yscale("log")
# # ax.set_xticks(eps, labels=[f"{e:.2f}" for e in eps])
# # ax.set_xticks(eps_minor, labels=[], minor=True)
# ax.tick_params(axis="x", labelsize=40, which="major", pad=18, direction="in", length=10, width=1.2)
# ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=5, width=1)
# ax.tick_params(axis="y", labelsize=40, which="major", pad=18, direction="in", length=8, width=1.1)
# ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=5, width=1)
# ax.tick_params(which="both", bottom=True, top=True)
# ax.tick_params(labelbottom=True, labeltop=False)
# handles = [patches.Patch(color=gregor_colours[i]+"98", label=f"{lengths[i]} kb") for i in range(len(lengths))]
# ax.legend(handles=handles, loc="center left", fontsize="28", ncols=1, frameon=False, borderpad=1)
# fig.tight_layout()
# plt.savefig(path.join(sim_dir, "C_R(t).svg"), format="svg")
# plt.close()

# tau_exp = [39.53656951044747103,
#            72.07541654553817523,
#            68.56142461161556412,
#            100.3457588639866600,
#            128.6890127669227297,
#            190.2885358950472892,
#            742.8575441880935841]
# slope1, intercept1 = np.polyfit(np.log10(lengths), np.log10(tau_exp), deg=1)
# slope2, intercept2 = np.polyfit(np.log10(lengths), np.log10(tau), deg=1)

# fig, ax = plt.subplots()
# fig.set_figwidth(14)
# fig.set_figheight(14)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("$s$    (kb)", fontdict=fd, labelpad=24)
# ax.set_ylabel("$\\tau$    ($\mathregular{t.u.}$)", fontdict=fd, labelpad=24)
# ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
# ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
# ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
# ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
# ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
# ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
# ax.scatter(lengths, tau, s=400, marker="o", color=default_colours[0], label="Simulation")
# pt_l, pt_r = ax.get_xlim()
# pt_range = np.linspace(pt_l, pt_r, 20)
# ax.plot(pt_range, pt_range ** slope2 * (10 ** (intercept2)), "--", linewidth=4, color=default_colours[0])
# # ax.plot(lengths[:-1], (lengths[:-1] ** 2) * 2,
# #          ":", color=default_colours[1], label="ideal chain")
# ax.plot(pt_range, (pt_range ** (0.7)) * 1.15,
#         "--", linewidth=4, color="black", label="Experiment")
# ax.plot(pt_range, (pt_range ** 2),
#          ":", linewidth=3, color=default_colours[1], label="ideal chain")
# ax.plot(pt_range, (pt_range ** (5 / 3)) * 1.15,
#          ":", linewidth=3, color=default_colours[2], label="crumpled chain")

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], loc="upper right", fontsize="34", frameon=False, borderpad=0.5, ncols=1)
# plt.tight_layout()
# fig.savefig(path.join(sim_dir, "Tau(s).svg"), format="svg")