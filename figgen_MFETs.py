import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from polychrom.hdf5_format import load_hdf5_file, save_hdf5_file
import pandas as pd
from itertools import chain
from os import path
from glob import glob
import json
from time import perf_counter

# Generate evenly-spaced bins on the log scale
def generate_bins(N, start=2, bins_per_order_magn=10):
    lstart = np.log10(start)
    lend = np.log10(N - 1) + 1e-6
    num = int(np.ceil((lend - lstart) * bins_per_order_magn))
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=max(num, 0)))
    if len(bins) > 0:
        assert bins[-1] == N - 1
    return bins

analyses_dirs = sorted(glob("./analyses/02_block_copolymer_eps=0.2[6-9]*") + glob("./analyses/02_block_copolymer_eps=0.3*"))
epsilons = [float(sd.split("=")[-1]) for sd in analyses_dirs]
analyses_dirs.insert(0, "./analyses/00_Rouse_excvol")
# analyses_dirs.insert(0, "./analyses/00_Rouse")
epsilons.insert(0, 0.25)
# epsilons.insert(0, 0.24)

# data = load_hdf5_file("./results/latest/sims/MFETs.h5")
# result_00 = data["C0-C0"][:]
# result_01 = data["C0-C1"][:]
# result_11 = data["C1-C1"][:]

# idxs_nonan_00 = np.all(~np.isnan(result_00), axis=1)
# idxs_nonan_01 = np.all(~np.isnan(result_01), axis=1)
# idxs_nonan_11 = np.all(~np.isnan(result_11), axis=1)

# result_00 = result_00[idxs_nonan_00]
# result_01 = result_01[idxs_nonan_01]
# result_11 = result_11[idxs_nonan_11]

# s_all_00 = s_all[idxs_nonan_00]
# s_all_01 = s_all[idxs_nonan_01]
# s_all_11 = s_all[idxs_nonan_11]

# print(result_00, result_00.shape)
# print(result_01, result_01.shape)
# print(result_11, result_11.shape)

# print(np.mean(np.isnan(result_00)))
# print(np.mean(np.isnan(result_01)))
# print(np.mean(np.isnan(result_11)))

result_00 = []
result_11 = []
result_01 = []
for ad in analyses_dirs:
    with h5py.File(path.join(ad, "analysis.h5")) as handler:
        s_all = handler["MFET/s"][:]
        result_00.append(handler["MFET/C0-C0"][:])
        result_11.append(handler["MFET/C1-C1"][:])
        result_01.append(handler["MFET/C0-C1"][:])
result_00 = np.asarray(result_00).T
result_11 = np.asarray(result_11).T
result_01 = np.asarray(result_01).T
result_00_diff = result_00 / result_00[:,0][..., np.newaxis]
result_11_diff = result_11 / result_11[:,0][..., np.newaxis]
result_01_diff = result_01 / result_01[:,0][..., np.newaxis]
print("Taking the following genomic distances: ", *s_all)
print(result_00.shape)
print(result_01.shape)
print(result_11.shape)

result_max = np.max([np.max(result_00), np.max(result_01), np.max(result_11)])
result_min = np.min([np.min(result_00), np.min(result_01), np.min(result_11)])

# save_hdf5_file(filename="./results/latest/sims/MFETs.h5",
#                data_dict={"s": s_all, "C0-C0": result_00, "C1-C1": result_11, "C0-C1": result_01})

# Plotting
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 60}
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#009ba5", "#008f4c", "#0085b6", "#004e9e", "#00177d", "#00053d"]
# eps = [0.23, 0.26, 0.275, 0.29, 0.32]
# eps_minor = [0.24, 0.25, 0.265, 0.27, 0.275, 0.28, 0.285, 0.295, 0.30]

title_size = 54
cbar_label_size = 44
cbar_tick_size = 34
major_tick_size = 36
minor_tick_size = 28

fig, ax = plt.subplots()
fig.set_figwidth(24)
fig.set_figheight(18)
ax.tick_params(axis="x", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="x", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(axis="y", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="y", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$\\epsilon$", fontdict=fd, labelpad=20)
ax.set_ylabel("$s$   (kb)", fontdict=fd, labelpad=20)
im = ax.imshow(result_00, norm=LogNorm(vmin=result_min, vmax=result_max), cmap="Reds", interpolation="nearest", aspect="auto")
cbar = fig.colorbar(mappable=im, ax=ax, shrink=0.85)
cbar.set_label(label="MFET   (t.u.)", size=cbar_label_size, labelpad=20)
cbar.ax.tick_params(labelsize=cbar_tick_size)
ax.set_xticks(range(len(epsilons)), labels=[f"{e:.3f}" if e > 0.25 else 0 for e in epsilons])
# ax.set_xticks(range(0, 13, 3), labels=[f"{e:.3f}" if e > 0.23 else 0 for e in eps])
# ax.set_xticks(range(len(eps_minor)), labels=[f"{e:.3f}" for e in eps_minor])
ax.set_yticks(range(0, len(s_all)), labels=s_all)
ax.set_title("C0-C0 \n", fontsize=title_size)
plt.tight_layout()
# fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C0.png"), bbox_inches="tight", format="png", dpi=192)
fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C0.svg"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots()
fig.set_figwidth(24)
fig.set_figheight(18)
ax.tick_params(axis="x", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="x", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(axis="y", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="y", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$\\epsilon$", fontdict=fd, labelpad=20)
ax.set_ylabel("$s$   (kb)", fontdict=fd, labelpad=20)
im = ax.imshow(np.log2(result_00_diff[:,1:]), vmin=-10, vmax=10, cmap="RdBu_r", interpolation="nearest", aspect="auto")
cbar = fig.colorbar(mappable=im, ax=ax, shrink=0.85)
cbar.set_label(label="log2($MFET_{div}$)", size=cbar_label_size, labelpad=20)
cbar.ax.tick_params(labelsize=cbar_tick_size)
ax.set_xticks(range(len(epsilons[1:])), labels=[f"{e:.3f}" if e > 0.25 else 0 for e in epsilons[1:]])
# ax.set_xticks(range(0, 13, 3), labels=[f"{e:.3f}" if e > 0.23 else 0 for e in eps])
# ax.set_xticks(range(len(eps_minor)), labels=[f"{e:.3f}" for e in eps_minor])
ax.set_yticks(range(0, len(s_all)), labels=s_all)
ax.set_title("C0-C0 \n", fontsize=50)
plt.tight_layout()
# fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C0_log2-diff.png"), bbox_inches="tight", format="png", dpi=192)
fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C0_log2-diff.svg"), bbox_inches="tight")
plt.close(fig)


###

fig, ax = plt.subplots()
fig.set_figwidth(24)
fig.set_figheight(18)
ax.tick_params(axis="x", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="x", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(axis="y", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="y", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$\\epsilon$", fontdict=fd, labelpad=20)
ax.set_ylabel("$s$   (kb)", fontdict=fd, labelpad=20)
im = ax.imshow(result_11, norm=LogNorm(vmin=result_min, vmax=result_max), cmap="Reds", interpolation="nearest", aspect="auto")
cbar = fig.colorbar(mappable=im, ax=ax, shrink=0.85)
cbar.set_label(label="MFET   (t.u.)", size=cbar_label_size, labelpad=20)
cbar.ax.tick_params(labelsize=cbar_tick_size)
ax.set_xticks(range(len(epsilons)), labels=[f"{e:.3f}" if e > 0.25 else 0 for e in epsilons])
# ax.set_xticks(range(0, 13, 3), labels=[f"{e:.3f}" if e > 0.23 else 0 for e in eps])
# ax.set_xticks(range(len(eps_minor)), labels=[f"{e:.3f}" for e in eps_minor])
ax.set_yticks(range(0, len(s_all)), labels=s_all)
ax.set_title("C1-C1 \n", fontsize=title_size)
plt.tight_layout()
# fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C1-C1.png"), bbox_inches="tight", format="png", dpi=192)
fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C1-C1.svg"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots()
fig.set_figwidth(24)
fig.set_figheight(18)
ax.tick_params(axis="x", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="x", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(axis="y", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="y", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$\\epsilon$", fontdict=fd, labelpad=20)
ax.set_ylabel("$s$   (kb)", fontdict=fd, labelpad=20)
im = ax.imshow(np.log2(result_11_diff[:,1:]), vmin=-10, vmax=10, cmap="RdBu_r", interpolation="nearest", aspect="auto")
cbar = fig.colorbar(mappable=im, ax=ax, shrink=0.85)
cbar.set_label(label="log2($MFET_{div}$)", size=cbar_label_size, labelpad=20)
cbar.ax.tick_params(labelsize=cbar_tick_size)
ax.set_xticks(range(len(epsilons[1:])), labels=[f"{e:.3f}" if e > 0.25 else 0 for e in epsilons[1:]])
# ax.set_xticks(range(0, 13, 3), labels=[f"{e:.3f}" if e > 0.23 else 0 for e in eps])
# ax.set_xticks(range(len(eps_minor)), labels=[f"{e:.3f}" for e in eps_minor])
ax.set_yticks(range(0, len(s_all)), labels=s_all)
ax.set_title("C1-C1 \n", fontsize=title_size)
plt.tight_layout()
# fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C1-C1_log2-diff.png"), bbox_inches="tight", format="png", dpi=192)
fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C1-C1_log2-diff.svg"), bbox_inches="tight")
plt.close(fig)


###

fig, ax = plt.subplots()
fig.set_figwidth(24)
fig.set_figheight(18)
ax.tick_params(axis="x", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="x", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(axis="y", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="y", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$\\epsilon$", fontdict=fd, labelpad=20)
ax.set_ylabel("$s$   (kb)", fontdict=fd, labelpad=20)
im = ax.imshow(result_01, norm=LogNorm(vmin=result_min, vmax=result_max), cmap="Reds", interpolation="nearest", aspect="auto")
cbar = fig.colorbar(mappable=im, ax=ax, shrink=0.85)
cbar.set_label(label="MFET   (t.u.)", size=cbar_label_size, labelpad=20)
cbar.ax.tick_params(labelsize=cbar_tick_size)
ax.set_xticks(range(len(epsilons)), labels=[f"{e:.3f}" if e > 0.25 else 0 for e in epsilons])
# ax.set_xticks(range(0, 13, 3), labels=[f"{e:.3f}" if e > 0.23 else 0 for e in eps])
# ax.set_xticks(range(len(eps_minor)), labels=[f"{e:.3f}" for e in eps_minor])
ax.set_yticks(range(0, len(s_all)), labels=s_all)
ax.set_title("C0-C1 \n", fontsize=title_size)
plt.tight_layout()
# fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C1.png"), bbox_inches="tight", format="png", dpi=192)
fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C1.svg"), bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots()
fig.set_figwidth(24)
fig.set_figheight(18)
ax.tick_params(axis="x", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="x", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(axis="y", labelsize=major_tick_size, which="major", pad=8, length=8, width=1.1)
ax.tick_params(axis="y", labelsize=minor_tick_size, which="minor", pad=8, length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$\\epsilon$", fontdict=fd, labelpad=20)
ax.set_ylabel("$s$   (kb)", fontdict=fd, labelpad=20)
im = ax.imshow(np.log2(result_01_diff[:,1:]), vmin=-10, vmax=10, cmap="RdBu_r", interpolation="nearest", aspect="auto")
cbar = fig.colorbar(mappable=im, ax=ax, shrink=0.85)
cbar.set_label(label="log2($MFET_{div}$)", size=cbar_label_size, labelpad=20)
cbar.ax.tick_params(labelsize=cbar_tick_size)
ax.set_xticks(range(len(epsilons[1:])), labels=[f"{e:.3f}" if e > 0.25 else 0 for e in epsilons[1:]])
# ax.set_xticks(range(0, 13, 3), labels=[f"{e:.3f}" if e > 0.23 else 0 for e in eps])
# ax.set_xticks(range(len(eps_minor)), labels=[f"{e:.3f}" for e in eps_minor])
ax.set_yticks(range(0, len(s_all)), labels=s_all)
ax.set_title("C0-C1 \n", fontsize=title_size)
plt.tight_layout()
# fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C1_log2-diff.png"), bbox_inches="tight", format="png", dpi=192)
fig.savefig(path.join("./results/latest/state-open/", "FET_vs_eps_block_copolymer_C0-C1_log2-diff.svg"), bbox_inches="tight")
plt.close(fig)