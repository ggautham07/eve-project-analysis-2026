import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
import h5py

from polychrom.hdf5_format import load_hdf5_file
from cooltools import expected_cis
import cooler
import cooltools.lib.plotting

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from os import path
import sys
import json
from glob import glob


def generate_bins(N, start=1, bins_per_order_magn=20):
    lstart = np.log10(start)
    lend = np.log10(N - 1) + 1e-6
    num = int(np.ceil((lend - lstart) * bins_per_order_magn))
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=max(num, 0)))
    if len(bins) > 0:
        assert bins[-1] == N - 1
    return bins


args = sys.argv
tag = args[1]
try:
    threshold = args[2]
except IndexError:
    threshold = 250     # in nm

dir_results = f"./analyses/{tag}/"

# Retrieve experimental data, normalise probability curve
file_exp_data = "./resources/processed_data/Hi-C_nc14_1kb.mcool"
clr_obj = cooler.Cooler(f"{file_exp_data}::resolutions/16000")
map_start, map_end = 6464000, 10672000
header_list = ["chrom", "start", "end", "name"]
values_list = [["2R"], [map_start], [map_end], ["eve"]]
view_df_cvd = pd.DataFrame(data=dict(zip(header_list, values_list)))
cvd = expected_cis(
     clr=clr_obj,
     view_df=view_df_cvd,
     smooth=False,
     aggregate_smoothed=False,
     )
genomic_dist = cvd["dist"][2:] * 16
probs_contact_exp_norm = cvd["balanced.avg"][2:] / np.sum(cvd["balanced.avg"][2:])
assert np.sum(probs_contact_exp_norm) == 1

contact_scaling = {}
contact_scaling["Experiment"] = probs_contact_exp_norm
with h5py.File(path.join(dir_results, "analysis.h5"), mode="r") as handler:
    try:
        genomic_dist = handler["s/Pc"][:]
    except KeyError:
        pass
    contact_scaling["Simulation"] = handler[f"Pc(s)/{threshold}"][:]
assert np.allclose(np.sum(contact_scaling["Simulation"]), 1, atol=1e-4)

# print(genomic_dist, len(genomic_dist))
idxs = generate_bins(N=len(genomic_dist)+1, start=1, bins_per_order_magn=10) - 1
# print(idxs)
# print(len(contact_scaling["Experiment"]), len(contact_scaling["Simulation"]))
# print(genomic_dist[idxs])
# sigma = np.sum(np.abs(contact_scaling["Experiment"][:1000] - contact_scaling["Simulation"][:1000]))
# print(np.asarray(contact_scaling["Experiment"]))
sigma = np.sum(np.abs(np.asarray(contact_scaling["Experiment"])[idxs] - contact_scaling["Simulation"][idxs]))

plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 40}
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
ax.set_xscale("log")
ax.set_yscale("log")
fig.set_layout_engine("tight")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("$s$   (kb)", fontdict=fd, labelpad=24)
ax.set_ylabel("$P_c$", fontdict=fd, labelpad=24)
ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
for i, (label, data) in enumerate(contact_scaling.items()):
    ax.plot(cvd["dist"][2:] * 16 if label == "Experiment" else genomic_dist,
            data,
            lw=7,
            color=default_colours[7]+"99" if label == "Experiment" else default_colours[1],
            label=label,
            ls=(0, (1, 1)) if label == "Experiment" else "solid",
            )
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize="34", frameon=False, borderpad=1)
fig.savefig(path.join(dir_results, "Pc(s)_fit.svg"), bbox_inches="tight", format="svg")
plt.close(fig)

print(sigma)
sys.exit(0)