from os import path
import sys
import numpy as np
import pandas as pd
import h5py
from cooltools import expected_cis
import cooler
import cooltools.lib.plotting
import matplotlib.pyplot as plt

# Generate log-spaced bins
def generate_bins(N, start=1, bins_per_order_magn=20):
    lstart = np.log10(start)
    lend = np.log10(N - 1) + 1e-6
    num = int(np.ceil((lend - lstart) * bins_per_order_magn))
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=max(num, 0)))
    if len(bins) > 0:
        assert bins[-1] == N - 1
    return bins

# Handle arguments to extract unique simulation tag and contact threshold 
args = sys.argv
tag = args[1]
try:
    threshold = args[2]
except IndexError:
    threshold = 250     # in nm

dir_results = f"./analyses/{tag}/"

# Retrieve experimental data, compute P(s) and normalise probability curve
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
genomic_dists = cvd["dist"][2:] * 16
probs_contact_exp_norm = cvd["balanced.avg"][2:] / np.sum(cvd["balanced.avg"][2:])
assert np.sum(probs_contact_exp_norm) == 1

# Create dictionary with labelled P(s) curve
contact_scaling = {}
contact_scaling["Experiment"] = probs_contact_exp_norm

# Load computed P(s) from simulation analysis file and add to dictionary
with h5py.File(path.join(dir_results, "analysis.h5"), mode="r") as handler:
    try:
        genomic_dists = handler["s/Pc"][:]
    except KeyError:
        pass
    contact_scaling["Simulation"] = handler[f"Pc(s)/{threshold}"][:]
assert np.allclose(np.sum(contact_scaling["Simulation"]), 1, atol=1e-4)

# Extract log-spaces values from P(s) curves to compute the deviation between simulation and experiment
idxs = generate_bins(N=len(genomic_dists)+1, start=1, bins_per_order_magn=10) - 1
sigma = np.sum(np.abs(np.asarray(contact_scaling["Experiment"])[idxs] - contact_scaling["Simulation"][idxs]))

# Parameters for plotting
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 40}
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot a manuscript-ready figure and save it
fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
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
    ax.plot(cvd["dist"][2:] * 16 if label == "Experiment" else genomic_dists,
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

# Display important results from fitting procedure
print(f"Deviation from experiment: {sigma}")