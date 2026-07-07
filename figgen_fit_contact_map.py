import sys
from os import path
import h5py

import numpy as np
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import matplotlib.cm

from cooltools.lib.numutils import observed_over_expected


# Normalisation of a Hi-C map
def SCN(D0, max_iter=10):
	"""
    Matrix normalisation using Sequential Component Normalization
    Out : SCN(D)
    Code version from Vincent Matthys
    """
	D = D0.astype(np.float32)
	# Iteration over max_iter
	for i in range(max_iter):
		D /= np.maximum(1, D.sum(axis = 0))       
		D /= np.maximum(1, D.sum(axis = 1)[:, None])    
		# To make matrix symetric again   
	return (D + D.T) / 2

# Colour map for Hi-C
fall = np.array(
        (
            (255, 255, 255),
            (255, 255, 204),
            (255, 237, 160),
            (254, 217, 118),
            (254, 178, 76),
            (253, 141, 60),
            (252, 78, 42),
            (227, 26, 28),
            (189, 0, 38),
            (128, 0, 38),
            (0, 0, 0),
        )
    )
def list_to_colormap(color_list, name=None):
    color_list = np.array(color_list)
    if color_list.min() < 0:
        raise ValueError("Colors should be 0 to 1, or 0 to 255")
    if color_list.max() > 1.0:
        if color_list.max() > 255:
            raise ValueError("Colors should be 0 to 1 or 0 to 255")
        else:
            color_list = color_list / 255.0
    return LinearSegmentedColormap.from_list(name, color_list, 256)
matplotlib.cm.register_cmap("fall", list_to_colormap(fall))
matplotlib.cm.register_cmap("fall" + "_r", list_to_colormap(fall[::-1]))


# Parsing arguments
args = sys.argv
tag = args[1]
try:
    threshold = int(args[2])
except IndexError:
    threshold = 250

with h5py.File(path.join("./analyses/", tag, "analysis.h5"), mode="r") as handler:
# with h5py.File(path.join("./results/latest/sims/", tag, "analysis.h5"), mode="r") as handler:
    sim_matrix_avg = handler[f"Hi-C_matrix/{threshold}"][:]

exp_matrix_avg = SCN(np.load("./resources/processed_data/Hi-C_matrix_16kbres.npy", allow_pickle=True))

# print(np.sum(np.isnan(sim_matrix_avg)))
# print(np.sum(np.isnan(exp_matrix_avg)))

tag_saving = "state-open" # "off_states"

default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 36}

premask = np.zeros_like(exp_matrix_avg)
np.fill_diagonal(premask, np.nan)
premask[np.tril_indices_from(premask)] = 1
exp_tri = np.ma.masked_array(exp_matrix_avg, premask == 0)
sim_tri = np.ma.masked_array(sim_matrix_avg, premask == 1)

fig, ax = plt.subplots(figsize=(10, 10))
fig.set_layout_engine("tight")
im_exp = ax.matshow(exp_tri, norm=LogNorm(vmin=1e-5, vmax=1e-1), cmap="fall");
im_sim = ax.matshow(sim_tri, norm=LogNorm(vmin=1e-5, vmax=1e-1), cmap="fall");   # (vmin=1e-8, vmax=6e0) # vmin=1e-7, vmax=4e0
cbar_exp = plt.colorbar(im_exp, ax=ax, orientation="horizontal", anchor=(0, 1.0), fraction=0.046, pad=0.04);
cbar_sim = plt.colorbar(im_sim, ax=ax, orientation="vertical", fraction=0.046, pad=0.04);
cbar_exp.set_label(label="normalized contact frequency", size=28, labelpad=16)
cbar_exp.ax.tick_params(
    length=2,
    width=0.8,
    labelsize=20,
    )
cbar_sim.ax.tick_params(
    length=2,
    width=0.8,
    labelsize=20,
    )
ax.tick_params(
    axis="x",
    which="both",
    labelbottom=False,
    labeltop=True,
    labelsize=20,
    length=5,
    width=1.5,
)
ax.tick_params(
    axis="y",
    which="both",
    left=True,
    right=True,
    length=5,
    width=1.5,
)
ax.ticklabel_format(useOffset=False, style="plain")
# ax.set_xticks([7e6, 8e6, 9e6, 1e7], [7, 8, 9, 10])
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.xaxis.set_label_position('top')
ax.tick_params(labelsize=20)
# fig.savefig(f"./results/latest/state-open/.svg", format="svg")
fig.savefig(path.join("./analyses/", tag, f"Hi-C_map_{threshold}-nm.svg"), format="svg")
plt.close(fig)


# Fitting the Hi-C
interlocus_dists = np.arange(1, len(exp_matrix_avg), 1)
corrs = []
for dg in interlocus_dists:
    contacts_sim_dg = np.diag(sim_matrix_avg, k=dg)
    contacts_exp_dg = np.diag(exp_matrix_avg, k=dg)
    corrs.append(np.corrcoef(contacts_exp_dg, contacts_sim_dg)[0,1])
corr_final = np.nanmean(corrs)
corr_final = np.corrcoef(sim_matrix_avg, exp_matrix_avg)[0, 1]

from cooltools.lib.numutils import observed_over_expected
from scipy.stats import spearmanr

corr_final = np.corrcoef(observed_over_expected(exp_matrix_avg)[0].flatten(), observed_over_expected(sim_matrix_avg)[0].flatten())[0,1]
# corr_final = spearmanr(observed_over_expected(exp_matrix_avg)[0].flatten(), observed_over_expected(sim_matrix_avg)[0].flatten())[0]

print("Correlation: ", corr_final)

# Printing error for redirecting out to file
# sigma = 1 - np.abs(corr_final)
# print(sigma)

sys.exit(0)