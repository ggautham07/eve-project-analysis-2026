import numpy as np
from polychrom.hdf5_format import list_URIs
import cooler
from cooltools import expected_cis
from polychrom.contactmaps import binnedContactMap
import h5py
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from os import path
from glob import glob
import argparse
import pandas as pd
import json


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


parser = argparse.ArgumentParser(description="Compute probability of contact scaling from simulations")
parser.add_argument("--sim_dirs",
                    required=True,
                    type=str,
                    help="Directories housing simulation trajectories")  
parser.add_argument("--start",
                    required=False,
                    type=int,
                    default=0,
                    help="Start index (default=0)")
parser.add_argument("--end",
                    required=False,
                    type=int,
                    default=-1,
                    help="End index (default=-1)")
parser.add_argument("--step",
                    required=False,
                    type=int,
                    default=1,
                    help="Step every indices (default=1)")  
args = parser.parse_args()

if args.sim_dirs.endswith("*"):
    args.sim_dirs = glob(args.sim_dirs)
else:
    args.sim_dirs = [args.sim_dirs]
for sd in args.sim_dirs:
    assert path.isdir(sd)

print("Taking the following simulation trajectories:")
print(*args.sim_dirs, end="\n")
print("The following parameters apply to all of them")
print(f"Start index: {args.start}; End index: {args.end}; Stepping every: {args.step}")


# Load translation from fit
tagno = path.basename(args.sim_dirs[0]).split("_")[0]
if tagno != "test":
    info_fits = json.load(open(f"./results/latest/state-open/fits_{tagno}.json"))
if tagno == "00":
    keyparam = "rouse_excvol" if "excvol" in args.sim_dirs[0] else "rouse"
elif tagno == "01":
    params = json.load(open(path.join(args.sim_dirs[0], "parameters_loop_extrusion.json")))
    prob_load = params["LEF_load_prob"]
    keyparam = prob_load
    # prob_unload = params["LEF_unload_prob"]
    # keyparam = prob_load / prob_unload
elif tagno == "02" or tagno == "05":
    keyparam = json.load(open(path.join(args.sim_dirs[0], "parameters.json")))["interaction_matrix"][1][1]
elif tagno == "03":
    eps = json.load(open(path.join(args.sim_dirs[0], "parameters.json")))["interaction_matrix"][1][1]
    p_L = json.load(open(path.join(args.sim_dirs[0], "parameters_loop_extrusion.json")))["LEF_load_prob"]
    keyparam = f"{eps:.3f}__{p_L:.2f}"
else:
    d_fit = -1.7
    keyparam = None
if tagno != "test":
    d_fit = info_fits["d_fit"]["R(s)"][str(keyparam)]
print(f"Tag: {tagno}; Key parameter value: {keyparam}, Translation: {d_fit:.3f}")

# Load all the data
URIs = []
for sim_dir in args.sim_dirs:
    URIs.extend(list_URIs(sim_dir)[args.start:args.end:args.step])
print(f"Found {len(URIs)} conformations")

thresholds = [100, 150, 200, 250, 300, 350, 400]    # in nm, will be different s.u. for different simulations

map_by_threshold = {}
Pc_by_threshold = {}
for threshold in thresholds:
    cutoff = threshold * (10 ** d_fit)
    print(f"{threshold} nm in monomer units: {cutoff}")
    sim_matrix_avg, _ = binnedContactMap(filenames=URIs,
                                         chains=None,
                                         binSize=16,
                                         cutoff=cutoff,
                                         n=32,)
    sim_matrix_avg = SCN(sim_matrix_avg)
    map_by_threshold[threshold] = sim_matrix_avg[:]
    # Extracting contact freqs by s and normalising
    data_by_diag = []
    for i in range(2, sim_matrix_avg.shape[0]):
        data_by_diag.append(np.nanmean(np.diag(sim_matrix_avg, k=i)))
    Pc = np.asarray(data_by_diag) / np.sum(data_by_diag)
    assert np.allclose(np.sum(Pc), 1, atol=1e-4)
    Pc_by_threshold[threshold] = Pc

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

# with h5py.File(path.join("./analyses/", path.basename(sim_dir), "analysis.h5"), mode="a") as handler:
with h5py.File(path.join("./results/latest/sims/", path.basename(sim_dir), "analysis.h5"), mode="a") as handler:
    if "P(s)" in handler.keys():
        del handler["P(s)"]
    if "contacts_k=5" in handler.keys():
        del handler["contacts_k=5"]
    if "contacts_k=5_scaling" in handler.keys():
        del handler["contacts_k=5_scaling"]
    if "contacts_scaling_genomic_dists" in handler.keys():
        del handler["contacts_scaling_genomic_dists"]
    for key in Pc_by_threshold.keys():
        if f"Hi-C_matrix/{key}" in handler.keys():
            del handler[f"Hi-C_matrix/{key}"]
        handler.create_dataset(name=f"Hi-C_matrix/{key}", data=map_by_threshold[key])
        if f"Pc(s)/{key}" in handler.keys():
            del handler[f"Pc(s)/{key}"]
        handler.create_dataset(name=f"Pc(s)/{key}", data=Pc_by_threshold[key])
    if "s/Pc" in handler.keys():
        del handler["s/Pc"]
    handler.create_dataset(name="s/Pc", data=genomic_dist)
    if not "s/org" in handler.keys():
        handler.create_dataset(name="s/org", data=[58, 82, 88, 149, 190, 595, 3327])

# Pc_by_threshold = dict(zip([str(t) for t in thresholds], [[] for _ in range(len(thresholds))]))
# for i, u in enumerate(URIs):
#     data = load_URI(u)["pos"]
#     N = len(data)
#     tree = cKDTree(data)
#     bins = generate_bins(N)
#     for threshold in thresholds:
#         s, Pc = contact_scaling(tree, threshold, bins, N)
#         Pc_by_threshold[str(threshold)].append(Pc)
#     if i % 1000 == 0 or i == len(URIs) - 1:
#         print(f"Contacts computed until conformation {i}")
# save_hdf5_file(path.join(sim_dir, "P(s).h5"), Pc_by_threshold)
# print(f"Saved contacts for thresholds {thresholds}")

# default_colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
# plt.figure(figsize=(9, 7))
# for i, threshold in enumerate(thresholds):
#     contact_scaling_final = np.mean(np.asarray(Pc_by_threshold[str(threshold)]), axis=0)
#     plt.plot(s, contact_scaling_final, linewidth=3, color=default_colours[i], label=f"$k = {threshold}$")
#     if threshold < 3:
#         start = 4
#         stop = 10
#         x = np.log10(s)[start:stop]
#         y = np.log10(contact_scaling_final)[start:stop]
#         slope_sim, intercept_sim = np.polyfit(x, y, deg=1)
#         x_axis = np.linspace(s[start], s[stop], num=100)
#         plt.plot(x_axis, x_axis ** slope_sim * (10 ** (intercept_sim - 0.2)), "--", color=default_colours[i], label=f"$\\beta = {slope_sim:.3f}$")
# plt.xscale("log")
# plt.yscale("log")
# plt.grid(lw=0.15)
# plt.xlabel("$s$")
# plt.ylabel("$P(s)$")
# plt.legend()
# plt.savefig(path.join(sim_dir, "P(s).png"))