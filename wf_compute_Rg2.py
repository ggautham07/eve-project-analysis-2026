from os import path
import sys
from glob import glob
import json
import h5py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polychrom.hdf5_format import list_URIs, load_URI


def Rg2(config):
    """Compute the squared radius of gyration for a polymer conformation"""
    assert config.ndim == 2
    assert config.shape[1] == 3
    return np.mean(np.linalg.norm(config - np.mean(config, axis=0), axis=-1) ** 2)

# Handle arguments
args = sys.argv
tag = args[1]
sim_dirs = sorted(glob(path.join(f"./trajectories/{tag}*")))
print("Found the following folders", *sim_dirs, sep="\n")
flag_mult = True if len(sim_dirs) > 1 else False
try:
    start = int(args[2])
except IndexError:
    print("Start block not specified. Defaulting to start = 1000")
    start = 1000
step = int(args[3])

dir_results = f"results/latest/sims/{tag}" if flag_mult else sim_dirs[0]
try:
    assert path.exists(path.join(dir_results, "analysis.h5"))
except AssertionError:
    print("Analysis file does not exist, will be created.")

params = json.load(open(path.join(sim_dirs[0], "parameters.json"), mode="r"))
N = params["polymer_length"]
bc = False
le = False
if "interaction_matrix" in list(params.keys()):
    eps = params["interaction_matrix"][0][0]
    bc = True
if path.exists(path.join(sim_dirs[0], "parameters_loop_extrusion.json")):
    params = json.load(open(path.join(sim_dirs[0], "parameters_loop_extrusion.json"), mode="r"))
    unload_prob = params["LEF_unload_prob"]
    load_prob = params["LEF_load_prob"]
    step_prob = params["LEF_step_prob"]
    le = True

URIs = []
for sim_dir in sim_dirs:
    URIs.extend(list_URIs(sim_dir)[start::step])

monomer_classes_file_path = "./resources/processed_data/evec_region.csv"
raw_evec_data = pd.read_csv(monomer_classes_file_path, sep="\t")
# getting linear loci positions from the data
MS2_pos = np.array([9969, 9985, 9969, 9985, 9969, 9985, 9985]) - (raw_evec_data["start"][0] // 1000)
parS_pos = np.array([10027, 9903, 10057, 9836, 10159, 9390, 6657]) - (raw_evec_data["start"][0] // 1000)
genomic_dists = abs(MS2_pos - parS_pos)
loci_positions = [(min(MS2_pos[i], parS_pos[i]), max(MS2_pos[i], parS_pos[i])) for i in range(len(MS2_pos))]

Rg2_traj = np.empty(len(URIs))
Rg2_s = np.empty((len(URIs), len(genomic_dists)))
for i, URI in enumerate(URIs):
    config = load_URI(URI)["pos"]
    Rg2_traj[i] = Rg2(config)
    for j in range(len(loci_positions)):
        Rg2_s[i,j] = Rg2(config[loci_positions[j][0]:loci_positions[j][1]])

with h5py.File(path.join(dir_results, "analysis.h5"), mode="a") as handler:
    if "Rg2" in handler.keys():
        del handler["Rg2"]
    handler.create_dataset(name="Rg2", data=Rg2_traj[:])
    if "Rg2(s)" in handler.keys():
        del handler["Rg2(s)"]
    handler.create_dataset(name="Rg2(s)", data=Rg2_s[:])

print(np.mean(Rg2_s, axis=0))

default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab20090", "#008f4c90", "#009ba590", "#0085b690", "#004e9e90", "#00177d90", "#00053d90"]
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 16}

x_axis = np.arange(start, start + (len(URIs) * step), step)
plt.figure(figsize=(9, 7))
plt.plot(x_axis, Rg2_traj, linewidth=2, label=f"$\\overline{{R_g^2}} = {np.mean(Rg2_traj):.2f}$")
plt.xlabel("sampled conformation index (increasing time)", fontdict=fd)
plt.ylabel("squared radius of gyration", fontdict=fd)
title = "Squared radius of gyration over trajectory"
if le and not bc:
    title = title + f"\nLoop extrusion\n$N = {N}$, $p_U = {unload_prob}$, $p_L = {load_prob}$, $p_S = {step_prob}$"
elif bc and not le:
    title = title +  f"\nBlock copolymer\n$N={4208}, \\epsilon={eps:.3f}$"
elif bc and le:
    title = title + f"\nBlock copolymer + loop extrusion\n$N = {N}$, $\\epsilon={eps:.3f}$, $p_U = {unload_prob}$, $p_L = {load_prob}$, $p_S = {step_prob}$"
else:
    title = title + f"\nRouse model, $N={4208}$"
plt.title(title, fontsize=18)
plt.grid(lw=0.2)
plt.legend(fontsize=14)
plt.savefig(path.join(dir_results, "Rg2_traj.svg"), format="svg")
plt.close()

Rg2_s_mean = np.mean(Rg2_s, axis=0)
slope, intercept = np.polyfit(genomic_dists, Rg2_s_mean, deg=1)

plt.figure(figsize=(9, 7))
plt.xscale("log")
plt.yscale("log")
plt.scatter(genomic_dists, Rg2_s_mean, marker="o", s=150, color=default_colours[0]+"99", label=f"$R_g^2 \\sim s^{{{slope:.2f}}}$")
# plt.plot(genomic_dists, (genomic_dists ** slope), "--", color=default_colours[0], label=f"$R_g^2 \\sim s^{{{slope:.2f}}}$")
plt.xlabel("genomic distance ($s$)", labelpad=10, fontdict=fd)
plt.ylabel("mean-squared gyration radius ($R_g^2$)", labelpad=10, fontdict=fd)
title = "Squared radius of gyration scaling"
if not bc and le:
    title = title + f"\nLoop extrusion, $N = {N / 1000:.1f} \\text{{Mb}}$, $p_U = {unload_prob}$, $p_L = {load_prob}$, $p_S = {step_prob}$"
elif bc and not le:
    title = title +  f"\nBlock copolymer, $N={N / 1000:.1f} \\text{{Mb}}, \\epsilon={eps:.3f}$"
elif bc and le:
    title = title + f"\nBlock copolymer + loop extrusion, $N = {N / 1000:.1f} \\text{{Mb}}$\n\\epsilon={eps:.3f}, $p_U = {unload_prob}$, $p_L = {load_prob}$"
else:
    title = title + f"\nRouse model, $N={N / 1000:.1f} \\text{{Mb}}$"
plt.title(title, fontsize=18)
plt.legend(fontsize=16)
plt.savefig(path.join(dir_results, "Rg2(s).svg"), format="svg")
plt.close()