import numpy as np
from polychrom.hdf5_format import load_hdf5_file
from h5py import File
import matplotlib.pyplot as plt
from os import path
import sys
import json

args = sys.argv
tag = args[1]
sim_dir = f"./trajectories/{tag}"
analyses_dir= f"./analyses/{tag}"
try:
    start = int(args[2])
    # print(start)
except IndexError:
    print("Start block not specified. Defaulting to start = 1000")
    start = 1000
step = int(args[3])
# print(step)
try:
    assert path.exists(path.join(analyses_dir, "analysis.h5"))
except AssertionError:
    print("Analysis file does not exist, will create it")

try:
    coordinates = load_hdf5_file(path.join(sim_dir, "selected_monomer_coordinates.h5"))["trajectory"]
except FileNotFoundError:
    coordinates = load_hdf5_file(path.join(sim_dir, "E-P_trajectory.h5"))["trajectory"]
num_rows, num_cols, _ = coordinates[0].shape
R = np.linalg.norm(coordinates[..., 1, :] - coordinates[..., 0, :], axis=-1)
with File(path.join(analyses_dir, "analysis.h5"), mode="a") as handler:
    if "R(s)" in handler.keys():
        del handler["R(s)"]
    handler.create_dataset(name="R(s)", data=R[start::step])

genomic_dists = [58, 82, 88, 149, 190, 595, 3327]
genomic_dists_log = np.log10(genomic_dists)
R_mean = np.mean(R, axis=0)
R_mean_log = np.log10(R_mean)
slope, intercept = np.polyfit(genomic_dists_log, R_mean_log, deg=1)

params = json.load(open(path.join(sim_dir, "parameters.json"), mode="r"))
N = params["polymer_length"]
bc = False
le = False
if "interaction matrix" in list(params.keys()):
    eps = params["interaction_matrix"][0][0]
    bc = True
if path.exists(path.join(sim_dir, "parameters_loop_extrusion.json")):
    params = json.load(open(path.join(sim_dir, "parameters_loop_extrusion.json"), mode="r"))
    unload_prob = params["LEF_unload_prob"]
    load_prob = params["LEF_load_prob"]
    step_prob = params["LEF_step_prob"]
    le = True

plt.figure(figsize=(11, 9))
for i in range(len(genomic_dists)):
    plt.hist(R[:, i], bins=50, density=True, edgecolor=gregor_colours[i], linewidth=2, histtype="step", label=f"{genomic_dists[i]} kb");
plt.xlabel("end-to-end distance", labelpad=10, fontdict=fd)
plt.ylabel("density", labelpad=10, fontdict=fd)
title = "End-to-end distance distribution"
if not bc and le:
    title = title + f" | Loop extrusion, $N = {N}$, $p_U = {unload_prob}$, $p_L = {load_prob}$"
elif bc and not le:
    title = title +  f" | Block copolymer, $N={4208}, \\epsilon={eps:.3f}$"
elif bc and le:
    title = title + f" Block copolymer + loop extrusion, $N = {N}$\n\\epsilon={eps:.3f}, $p_U = {unload_prob}$, $p_L = {load_prob}$, $p_S = {step_prob}$"
else:
    title = title + f" | Rouse model, $N={4208}$"
plt.title(title, fontsize=16)
plt.legend(fontsize=14)
plt.grid(lw=0.2)
plt.savefig(path.join(sim_dir, "P(R)_explike.png"), format="png", dpi=300)
plt.close()

default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab20090", "#008f4c90", "#009ba590", "#0085b690", "#004e9e90", "#00177d90", "#00053d90"]
plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 16}

plt.figure(figsize=(9, 7))
plt.xscale("log")
plt.yscale("log")
plt.scatter(genomic_dists, R_mean, marker="o", s=150, color=default_colours[0]+"99")
plt.plot(genomic_dists, (genomic_dists ** slope) * (10 ** intercept), "--", color=default_colours[0], label=f"$R \\sim s^{{{slope:.2f}}}$")
plt.xlabel("genomic distance ($s$)", labelpad=10, fontdict=fd)
plt.ylabel("mean end-to-end distance ($R$)", labelpad=10, fontdict=fd)
title = "End-to-end distance scaling"
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
plt.savefig(path.join(sim_dir, "R(s).svg"), format="svg")
plt.close()