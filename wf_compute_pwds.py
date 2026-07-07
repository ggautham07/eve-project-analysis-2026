import numpy as np
from scipy.spatial.distance import pdist, squareform
from polychrom.hdf5_format import list_URIs, load_URI, load_hdf5_file, save_hdf5_file
import matplotlib.pyplot as plt
from os import path
import sys

args = sys.argv
sim_dir = args[1]
assert path.isdir(sim_dir)
try:
    recompute = bool(int(args[2].strip()))
except IndexError:
    recompute = False
    print("Recompute = FALSE")
idx = sim_dir.index("=")+1
eps = float(sim_dir[idx:idx+5])
print(f"Simulation: eps = {eps}...")

try:
    if recompute:
        raise FileNotFoundError
    pairwise_distances = load_hdf5_file(path.join(sim_dir, "pairwise_distances_16kbres.npy"))["pairwise_distances"]
except FileNotFoundError:
    URIs = list_URIs(sim_dir)[:]
    sample_data = load_URI(URIs[0])["pos"]
    N0 = sample_data.shape[0]
    N = N0 // 16
    bins0 = np.linspace(start=0, stop=N0, num=N+1, dtype=int)
    bins = np.concatenate((bins0[:-1][...,np.newaxis], bins0[1:][...,np.newaxis]), axis=1)
    pairwise_distances = np.empty((len(URIs), N * (N - 1) // 2))
    for i in range(len(URIs)):
        data = load_URI(URIs[i])["pos"]
        data_cg = np.empty((N, 3))
        for n, start, end in zip(np.arange(N), bins[:, 0], bins[:, 1]):
            data_cg[n, :] = np.mean(data[start:end, :], axis=0)
        pairwise_distances[i] = pdist(data_cg)
    save_hdf5_file(path.join(sim_dir, "pairwise_distances.h5"), {"pairwise_distances": pairwise_distances})

pairwise_distances_ensemble = squareform(np.mean(pairwise_distances, axis=0))
np.fill_diagonal(pairwise_distances_ensemble, 0)
plt.rcParams["font.family"] = ["serif", "sans-serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.figure(figsize=(13, 11))
plt.imshow(pairwise_distances_ensemble, cmap="inferno")
plt.xticks(np.arange(0.5, pairwise_distances_ensemble.shape[0], 1), minor=True)
plt.yticks(np.arange(0.5, pairwise_distances_ensemble.shape[0], 1), minor=True)
plt.tick_params(which="minor", bottom=False, left=False)
plt.grid(which="minor", color="white", linewidth=0.05)
plt.colorbar(label="distance", shrink=0.95)
plt.title(f"PWD map from block copolymer simulation | $n = {pairwise_distances.shape[0]}$, $\\epsilon={eps}$");
plt.savefig(path.join(sim_dir, "pwd_map_16kbres.png"), dpi=300)
plt.close()
