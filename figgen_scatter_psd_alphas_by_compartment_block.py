import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.fft import dct
from polychrom.hdf5_format import list_URIs, load_URI, save_hdf5_file, load_hdf5_file
import pandas as pd
from itertools import chain
from glob import glob
import seaborn as sns

plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 48}
default_colours = sns.color_palette("magma")     # plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#009ba5", "#008f4c", "#0085b6", "#004e9e", "#00177d", "#00053d"]


# Method for squared discrete cosine transform from Michael
def dct2(data):
    """
    From Michael Liefsons
    """
    from scipy.fft import dct
    N = data.shape[1]
    inv_norm = 1 / (2 * N)
    return inv_norm * dct(data, axis=1)


# Getting the monomer compartment blocks
monomer_classes_file_path = "./resources/processed_data/evec_region.csv"
raw_evec_data = pd.read_csv(monomer_classes_file_path, sep="\t")
compartments = (raw_evec_data["E1"].to_numpy() >= 0).astype(int)
monomer_classes = np.asarray(list(chain.from_iterable([[compartments[i] for _ in range(16)] \
                                                       for i in range(len(compartments))])))

compartment_switch_pos = np.nonzero(monomer_classes[1:] - monomer_classes[:-1])[0]
compartment_switch_pos = np.concatenate([[0], compartment_switch_pos])
compartment_lengths = compartment_switch_pos[1:] - compartment_switch_pos[:-1]

# Getting all simulation directories and labels
sim_dirs = ["./trajectories/00_Rouse",
            "./trajectories/02_block_copolymer_eps=0.200",
            "./trajectories/02_block_copolymer_eps=0.275",
            "./trajectories/02_block_copolymer_eps=0.290",
            "./trajectories/02_block_copolymer_eps=0.320"]
epsilons = [0] + [float(sd.split("=")[-1]) for sd in sim_dirs[1:]]
labels = [f"$\\epsilon = {e:.3f}$" if e > 0 else "Ideal" for e in epsilons]
cmap = mpl.colormaps["winter"]
colours = cmap(np.linspace(0, 1, len(sim_dirs)))

RECOMPUTE = False

if RECOMPUTE:

    alphas = np.empty((len(compartment_lengths), len(sim_dirs)))

    for i, (sim_dir, label) in enumerate(zip(sim_dirs, labels)):

        # Extracting trajectories
        URIs = list_URIs(sim_dir)[10000::1000]     # for eve locus simulations
        sample = load_URI(URIs[0])["pos"]
        N = sample.shape[0]
        Z = len(URIs)
        configs = np.empty((Z, N, 3))
        for z in range(Z):
            data = load_URI(URIs[z])["pos"]
            configs[z,:] = data[:]

        for j, c_len in enumerate(compartment_lengths):
            comp = 0 if j % 2 == 0 else 1
            c_len = compartment_lengths[j]
            idxs = (compartment_switch_pos[j], compartment_switch_pos[j+1])
            configs_filt = configs[:, idxs[0]:idxs[1], :]
            psd_filt = np.square(np.abs(dct2(configs_filt))).mean(axis=0).sum(axis=-1)
            N2 = configs_filt.shape[1]
            x_ax = np.log(np.arange(1, N2))
            slope, _ = np.polyfit(x_ax[:3], np.log(psd_filt)[1:4], deg=1)
            alphas[j,i] = slope

    save_hdf5_file(filename="./results/latest/sims/psd_alphas_selective_by-compartment-size.h5",
                data_dict={"s": compartment_lengths, "alpha": alphas})

else:
    data = load_hdf5_file("./results/latest/sims/psd_alphas_selective_by-compartment-size.h5")
    compartment_lengths = data["s"][:]
    alphas = data["alpha"][:]

markers = ["o", "X", "s", "*", "^", "p", "d"]
fig, ax = plt.subplots()
fig.set_figwidth(12)
fig.set_figheight(10)
fig.set_layout_engine("tight")
for i in range(len(epsilons)):
    ax.scatter(compartment_lengths, alphas[:,i],
               marker=markers[i], s=180, edgecolor=default_colours[i], color=default_colours[i],
               label=labels[i] if epsilons[i] > 0 else "Ideal")
    slope, intercept = np.polyfit(x=compartment_lengths, y=alphas[:,i], deg=1)
    print(epsilons[i], slope, intercept)
    ax.plot(np.unique(compartment_lengths), (slope * np.unique(compartment_lengths)) + intercept, ls="--", lw=2, color=default_colours[i])
    # ax.axhline(np.mean(alphas[:,i]), lw=3, ls="--", color=default_colours[i])
ax.tick_params(axis="x", labelsize=30, which="major", pad=12, direction="in", length=10, width=1.2)
ax.tick_params(axis="y", labelsize=30, which="major", pad=12, direction="in", length=8, width=1.1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xlabel("$s$    (kb)", fontsize=32, labelpad=16)
ax.set_ylabel("$\\alpha$", fontsize=32, labelpad=16)
ax.legend(fontsize="25", frameon=False, borderpad=1, ncols=1)
fig.savefig("./results/latest/state-open/psd_alphas_selective_by-compartment-size.svg", bbox_inches="tight")
plt.close(fig)

markers = ["o", "X", "s"]
fig, ax = plt.subplots()
fig.set_figwidth(15)
fig.set_figheight(10)
ax.set_ylabel("$\\alpha$", fontsize=32, labelpad=16)
fig.set_layout_engine("tight")
ax.violinplot(alphas, positions=range(len(epsilons)), showmeans=True, showextrema=True)
for i in range(len(epsilons)):
    ax.scatter([i] * len(alphas[:,i]), alphas[:,i], s=60, color=default_colours[0], edgecolor="#fff")
ax.tick_params(axis="x", labelsize=30, which="major", pad=12, direction="in", length=10, width=1.2)
ax.tick_params(axis="y", labelsize=30, which="major", pad=12, direction="in", length=8, width=1.1)
ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
ax.set_xticks(range(len(epsilons)), labels=labels)
fig.savefig("./results/latest/state-open/psd_alphas_selective_by-model.svg", bbox_inches="tight")
plt.close(fig)