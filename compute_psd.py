import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from polychrom.hdf5_format import load_URI, list_URIs
from os import path
from glob import glob

plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 16}

def dct2(data):
    """
    Apply the Discrete Cosine Transform to a set of polymer conformations

    From Michael Liefsons
    """
    from scipy.fft import dct
    N = data.shape[1]
    inv_norm = 1 / (2 * N)
    return inv_norm * dct(data, axis=1)

sim_dir_all = ["./trajectories/00_Rouse/"]
sim_dirs_eps = glob("./trajectories/02*")
sim_dirs_eps.sort()
epsilons = [sim_dirs_eps[i].split("=")[-1] for i in range(len(sim_dirs_eps))]
print(epsilons)
sim_dir_all.extend(sim_dirs_eps)

alphas_low = []
alphas_high = []

for sim_dir in sim_dir_all:

    print(sim_dir)
    URIs = list_URIs(sim_dir)[20000::1000] if "Rouse" in sim_dir else list_URIs(sim_dir)[10000::500]
    sample = load_URI(URIs[0])["pos"]
    N = sample.shape[0]
    Z = len(URIs)

    # For one axis, ML's method
    configs = np.empty((Z, N, 3))
    for z in range(Z):
        data = load_URI(URIs[z])["pos"]
        configs[z,:] = data[:]
    psd = np.square(np.abs(dct2(configs))).mean(axis=0).sum(axis=-1)

    slope0, intercept0 = np.polyfit(np.log(np.arange(1, 4)), np.log(psd)[1:4], deg=1)
    slope, intercept = np.polyfit(np.log(np.arange(100, (N//2)-1)), np.log(psd)[100:(N//2)-1], deg=1)

    alphas_low.append(slope0)
    alphas_high.append(slope)

    print(slope0, slope)

    plt.figure(figsize=(7, 6))
    plt.plot(np.log(np.arange(1, N//2)), np.log(psd)[1:N//2], "-o", color="blue");
    plt.plot(np.log(np.arange(1, 4)), np.log(np.arange(1, 4)) * slope0 + intercept0 * 1.2, "--", color="black", label=f"${slope0:.2f}$")
    plt.plot(np.log(np.arange(100, N//2)), np.log(np.arange(100, N//2)) * slope + intercept * 1.2, "--", color="blue", label=f"${slope:.2f}$")
    plt.xlabel("$ln(p)$", fontdict=fd)
    plt.ylabel("$ln(X^2_p)$", fontdict=fd)
    plt.legend()
    plt.savefig(path.join(sim_dir, "psd.svg"), format="svg")
    plt.close()

plt.figure(figsize=(6, 6))
plt.plot(epsilons, alphas_low[1:], "-o", color="blue");
plt.axhline(y=alphas_high[0], ls=":", color="black", label=f"Ideal, $\\alpha = {alphas_high[0]:.2f}$")
plt.xticks(epsilons[::3], labels=epsilons[::3])
plt.xlabel("$\\epsilon$", fontdict=fd)
plt.ylabel("$\\alpha$", fontdict=fd)
plt.grid(lw=0.15)
plt.legend()
plt.savefig("./results/open/psd_alpha_eps_block_copolymers.svg", format="svg")
plt.close()

sim_dirs_all = sorted(glob("./trajectories/02*")[:])
cmap = mpl.colormaps["winter"]
colours = cmap(np.linspace(0, 1, len(sim_dirs_all)))

epsilons = [sim_dirs_eps[i].split("=")[-1] for i in range(len(sim_dirs_eps))]
print(epsilons)

plt.figure(figsize=(8, 8))

for i, sim_dir in enumerate(sim_dirs_all):

    print(sim_dir)
    URIs = list_URIs(sim_dir)[10000::500]
    sample = load_URI(URIs[0])["pos"]
    N = sample.shape[0]
    Z = len(URIs)

    # For one axis, ML's method
    configs = np.empty((Z, N, 3))
    for z in range(Z):
        data = load_URI(URIs[z])["pos"]
        configs[z,:] = data[:]
    psd = np.square(np.abs(dct2(configs))).mean(axis=0).sum(axis=-1)

    slope0, intercept0 = np.polyfit(np.log(np.arange(1, 4)), np.log(psd)[1:4], deg=1)
    slope, intercept = np.polyfit(np.log(np.arange(100, (N//2)-1)), np.log(psd)[100:(N//2)-1], deg=1)

    print(slope0, slope)
    
    plt.plot(np.log(np.arange(1, N//2)), np.log(psd)[1:N//2], "-o", color=colours[i]);
    if i == 0:
        plt.plot(np.log(np.arange(1, 4)), np.log(np.arange(1, 4)) * slope0 + intercept0 * 1.1, ":", color="black", label=f"${slope0:.2f}$")
        plt.plot(np.log(np.arange(100, N//2)), np.log(np.arange(100, N//2)) * slope + intercept * 1.1, "--", color="black", label=f"${slope:.2f}$")
    if i == len(sim_dirs_all) - 1:
        plt.plot(np.log(np.arange(1, 4)), np.log(np.arange(1, 4)) * slope0 + intercept0 * 0.5, ":", color="black", label=f"${slope0:.2f}$")


plt.xlabel("$ln(p)$", fontdict=fd)
plt.ylabel("$ln(X^2_p)$", fontdict=fd)
plt.legend()
plt.savefig("./results/open/psd_all_eps_block_copolymers.svg")
plt.close()