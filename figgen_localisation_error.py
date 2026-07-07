from os import path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from polychrom.hdf5_format import load_hdf5_file

genomic_dists = np.array([58, 82, 88, 149, 190, 595, 3327])

# Taking only best-fit simulation cases
sim_dirs = ["./trajectories/00_Rouse",
            "./trajectories/01_Rouse_excvol__loopex_unload=0.01_load=0.48",
            "./trajectories/02_block_copolymer_eps=0.290", 
            "./trajectories/03_block_copolymer_eps=0.285__loopex_unload=0.01_load=0.32",]

d_fits = [-1.7484057550487109,
          -1.594253793200877,
          -1.7021766289825127,
          -1.727653495652224,]

tags = ["Rouse",
        "loopex",
        "block_copolymer",
        "block_copolymer_with_loopex",]

for sim_dir, d_fit, tag in zip(sim_dirs, d_fits, tags):

    # From Chen et al., verified with JB
    mean_loc_error = np.round(10 ** (np.log10(180) + d_fit), 2)    # 180 nm

    start = 4000
    step = 100

    # Load trajectory
    coordinates = load_hdf5_file(path.join(sim_dir, "E-P_trajectory.h5"))["trajectory"]

    # Compute R with and without localisation error
    R = np.linalg.norm(coordinates[..., 1, :] - coordinates[..., 0, :], axis=-1)
    R_ = np.mean(R, axis=0)
    R_with_error = np.random.normal(loc=R, scale=mean_loc_error, size=R.shape)
    R_with_error = np.mean(R_with_error, axis=0)

    # Compute respective scaling exponents
    slope1, intercept1 = np.polyfit(np.log10(genomic_dists[:5]), np.log10(R_[:5]), deg=1)
    slope2, intercept2 = np.polyfit(np.log10(genomic_dists[:5]), np.log10(R_with_error[:5]), deg=1)

    # Parameters for plotting
    default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]
    plt.rcParams['font.family'] = ["serif", "sans-serif"]
    plt.rcParams['mathtext.fontset'] = "dejavuserif"
    fd = {"fontfamily": "serif", "fontsize": 40}

    # Plot a manuscript-ready figure and save it
    fig, ax = plt.subplots()
    fig.set_figwidth(12)
    fig.set_figheight(10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.scatter(genomic_dists, R_, s=480, marker="o", color=default_colours[7], edgecolor="#000", label=f"Original")
    ax.scatter(genomic_dists, R_with_error, s=300, marker="X", color=default_colours[1], edgecolor="#000", label=f"With localization error")
    ax.plot(genomic_dists, genomic_dists ** slope1 * (10 ** intercept1), ":", linewidth=3.5, color=default_colours[1])
    ax.plot(genomic_dists, genomic_dists ** slope2 * (10 ** intercept2), ":", linewidth=3.5, color=default_colours[7])
    ax.set_xlabel("$s$   (kb)", fontdict=fd, labelpad=24)
    ax.set_ylabel("$\\langle R \\rangle$   ($\mathregular{s.u.}$)", fontdict=fd, labelpad=24)
    ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
    ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
    ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
    ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
    ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
    ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize="30", frameon=False, borderpad=1)
    plt.tight_layout()
    fig.savefig(f"./results/open/{tag}_R(s)_localisation-error.svg", format="svg")
    plt.close()