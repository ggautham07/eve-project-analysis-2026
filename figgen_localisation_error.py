import matplotlib.pyplot as plt
import numpy as np
import h5py
from os import path


# 00_Rouse  -1.7549776801395776
# 01_Rouse_excvol__loopex_unload=0.01_load=0.48    -1.5930252595751395 
# 02_block_copolymer_eps=0.290  -1.7021766289825127
# 03_block_copolymer_eps=0.290__loopex_unload=0.01_load=0.16    -1.709346216860327


genomic_dists = np.array([58, 82, 88, 149, 190, 595, 3327])

sim_dirs = ["./trajectories/00_Rouse",
            "./trajectories/01_Rouse_excvol__loopex_unload=0.01_load=0.48",
            "./trajectories/02_block_copolymer_eps=0.290", 
            "./trajectories/03_block_copolymer_eps=0.290__loopex_unload=0.01_load=0.16",]

d_fits = [-1.7549776801395776,
          -1.5930252595751395,
          -1.7021766289825127,
          -1.709346216860327,]

tags = ["Rouse",
        "loopex",
        "block_copolymer",
        "block_copolymer_with_loopex",]

for sim_dir, d_fit, tag in zip(sim_dirs, d_fits, tags):

    mean_loc_error = np.round(10 ** (np.log10(180) + d_fit), 2)    # 180 nm

    start = 4000
    step = 100

    with h5py.File(path.join(sim_dir, "selected_monomer_coordinates.h5"), "r") as handler:
        coordinates_raw = {x:y[:] for x, y in handler.items() if x!= "monomers"}
    time = [int(t) for t in coordinates_raw.keys()]
    time.sort()
    time = time[start::step]
    trajectory_length = len(time)
    num_sets, num_loci, _ = coordinates_raw["0"].shape
    coordinates = np.empty((trajectory_length, num_sets, num_loci, 3))
    for i, t in enumerate(time):
        coordinates[i] = coordinates_raw[str(t)]

    R = np.linalg.norm(coordinates[..., 1, :] - coordinates[..., 0, :], axis=-1)
    R_ = np.mean(R, axis=0)
    R_with_error = np.random.normal(loc=R, scale=mean_loc_error, size=R.shape)
    R_with_error = np.mean(R_with_error, axis=0)

    slope1, intercept1 = np.polyfit(np.log10(genomic_dists[:5]), np.log10(R_[:5]), deg=1)
    slope2, intercept2 = np.polyfit(np.log10(genomic_dists[:5]), np.log10(R_with_error[:5]), deg=1)

    default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    gregor_colours = ["#7ab200", "#008f4c", "#009ba5", "#0085b6", "#004e9e", "#00177d", "#00053d"]
    plt.rcParams['font.family'] = ["serif", "sans-serif"]
    plt.rcParams['mathtext.fontset'] = "dejavuserif"
    fd = {"fontfamily": "serif", "fontsize": 40}

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