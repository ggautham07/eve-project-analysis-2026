import numpy as np
from polychrom.hdf5_format import load_hdf5_file
from h5py import File
import matplotlib.pyplot as plt
from os import path
import sys
import json

def generate_bins(N, start=1, bins_per_order_magn=20):
    lstart = np.log10(start)
    lend = np.log10(N - 1) + 1e-6
    num = int(np.ceil((lend - lstart) * bins_per_order_magn))
    bins = np.unique(np.logspace(lstart, lend, dtype=int, num=max(num, 0)))
    if len(bins) > 0:
        assert bins[-1] == N - 1
    return bins

linear_interlocus_dists = np.array([58, 82, 88, 149, 190, 595, 3327])

args = sys.argv
sim_dir = args[1]
tag = path.basename(sim_dir)

try:
    coordinates = load_hdf5_file(path.join(sim_dir, "selected_monomer_coordinates.h5"))["trajectory"]
except FileNotFoundError:
    coordinates = load_hdf5_file(path.join(sim_dir, "E-P_trajectory.h5"))["trajectory"]
num_rows, num_cols, _ = coordinates[0].shape

params = json.load(open(path.join(sim_dir, "parameters.json"), mode="r"))
N = params["polymer_length"]
moldyn_steps = params["moldyn_steps"]
bc = False
le = False
if "interaction_matrix" in list(params.keys()):
    eps = params["interaction_matrix"][0][0]
    bc = True
if path.exists(path.join(sim_dir, "parameters_loop_extrusion.json")):
    params = json.load(open(path.join(sim_dir, "parameters_loop_extrusion.json"), mode="r"))
    unload_prob = params["LEF_unload_prob"]
    load_prob = params["LEF_load_prob"]
    step_prob = params["LEF_step_prob"]
    le = True

eq_time = int(args[2])
print(f"Equilibriation time supplied: {eq_time:.2E}")
print(f"Number of timesteps between each sampled conformation: {moldyn_steps:.2E}")
coordinates = coordinates[(eq_time // moldyn_steps):]
print(f"Discarded first {eq_time // moldyn_steps} conformations")
trajectory_length = coordinates.shape[0]
dt_max = trajectory_length // 100
dt_range = generate_bins(dt_max + 1, bins_per_order_magn=20)
num_frames = trajectory_length - dt_max
mean_squared_disp_1loc = np.empty((len(dt_range), num_rows, num_cols))
mean_squared_disp_2loc = np.empty((len(dt_range), num_rows))
# Non-overlapping windows
for i, dt in enumerate(dt_range):
    # Single-locus MSD for the enhancer and promoter site
    disp_vector_1loc_dt = coordinates[dt:num_frames+dt:dt] - coordinates[0:num_frames:dt]
    mean_squared_disp_1loc[i] = np.mean(np.square(np.linalg.norm(disp_vector_1loc_dt, axis=-1)), axis=0)
    # Two-locus MSD for the distance vector between enhancer and promoter site
    disp_vector_2loc_curr = coordinates[0:num_frames:dt, :, 1, :] - coordinates[0:num_frames:dt, :, 0, :]
    disp_vector_2loc_next = coordinates[dt:num_frames+dt:dt, :, 1, :] - coordinates[dt:num_frames+dt:dt, :, 0, :]
    disp_vector_2loc_dt = disp_vector_2loc_next - disp_vector_2loc_curr
    mean_squared_disp_2loc[i] = np.mean(np.sum(np.square(disp_vector_2loc_dt), axis=-1), axis=0)

dt_range = dt_range * moldyn_steps
with File(path.join(f"./analyses/{tag}", "analysis.h5"), mode="a") as handler:
    if "time" in handler.keys():
        del handler["time"]
    handler.create_dataset(name="time", data=dt_range[:])
    if "M1(t)" in handler.keys():
        del handler["M1(t)"]
    handler.create_dataset(name="M1(t)", data=mean_squared_disp_1loc[:])
    if "M2(t)" in handler.keys():
        del handler["M2(t)"]
    handler.create_dataset(name="M2(t)", data=mean_squared_disp_2loc[:])

print("Saved computed single- and two-locus mean-squared displacements")


# # raw_data = load_hdf5_file(path.join(sim_dir, "computed_MSDs.h5"))
# # dt_range = raw_data["time_lags"]
# # mean_squared_disp_1loc = raw_data["single_locus_MSD"]
# # mean_squared_disp_2loc = raw_data["two_locus_MSD"]

# print("Plotting the computed values")

# dt_range_log = np.log10(dt_range)
# mean_squared_disp_1loc_log = np.log10(mean_squared_disp_1loc)
# mean_squared_disp_2loc_log = np.log10(mean_squared_disp_2loc)

# default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
# gregor_colours = ["#7ab20090", "#008f4c90", "#009ba590", "#0085b690", "#004e9e90", "#00177d90", "#00053d90"]
# plt.rcParams['font.family'] = ["serif", "sans-serif"]
# plt.rcParams['mathtext.fontset'] = "dejavuserif"
# fd = {"fontfamily": "serif", "fontsize": 16}

# start = 2
# end = 8
# beta, intercept = np.polyfit(dt_range_log[start:end], mean_squared_disp_1loc_log[start:end], deg=1)
# beta = np.mean(beta)
# intercept = np.mean(intercept)
# plt.figure(figsize=(9, 7))
# plt.xscale("log")
# plt.yscale("log")
# for i, c in zip(range(2), [1, -2]):
#     plt.plot(dt_range, mean_squared_disp_1loc[:, i], "-o", color=gregor_colours[c], linewidth=3, label="enhancer" if i == 0 else "promoter")
# plt.plot(dt_range[start:end+10], (dt_range[start:end+10] ** beta * (10 ** (intercept + 0.25))), "--", linewidth=1, color="#010101", label=f"$\\beta = {beta:.2f}$")
# title = "Single-locus MSD"
# if not bc and le:
#     title = title + f"\nLoop extrusion, $N = {N / 1000:.1f} \\text{{Mb}}$, $p_U = {unload_prob}$, $p_L = {load_prob}$, $p_S = {step_prob}$"
# elif bc and not le:
#     title = title +  f"\nBlock copolymer, $N={N / 1000:.1f} \\text{{Mb}}, \\epsilon={eps:.3f}$"
# elif bc and le:
#     title = title + f"\nBlock copolymer + loop extrusion, $N = {N / 1000:.1f} \\text{{Mb}}$\n\\epsilon={eps:.3f}, $p_U = {unload_prob}$, $p_L = {load_prob}$"
# else:
#     title = title + f"\nRouse model, $N={N / 1000:.1f} \\text{{Mb}}$"
# plt.title(title, fontsize=18)
# plt.xlabel("time lag", fontdict=fd)
# plt.ylabel("mean-squared displacement", fontdict=fd)
# plt.legend(fontsize=16)
# plt.savefig(path.join(sim_dir, "M1(t).svg"), format="svg")
# plt.close()

# start = 1
# end = 5
# beta, intercept = np.polyfit(dt_range_log[start:end], mean_squared_disp_1loc_log[start:end], deg=1)
# beta = np.mean(beta)
# intercept = np.mean(intercept)
# plt.figure(figsize=(9, 7))
# plt.xscale("log")
# plt.yscale("log")
# for i in range(len(linear_interlocus_dists)):
#     plt.plot(dt_range, mean_squared_disp_2loc[:, i], "-o", color=gregor_colours[i], linewidth=3, label=f"$d = {linear_interlocus_dists[i]}$ kb")
# plt.plot(dt_range[start:end+10], (dt_range[start:end+10] ** beta) * (10 ** (intercept + 0.5)), "--", linewidth=1, color="#010101", label=f"$\\beta = {beta:.2f}$")
# title = "Two-locus MSD"
# if not bc and le:
#     title = title + f"\nLoop extrusion, $N = {N / 1000:.1f} \\text{{Mb}}$, $p_U = {unload_prob}$, $p_L = {load_prob}$, $p_S = {step_prob}$"
# elif bc and not le:
#     title = title +  f"\nBlock copolymer, $N={N / 1000:.1f} \\text{{Mb}}, \\epsilon={eps:.3f}$"
# elif bc and le:
#     title = title + f"\nBlock copolymer + loop extrusion, $N = {N / 1000:.1f} \\text{{Mb}}$\n\\epsilon={eps:.3f}, $p_U = {unload_prob}$, $p_L = {load_prob}$"
# else:
#     title = title + f"\nRouse model, $N={N / 1000:.1f} \\text{{Mb}}$"
# plt.title(title, fontsize=18)
# plt.xlabel("time lag", fontdict=fd)
# plt.ylabel("mean-squared displacement", fontdict=fd)
# plt.legend(fontsize=16)
# plt.savefig(path.join(sim_dir, "M2(t).svg"), format="svg")
# plt.close()