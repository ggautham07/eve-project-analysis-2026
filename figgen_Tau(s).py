import sys
from os import path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize


# Cost function for the fit
def cost(d, test_data, ref_data):
    return np.sum((np.abs(ref_data - (test_data + d))))
    # return np.mean(np.sqrt(np.abs(ref_data - (test_data + d))))


plt.rcParams['font.family'] = ["serif", "sans-serif"]
plt.rcParams['mathtext.fontset'] = "dejavuserif"
fd = {"fontfamily": "serif", "fontsize": 48}
default_colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
gregor_colours = ["#7ab200", "#009ba5", "#008f4c", "#0085b6", "#004e9e", "#00177d", "#00053d"]


raw_evec_data = pd.read_csv("./resources/processed_data/evec_region.csv", sep="\t")
system_length = (raw_evec_data["end"].to_list() - raw_evec_data["start"][0]) // 1000
MS2_pos = np.array([9969, 9985, 9969, 9985, 9969, 9985, 9985]) - (raw_evec_data["start"][0] // 1000)
parS_pos = np.array([10027, 9903, 10057, 9836, 10159, 9390, 6657]) - (raw_evec_data["start"][0] // 1000)
linear_interlocus_dists = abs(MS2_pos - parS_pos)
linear_interlocus_dists_log = np.log10(linear_interlocus_dists)
print(linear_interlocus_dists)

args = sys.argv
tag = args[1]
analyses_dir = f"./analyses/{tag}"

with h5py.File(path.join(analyses_dir, "analysis.h5"), mode="a") as handler:
    tau = handler["Tau(s)"][:] * 800    # to include 800 as the length of each t.u. in time steps

tau_exp = [39.53656951044747103,
           72.07541654553817523,
           68.56142461161556412,
           100.3457588639866600,
           128.6890127669227297,
           190.2885358950472892,
           742.8575441880935841]

optim_results = []
sigma = []
for _ in range(10):
    init_guess = np.random.randint(low=-1, high=1, size=1)
    result = minimize(fun=cost, x0=init_guess, args=(np.log10(tau_exp), np.log10(tau),), method="L-BFGS-B")
    optim_results.append(result)
    sigma.append(result["fun"])
result = optim_results[np.argmin(sigma)]
d_fit = np.squeeze(result["x"])
sigma = result["fun"]
print(d_fit, sigma)

slope1, intercept1 = np.polyfit(linear_interlocus_dists_log, np.log10(tau_exp), deg=1)
slope2, intercept2 = np.polyfit(linear_interlocus_dists_log, np.log10(tau), deg=1)

# fig, ax = plt.subplots()
# fig.set_figwidth(12)
# fig.set_figheight(10)
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlabel("$s$    (kb)", fontdict=fd, labelpad=24)
# ax.set_ylabel("$\\tau$    ($\mathregular{t.u.}$)", fontdict=fd, labelpad=24)
# ax.tick_params(axis="x", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
# ax.tick_params(axis="x", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
# ax.tick_params(axis="y", labelsize=36, which="major", pad=18, direction="in", length=8, width=1.1)
# ax.tick_params(axis="y", labelsize=28, which="minor", pad=18, direction="in", length=4, width=1)
# ax.tick_params(which="both", bottom=True, top=True, left=True, right=True)
# ax.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
# ax.scatter(linear_interlocus_dists, 10 ** (np.log10(tau_exp) + d_fit),
#            s=480, marker="o", color=default_colours[7], edgecolor="#000",
#            label=f"Experiment", zorder=10)
# ax.scatter(linear_interlocus_dists, 10 ** np.log10(tau),
#            s=300, marker="X", color=default_colours[1], edgecolor="#000",
#            label=f"Simulation", zorder=11)
# pt_l, pt_r = ax.get_xlim()
# pt_range = np.linspace(pt_l, pt_r, 20)
# ax.plot(pt_range, pt_range ** slope2 * (10 ** intercept2),
#         "--", linewidth=4, color=default_colours[1])
# ax.plot(pt_range, ((pt_range ** slope1) * (10 ** intercept1)) * (10 ** d_fit),
#         "--", linewidth=4, color=default_colours[7])
# ax.plot(pt_range, pt_range ** 2 * (10 ** (intercept2-1.5)),
#          linestyle=(0, (1, 1)), linewidth=3, color=default_colours[0]+"99", label="ideal chain")
# ax.plot(pt_range, pt_range ** (5 / 3) * (10 ** (intercept2-1.5)),
#          linestyle=(0, (3, 1, 1, 1)), linewidth=3, color=default_colours[2]+"99", label="crumpled chain")

# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], loc="upper left", fontsize="34", frameon=False, borderpad=0.5, ncols=1)
# plt.tight_layout()
# fig.savefig(path.join(analyses_dir, "Tau(s).svg"), format="svg")

print(f"Scaling exponent from experiment: {slope1:.3f}")
print(f"Scaling exponent from simulation: {slope2:.3f}")