from time import perf_counter
import numpy as np
from polychrom.hdf5_format import list_URIs, load_URI
from h5py import File
# from polychrom.polymer_analyses import Rg2
import matplotlib.pyplot as plt
import sys
from itertools import chain
import pandas as pd
from os import path


def Rg2(config):
    assert config.ndim == 2
    assert config.shape[1] == 3
    return np.mean(np.linalg.norm(config - np.mean(config, axis=0), axis=-1) ** 2)


# Getting the monomer compartment blocks
monomer_classes_file_path = "./resources/processed_data/evec_region.csv"
raw_evec_data = pd.read_csv(monomer_classes_file_path, sep="\t")
compartments = (raw_evec_data["E1"].to_numpy() >= 0).astype(int)
monomer_classes = np.asarray(list(chain.from_iterable([[compartments[i] for _ in range(16)] \
                                                       for i in range(len(compartments))])))

compartment_switch_pos = np.nonzero(monomer_classes[1:] - monomer_classes[:-1])[0]
compartment_switch_pos = np.concatenate([[0], compartment_switch_pos])
compartment_lengths = compartment_switch_pos[1:] - compartment_switch_pos[:-1]

args = sys.argv
sim_dir = args[1]
try:
    start = int(args[2])
except IndexError:
    start = 1000
try:
    step = int(args[3])
except IndexError:
    step = 100

try:
    st = perf_counter()
    # Extracting trajectories
    URIs = list_URIs(sim_dir)[start::step]     # for eve locus simulations
    sample = load_URI(URIs[0])["pos"]
    N = sample.shape[0]
    Z = len(URIs)
    # For one axis, ML's method
    trajectory = np.empty((Z, N, 3))
    for z in range(Z):
        data = load_URI(URIs[z])["pos"]
        trajectory[z,:] = data[:]
    print(f"Successfully loaded trajectory data in {perf_counter() - st:.2f} seconds")
except Exception as err:
    raise err

Rg2_by_block_all = []
for m in range(len(compartment_lengths)):
    comp = 0 if m % 2 == 0 else 1
    length = compartment_lengths[m]
    if length >= 250:
        idxs = (compartment_switch_pos[m], compartment_switch_pos[m+1])
        print(f"Computing for C{comp} of length {length}; indices {idxs}")
        Rg2_by_block = np.zeros(len(URIs))
        for z in range(Z):
            Rg2_by_block[z] = Rg2(trajectory[z, idxs[0]:idxs[1], :])
        Rg2_by_block_all.append(Rg2_by_block)

Rg2_by_block_all = np.asarray(Rg2_by_block_all)
# print(Rg2_by_block_all.shape)

np.save(file=f"./results/latest/sims/{path.basename(sim_dir)}/Rg2_compartment_blocks.npy", arr=Rg2_by_block_all, allow_pickle=False)

print(f"Mean Rg2 by compartment block was", *np.mean(Rg2_by_block_all, axis=1), sep="\n")