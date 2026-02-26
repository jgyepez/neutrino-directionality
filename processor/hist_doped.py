#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: hist_doped.py
Author: Jeffrey G. Yepez
Date: 25 Feb 2026
Description: Plots histograms for neutron capture and positron annihilation track length for two different doping levels, 0.1% and 0.5%.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

save = False

plt.rc("font", family="serif", size = 16)
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"

# 0.1% doped data file.
datafile001wt = "data/10k_001wt_unfiltered.json"

# 0.5% doped data file.
datafile005wt = "data/10k_005wt_unfiltered.json"
        
with open(datafile001wt, "r") as f:
    data001wt = json.load(f)
    f.close()
    
with open(datafile005wt, "r") as f:
    data005wt = json.load(f)
    f.close()

# First the positrons.
d001wt_track_lengths = []
for event in list(data001wt.keys()):

    # Pulls coordinates from data dictionary.
    x0, y0, z0 = data001wt[event]["vertex"]
    x1, y1, z1 = data001wt[event]["annihilation"]
    
    xbar = x1 - x0
    ybar = y1 - y0
    zbar = z1 - z0

    track_length = np.sqrt(xbar**2 + ybar**2 + zbar**2)

    d001wt_track_lengths.append(track_length)

mean001wt = sum(d001wt_track_lengths) / len(d001wt_track_lengths)
print(f"Mean positron track length for 0.1% doped:", mean001wt, "mm")

d005wt_track_lengths = []
for event in list(data005wt.keys()):

    # Pulls coordinates from data dictionary.
    x0, y0, z0 = data005wt[event]["vertex"]
    x1, y1, z1 = data005wt[event]["annihilation"]
    
    xbar = x1 - x0
    ybar = y1 - y0
    zbar = z1 - z0

    track_length = np.sqrt(xbar**2 + ybar**2 + zbar**2)

    d005wt_track_lengths.append(track_length)

mean005wt = sum(d005wt_track_lengths) / len(d005wt_track_lengths)
print(f"Mean positron track length for 0.5% doped:", mean005wt, "mm")

bin_width = 1.0
nbins = 100
bins = []
for i in range(nbins+1):
    bins.append(bin_width * i)

hist1 = plt.hist(d001wt_track_lengths, bins=bins, edgecolor=(1,0,0,1.0), fc=(1,0,0,0.05), label="0.1%, $\\overline{d}_{+,0.1\%}$:" + f" {round(mean001wt, 2)} mm")
hist2 = plt.hist(d005wt_track_lengths, bins=bins, edgecolor=(0,0,1,0.8), fc=(1,1,1,0), label="0.5%, $\\overline{d}_{+,0.5\%}$:" + f" {round(mean005wt, 2)} mm")

plt.legend()
plt.grid()

plt.xlim(0, 50)
plt.xlabel("$d_+$ (mm)")
plt.ylabel("Counts")
plt.title("Positron path length distribution ($d_+$, 10k events)")

if save:
    plt.savefig("positron_track_length_hist.pdf", format="pdf")

plt.show()

# Now the neutrons case.
d001wt_track_lengths = []
for event in list(data001wt.keys()):

    # Pulls coordinates from data dictionary.
    x0, y0, z0 = data001wt[event]["vertex"]
    x1, y1, z1 = data001wt[event]["capture"]
    
    xbar = x1 - x0
    ybar = y1 - y0
    zbar = z1 - z0

    track_length = np.sqrt(xbar**2 + ybar**2 + zbar**2)

    d001wt_track_lengths.append(track_length)

mean001wt = sum(d001wt_track_lengths) / len(d001wt_track_lengths)
print(f"Mean neutron track length for 0.1% doped:", mean001wt, "mm")

d005wt_track_lengths = []
for event in list(data005wt.keys()):

    # Pulls coordinates from data dictionary.
    x0, y0, z0 = data005wt[event]["vertex"]
    x1, y1, z1 = data005wt[event]["capture"]
    
    xbar = x1 - x0
    ybar = y1 - y0
    zbar = z1 - z0

    track_length = np.sqrt(xbar**2 + ybar**2 + zbar**2)

    d005wt_track_lengths.append(track_length)

mean005wt = sum(d005wt_track_lengths) / len(d005wt_track_lengths)
print(f"Mean neutron track length for 0.5% doped:", mean005wt, "mm")

bin_width = 5.0
nbins = 100
bins = []
for i in range(nbins+1):
    bins.append(bin_width * i)

hist1 = plt.hist(d001wt_track_lengths, bins=bins, edgecolor=(1,0,0,1.0), fc=(1,0,0,0.05), label="0.1%, $\\overline{d}_{n,0.1\%}$:" + f" {round(mean001wt, 1)} mm")
hist2 = plt.hist(d005wt_track_lengths, bins=bins, edgecolor=(0,0,1,0.8), fc=(1,1,1,0), label="0.5%, $\\overline{d}_{n,0.5\%}$:" + f" {round(mean005wt, 1)} mm")

plt.legend()
plt.grid()

plt.xlim(0, 250)
plt.xlabel("$d_n$ (mm)")
plt.ylabel("Counts")
plt.title("Neutron path length distribution ($d_n$, 10k events)")

if save:
    plt.savefig("neutron_track_length_hist.pdf", format="pdf")

plt.show()

exit()
