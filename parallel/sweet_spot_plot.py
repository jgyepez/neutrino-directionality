#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: sweet_spot_plot.py
Author: Jeffrey G. Yepez
Date: 26 Feb 2026
Description: Works with the angular uncertainty output of 'main.py' to produce an angular uncertainty plot varied in segment size. Performs a fit to approximate the ideal segment size of a segmented neutrino detector.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

save = False

plt.rc("font", family="serif", size = 18)
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

data_file = "data14/func_iter_n_300.txt"

dx = []
dt = []

with open(data_file, "r") as f:
    for line in f:
        elems = line.split()

        dx.append(float(elems[0]))
        dt.append(float(elems[1]))

    f.close()

err = np.array(dt) / np.sqrt(300)

params, covariance = curve_fit(cubic, dx, dt)

a, b, c, d = params
print(f"Fitted parameters: a={a:.8f}, b={b:.8f}, c={c:.8f}, c={d:.8f}")

# Find minimum.
alpha = 3 * a
beta = 2 * b
gamma = c

# Roots of the first derivative.
minimum_loc = (-beta + np.sqrt(beta**2 - 4*alpha*gamma)) / (2 * alpha)

avg_track_length = 69.98286774221765

ratio = minimum_loc / avg_track_length

print()
print(f"The ideal segment size calculated from fit: {minimum_loc:.2f}")
print(f"The average track length calculated from 1M event fiducial dataset: {avg_track_length:.2f}")
print(f"Their ratio is: {minimum_loc/avg_track_length:.2f}")
print()

# Generate smooth segment size values for plotting the fit.
x_fit = np.linspace(min(dx), max(dx), 200)
y_fit = cubic(x_fit, *params)

# Plot raw data over the fitted curve.
plt.errorbar(dx, dt, yerr=err, fmt=".", color="black", capsize=3, label="Data")
plt.plot(x_fit, y_fit, "r-", label=f"Polyfit min at {int(minimum_loc)} mm")

plt.xlabel("Segment size $\\Delta x$ (mm)")
plt.ylabel("Angular uncertainty $\\delta \\vartheta$ (${}^\\circ$)")
plt.legend()

if save:
    plt.savefig("sweet_spot.pdf", format="pdf", bbox_inches="tight")

plt.show()

exit()

        
