#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: usable_uncertainty_plot.py
Author: Jeffrey G. Yepez
Date: 26 Feb 2026
Description: Works with the angular uncertainty output of 'main.py' to produce a usable event angular uncertainty plot varied in counts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

save = False
center = False

plt.rc("font", family="serif", size = 16)
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def model(n, a, b):
    return a / np.sqrt(n + b)

def model_log(n, b, c):
    return np.log10(a) - c * np.log10(n + b)

def model_log_bounded(n, b, c):
    return np.log10(180) + c * np.log10(1 + b) - c * np.log10(n + b)

def model_arctan(n, a, b, c):
    return a - c * np.arctan(np.log10((n + b) / b))

def model_arctan_4p(n, a, b, c, d):
    return a - d * np.arctan(np.log10((n + b)) / c)

# Usable event data presented in the paper.
dataf_5 = "data32/func_iter_dx_5_usable.txt"
dataf_50 = "data32/func_iter_dx_50_usable.txt"
dataf_150 = "data32/func_iter_dx_150_usable.txt"

# Higher resolution usable event data up to 10^4 events.
#dataf_5 = "data34/func_iter_dx_5_usable.txt"
#dataf_50 = "data34/func_iter_dx_50_usable.txt"
#dataf_150 = "data34/func_iter_dx_150_usable.txt"

dx_5 = []
dt_5 = []
de_5 = []

dx_50 = []
dt_50 = []
de_50 = []

dx_150 = []
dt_150 = []
de_150 = []

with open(dataf_5, "r") as f:
    i = 0
    for line in f:
        elems = line.split()

        dx_5.append(float(elems[0]))
        dt_5.append(float(elems[1]))
        de_5.append(float(elems[2]))

        i += 1

    f.close()

with open(dataf_50, "r") as f:
    i = 0
    for line in f:
        elems = line.split()

        dx_50.append(float(elems[0]))
        dt_50.append(float(elems[1]))
        de_50.append(float(elems[2]))

        i += 1

    f.close()

with open(dataf_150, "r") as f:
    i = 0
    for line in f:
        elems = line.split()

        dx_150.append(float(elems[0]))
        dt_150.append(float(elems[1]))
        de_150.append(float(elems[2]))

        i += 1

    f.close()

# Removes first element because error bar is too large.
dx_5 = dx_5[1:]
dt_5 = dt_5[1:]
de_5 = de_5[1:]

dx_50 = dx_50[1:]
dt_50 = dt_50[1:]
de_50 = de_50[1:]

dx_150 = dx_150[1:]
dt_150 = dt_150[1:]
de_150 = de_150[1:]
    
# Plot raw data.
plt.errorbar(dx_5, dt_5, yerr=de_5, fmt="r.-", capsize=3, label="$\Delta x$ = 5 mm")
plt.errorbar(dx_50, dt_50, yerr=de_50, fmt="g.-", capsize=3, label="$\Delta x$ = 50 mm")
plt.errorbar(dx_150, dt_150, yerr=de_150, fmt="b.-", capsize=3, label="$\Delta x$ = 150 mm")

plt.xscale("log")
plt.yscale("log")

plt.xlabel(("C" if center else "Usable c") + "ounts $n$")
plt.ylabel("Angular uncertainty $\\delta \\vartheta$ (${}^\\circ$)")
plt.legend(framealpha=1.0)

plt.grid(True, which="both", linestyle="--", linewidth=0.5)

if save:
    plt.savefig("06_money_plot_fit_arctan_log_4p.pdf", format="pdf", bbox_inches="tight")

plt.show()

exit()
