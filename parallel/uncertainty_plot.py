#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: uncertainty_plot.py
Author: Jeffrey G. Yepez
Date: 26 Feb 2026
Description: Works with the angular uncertainty output of 'main.py' to produce an angular uncertainty plot varied in counts. Performs an arctangent-based fit to the uncertainty data.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

save = False
center = True

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

# Chosen model to fit to the uncertainty data for error bar calculation.
def model_arctan_4p(n, a, b, c, d):
    return a - d * np.arctan(np.log10((n + b)) / c)

# Detected data presented in the paper.
dataf_5 = "data30/func_iter_dx_5_detected.txt"
dataf_50 = "data30/func_iter_dx_50_detected.txt"
dataf_150 = "data30/func_iter_dx_150_detected.txt"

# Higher resolution detected data up to 10^6 events.
#dataf_5 = "data33/func_iter_dx_5_detected.txt"
#dataf_50 = "data33/func_iter_dx_50_detected.txt"
#dataf_150 = "data33/func_iter_dx_150_detected.txt"

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
plt.errorbar(dx_5, dt_5, yerr=de_5, fmt="r.--", capsize=3)
plt.errorbar(dx_50, dt_50, yerr=de_50, fmt="g.--", capsize=3)
plt.errorbar(dx_150, dt_150, yerr=de_150, fmt="b.--", capsize=3)

# Reassign data arrays.
x = np.array(dx_5)
y = np.array(dt_5)
yerr = np.array(de_5)

x2 = np.array(dx_50)
y2 = np.array(dt_50)
yerr2 = np.array(de_50)

x3 = np.array(dx_150)
y3 = np.array(dt_150)
yerr3 = np.array(de_150)

# Transform to log space.
y_log = np.log10(y)
y2_log = np.log10(y2)
y3_log = np.log10(y3)

# Propagate linear errors into log space for weighted fit.
yerr_log = yerr / (y * np.log(10))
y2err_log = yerr2 / (y2 * np.log(10))
y3err_log = yerr3 / (y3 * np.log(10))

# Weighted fits in log space.
popt, pcov = curve_fit(
    model_arctan_4p,
    x,
    y_log,
    p0=[1,1,1,1],
    sigma=yerr_log,
    absolute_sigma=True
)

popt2, pcov2 = curve_fit(
    model_arctan_4p,
    x2,
    y2_log,
    p0=[1,1,1,1],
    sigma=y2err_log,
    absolute_sigma=True
)

popt3, pcov3 = curve_fit(
    model_arctan_4p,
    x3,
    y3_log,
    p0=[1,1,1,1],
    sigma=y3err_log,
    absolute_sigma=True
)

# Best-fit parameters.
a_fit, b_fit, c_fit, d_fit = popt
a2_fit, b2_fit, c2_fit, d2_fit = popt2
a3_fit, b3_fit, c3_fit, d3_fit = popt3

# Parameter uncertainties.
perr = np.sqrt(np.diag(pcov))
p2err = np.sqrt(np.diag(pcov2))
p3err = np.sqrt(np.diag(pcov3))

print("Fit parameters (5 mm):")
print(f"a = {a_fit} ± {perr[0]}")
print(f"b = {b_fit} ± {perr[1]}")
print(f"c = {c_fit} ± {perr[2]}")
print(f"d = {d_fit} ± {perr[3]}")

print("Fit parameters (50 mm):")
print(f"a = {a2_fit} ± {p2err[0]}")
print(f"b = {b2_fit} ± {p2err[1]}")
print(f"c = {c2_fit} ± {p2err[2]}")
print(f"d = {d2_fit} ± {p2err[3]}")

print("Fit parameters (150 mm):")
print(f"a = {a3_fit} ± {p3err[0]}")
print(f"b = {b3_fit} ± {p3err[1]}")
print(f"c = {c3_fit} ± {p3err[2]}")
print(f"d = {d3_fit} ± {p3err[3]}")

x_fit = np.logspace(np.log10(dx_5[0]), np.log10(dx_5[-1]), 128)
x2_fit = np.logspace(np.log10(dx_50[0]), np.log10(dx_50[-1]), 128)
x3_fit = np.logspace(np.log10(dx_150[0]), np.log10(dx_150[-1]), 128)

y_fit = 10**model_arctan_4p(x_fit, a_fit, b_fit, c_fit, d_fit)
y2_fit = 10**model_arctan_4p(x2_fit, a2_fit, b2_fit, c2_fit, d2_fit)
y3_fit = 10**model_arctan_4p(x3_fit, a3_fit, b3_fit, c3_fit, d3_fit)

# Plot the data oer the fit functions.
plt.plot(x_fit, y_fit, "r-", lw=2, label="$\\Delta x = 5 \\,\\mathrm{mm}$")
plt.plot(x2_fit, y2_fit, "g-", lw=2, label="$\\Delta x = 50 \\,\\mathrm{mm}$")
plt.plot(x3_fit, y3_fit, "b-", lw=2, label="$\\Delta x = 150 \\,\\mathrm{mm}$")

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
