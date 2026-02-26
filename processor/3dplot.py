#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: 3dplot.py
Author: Jeffrey G. Yepez
Date: 25 Feb 2026
Description: Tests Matplotlib's 3D plotting features.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rc("font", family="serif", size = 16)
plt.rcParams["text.usetex"] = True
plt.rcParams["mathtext.fontset"] = "cm"

# Fixing random state for reproducibility.
np.random.seed(19680801)

def randrange(n, vmin, vmax):
    #Helper function to make an array of random numbers having shape (n, ) with each number distributed Uniform(vmin, vmax).
    return (vmax - vmin) * np.random.rand(n) + vmin

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

n = 100

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zlow, zhigh)
    ax.scatter(xs, ys, zs, marker=m)

ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.set_zlabel("$z$")

plt.show()

print(xs,ys,zs)
print(len(xs),len(ys),len(zs))

exit()
