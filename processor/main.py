#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File: main.py
Author: Jeffrey G. Yepez
Date: 25 Feb 2026
Description: RATPAC2 data processor. Post processing methods for RATPAC2 positron and neutron output track data. This code produces figures for the UH Manoa neutrino directionality research group by processing simulated data with RATPAC2.
"""

import os
import sys
import time
import random
import json
import math
from tqdm import tqdm
import copy

import pandas as pd

from scipy.stats import poisson
from scipy.special import gamma
from scipy.optimize import curve_fit
import scipy.integrate as spi

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.stats import multivariate_normal

class DataProcessor():
    def __init__(self):

        # Control booleans for debugging and formatting.
        self.debug = False
        self.latex = True
        
        if self.debug:
            print("DataProcessor class method: __init__")

        if self.latex:
            plt.rc("font", family="serif", size = 16)
            plt.rcParams["text.usetex"] = True
            plt.rcParams["mathtext.fontset"] = "cm"

        # Number of events to read and process from the RATPAC output. WARNING: No longer used to read data.
        self.N = 10000

        # Neutron file plus other track PDGs. This class converts and stores ASCII neutron data into pandas dataframe.
        self.dataFile = "/Users/gabriel/Downloads/ibd_cube_001wt_10k_run1/truth.txt"

        # Positron file if available. If not then set this variable to None.
        self.positronFile = "/Users/gabriel/Downloads/ibd_cube_001wt_10k_run1/positrons.txt"
        #self.positronFile = None

        # Initialization variables for 2d square grid segmentation of size (self.grid_size x self.grid_size). This variable represents the number of squares on a side of the grid. Must be an odd number so that there is a clearly defined center segment for prompt.
        self.grid_size = 9

        # Size of individual detector segment in mm.
        #self.cube_size = 5
        self.cube_size = 50
        #self.cube_size = 150

        # Set the kind of neutron event state to observe. Currently only 'capture', '1st-scatter', or '2nd-scatter' supported. Current routines that depend on self.kind are: neutronHistogram, angularDist, cloudPlot, dimDist
        #self.kind = "1st-scatter"
        #self.kind = "2nd-scatter"
        self.kind = "capture"

        # Reads ASCII data and store in memory as pandas dataframe.
        self.initData()
        
        # Initializes the 2d square grid geometry specified in initGrid subroutine.
        self.initGrid()

        # Setting the kind must come after initData in initialization. After that no need to worry, can run whenever.
        self.setKind(self.kind)
        
        # Simulation parameters.
        self.mux = 1      # Mean (center of dist in x)
        self.muy = 0      # Mean (center of dist in y)
        self.sigma = 30   # Standard deviation
        self.Nsim = 10000 # Number of points

        # Generate coordinates produced from normal distribution.
        self.x_coords_sim = np.random.normal(self.mux, self.sigma, self.Nsim)
        self.y_coords_sim = np.random.normal(self.muy, self.sigma, self.Nsim)

        return

    def initData(self):
        if self.debug:
            print("DataProcessor class method: initData")

        try:
            self.data = pd.read_csv(self.dataFile, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
        except Exception as err:
            print("Error in reading neutron data:")
            print(err)

            print()
            print("Quitting...")
            sys.exit()

        if self.positronFile == None:
            return

        try:
            self.positron_data = pd.read_csv(self.positronFile, sep="\\s+", skiprows=1, header=0, skipinitialspace=True)
        except Exception as err:
            print("Error in reading positron data:")
            print(err)

            print()
            print("Quitting...")
            sys.exit()
            
        return

    def initGrid(self):
        if self.debug:
            print("DataProcessor class method: initGrid")

        # Size of half of the total detector grid in mm.
        self.half_size = self.cube_size * self.grid_size / 2.0

        # Initializes matrix for segmentation.
        self.seg = np.zeros((self.grid_size, self.grid_size), dtype=dict)

        # Creates bounds and event count bins for specified detector geometry.
        for i in range(self.grid_size):
            for j in range(self.grid_size):

                xlow  = self.cube_size * j - self.half_size
                xhigh = self.cube_size * j + self.cube_size - self.half_size
                
                yhigh = -(self.cube_size * i - self.half_size)
                ylow  = -(self.cube_size * i + self.cube_size - self.half_size)

                self.seg[i][j] = {"xbounds": [xlow, xhigh],
                                  "ybounds": [ylow, yhigh],
                                  "counts":  0}

        return

    def setKind(self, kind):
        if self.debug:
            print("DataProcessor class method: setKind")

        # Set the global variable to the kind of event so code behaves accordingly.
        self.kind = kind
        
        # Get the row of each event that corresponds to each kind of particle event.
        if kind == "capture":
            locs = self.data.groupby("Row").last()
        elif kind == "1st-scatter":
            locs = self.data.groupby("Row").nth(1)
        elif kind == "2nd-scatter":
            locs = self.data.groupby("Row").nth(2)
        else:
            print("Invalid input. Please use only 'capture', '1st-scatter', or '2nd-scatter'.")
            return

        self.coords = {
            "x" : locs["trackPosX"] - locs["mcx"],
            "y" : locs["trackPosY"] - locs["mcy"],
            "z" : locs["trackPosZ"] - locs["mcz"]
        }

        return

    def saveCoords(self):
        if self.debug:
            print("DataProcessor class method: saveCoords")

        np.save("x_coords.npy", np.array(self.coords["x"]))
        np.save("y_coords.npy", np.array(self.coords["y"]))
        np.save("z_coords.npy", np.array(self.coords["z"]))

        return

    def neutronHistogram(self, save=False):
        if self.debug:
            print("DataProcessor class method: neutronHistogram")

        # Calculate the Euclidean distance between the neutron event location and MC truth location (IBD vertex).
        distances = np.sqrt((self.coords["x"])**2 + (self.coords["y"])**2 + (self.coords["z"])**2)
        
        mean = sum(distances) / len(distances)
        print(f"Mean neutron ({self.kind}) track length: {mean} mm")

        # Plots histogram.
        hist = plt.hist(distances, bins=64, color="orange", histtype="step")

        plt.xlabel("$d_n$ (mm)")
        plt.ylabel("Counts")
                
        if save:
            plt.savefig(f"neutron_histogram_{self.kind}.pdf", format="pdf", bbox_inches="tight")
    
        plt.show()

        return

    def positronHistogram(self, save=False):
        if self.debug:
            print("DataProcessor class method: positronHistogram")

        if self.positronFile == None:
            print("No positron file defined...")
            return

        # Get the last row (annihilation location) of each event.
        locs = self.positron_data.groupby("Row").last()

        pl = len(locs)

        neu_locs = self.data.groupby("Row").last()

        px_coords = locs["trackPosX"] - neu_locs["mcx"]
        py_coords = locs["trackPosY"] - neu_locs["mcy"]
        pz_coords = locs["trackPosZ"] - neu_locs["mcz"]

        # Calculate the Euclidean distance between the neutron event location and MC truth location.
        distances = np.sqrt((px_coords)**2 + (py_coords)**2 + (pz_coords)**2)
        
        mean = sum(distances) / len(distances)
        print(f"Mean positron track length: {mean} mm")

        # Plots histogram.
        hist = plt.hist(distances, bins=512, color="red", histtype="step")

        plt.xlim(0, 50)

        plt.xlabel("$d_+$ (mm)")
        plt.ylabel("Counts")
                
        if save:
            plt.savefig(f"positron_histogram.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        return

    def rotateCoords(self, x_coords, y_coords, theta, plot=False):
        if self.debug:
            print("DataProcessor class method: rotateCoords")

        x_rot = []
        y_rot = []

        for x, y in zip(x_coords, y_coords):

            # Distance calculation in 2d.
            r = np.sqrt(x**2 + y**2)

            # Initial angle of capture.
            theta0 = np.arctan2(y, x)

            # Rotation calculation.
            phi = theta * np.pi / 180.0

            # Coordinate transformation.
            xprime = r * np.cos(theta0 - phi)
            yprime = r * np.sin(theta0 - phi)

            x_rot.append(xprime)
            y_rot.append(yprime)

        return x_rot, y_rot

    def binEvents(self, theta, plot=False):
        if self.debug:
            print("DataProcessor class method: binEvents")

        seg = copy.deepcopy(self.seg)

        locs = self.data.groupby("Row").last()

        x_coords = locs["trackPosX"] - locs["mcx"]
        y_coords = locs["trackPosY"] - locs["mcy"]
        z_coords = locs["trackPosZ"] - locs["mcz"]

        for x, y, z in zip(x_coords, y_coords, z_coords):

            # Generate a random point in the center square segment.
            xrand = random.uniform(-self.cube_size / 2.0, self.cube_size / 2.0)
            yrand = random.uniform(-self.cube_size / 2.0, self.cube_size / 2.0)

            # Distance calculation in 2d.
            r = np.sqrt(x**2 + y**2)

            # Initial angle of capture.
            theta0 = np.arctan2(y, x)

            # Rotation calculation.
            phi = theta * np.pi / 180.0

            # Coordinate transformation.
            xprime = r * np.cos(theta0 - phi)
            yprime = r * np.sin(theta0 - phi)

            # Randomizes the rotated event in the center square segment.
            xc = xprime + xrand
            yc = yprime + yrand

            if plot:
                plt.plot([xrand], [yrand], "k.", label="IBD vertex")
                plt.plot([x + xrand], [y + yrand], "r.", label="Event +x-dir")
                plt.plot([xc], [yc], "b.", label="Rotated event")
                plt.xlim(-self.half_size, self.half_size)
                plt.ylim(-self.half_size, self.half_size)
                for i in range(1, self.grid_size):
                        plt.axvline(i * self.cube_size - self.half_size)
                        plt.axhline(i * self.cube_size - self.half_size)
                plt.gca().set_aspect("equal")
                plt.title(f"event={event}, rot={theta}")
                plt.xlabel("x (mm)")
                plt.ylabel("y (mm)")
                plt.legend()
                plt.show()

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    s = seg[i][j]
                    
                    if (s["xbounds"][0] < xc) and (xc < s["xbounds"][1]) and \
                       (s["ybounds"][0] < yc) and (yc < s["ybounds"][1]):
                        s["counts"] += 1
                        
        caps = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                caps[i][j] = seg[i][j]["counts"]

        return caps

    def printBinDists(self, theta_range=[0, 45], plot=False, save=False):
        if self.debug:
            print("DataProcessor class method: printBinDists")

        print("Segmentation:", self.seg)

        if save:
            f = open("bin_dists.txt", "w")

        for theta in range(theta_range[0], theta_range[1]+1):
            caps = self.binEvents(theta, plot=plot)
            print(f"theta = {theta} deg")
            print(caps)

            if save:
                f.write(f"Angle: {theta} deg\n")
                f.write(str(caps) + "\n")

        if save:
            f.close()
                    
        return

    def binDistColormap(self, theta, plot_cmap=True, plot=False, save=False):
        if self.debug:
            print("DataProcessor class method: binDistColormap")

        print("Segmentation:", self.seg)

        caps = self.binEvents(theta)

        ext_low  = -self.half_size
        ext_high =  self.half_size

        im = plt.imshow(caps, cmap="viridis", extent=(ext_low, ext_high, ext_low, ext_high))

        cmax = 1000
        
        cbar = plt.colorbar(im)
        cbar.set_label("Counts")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
        im.set_clim(0, cmax)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                text = plt.text((j-np.floor((self.grid_size / 2.0)))*self.cube_size, -(i-np.floor((self.grid_size / 2.0)))*self.cube_size, f"{int(caps[i, j])}", 
                                ha="center", va="center", color=("w" if int(caps[i, j]) <= 0.6 * cmax else "k"))

        if save:
            plt.savefig(f"bin_dist_cmap_{theta}deg.pdf", format="pdf")

        plt.show()
                    
        return

    def angularDist(self, save=False):
        if self.debug:
            print("DataProcessor class method: angularDist")

        theta_dist = []

        for x, y, z in zip(self.coords["x"], self.coords["y"], self.coords["z"]):

            # Initial angle of capture.
            theta = np.arctan2(y, x)

            theta_dist.append(theta)
            
        # Create the figure and polar axes
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

        # Create the polar histogram.
        hist = ax.hist(theta_dist, bins=np.linspace(-np.pi, np.pi, 37), edgecolor="blue", color="white", linewidth=2, bottom=0)
        
        max_theta = max(hist[0])

        hole_rad = 0.8 * max_theta

        plt.gca().set_yticklabels([])
        plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        ax.set_rorigin(-hole_rad)

        ax.set_ylim(0, max_theta + 0.1 * max_theta)
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2], ["0", "$\\pi/2$","$\\pi$","$3\\pi/2$"])

        plt.text(0, -hole_rad, self.kind.replace("-", " "), ha="center", va="center")
        if save:
            plt.savefig(f"polar_hist_{self.kind}.pdf", format="pdf", bbox_inches="tight")

        plt.show()
        
        return

    def cloudPlot(self, save=False):
        if self.debug:
            print("DataProcessor class method: angularDist")
            
        # Create the figure and polar axes
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.set_title(self.kind.replace("-", " "))

        ax.plot(self.coords["x"], self.coords["y"], ".", markersize=1)

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        l = (100 if self.kind == "capture" else 20)
            
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)

        ax.set_aspect("equal")
        
        if save:
            plt.savefig(f"cloud_plot_{self.kind}.pdf", format="pdf", bbox_inches="tight")

        plt.show()
                    
        return

    def spatialDist(self, dim, save=False):
        if self.debug:
            print("DataProcessor class method: spatialDist")

        plt.hist(self.coords[dim], bins=128, histtype="step")            
        plt.grid()
        
        # The variable 'dim' may be set to 'x', 'y', or 'z'.
        plt.title(f"{dim}-location histogram ({self.kind.replace('-', ' ')})")
        plt.xlabel(f"{self.kind} $-$ vertex (mm)")
        plt.ylabel("Counts")
        plt.show()
        
        return
    
    def gaussian(self, x, amplitude, mean, stddev):
        return amplitude * np.exp(-((x - mean) / stddev)**2 / 2)
    
    def abs_sin(self, x, amplitude, freq, offset):
        return amplitude * np.abs(np.sin(freq * (x - offset)))

    def frobeniusNormAnalysis(self, save=False):
        if self.debug:
            print("DataProcessor class method: frobeniusNormAnalysis")

        # Simulation parameters. The variable 'ref_angle' is the angle of the reference dataset. The variable 'data_range' is the angular range on either side of the reference angle to test. The range of the simulation is 'ref_data' +/- 'data_range'.
        ref_angle = 45
        data_range = 180

        # Range of the data ('ref_data' +/- 'fit_range') to use.
        fit_range = 179

        # Find center diagonal element of segmentation array. Used to zero center segment if necessary.
        center_elem = int(np.floor(self.grid_size / 2.0))

        # Creates a reference dataset at a rotation of 'ref_angle' degrees.
        ref_data = self.binEvents(theta=ref_angle)

        # Creates range of angles to test using 'ref_angle' and 'data_range' variables.
        trange = [ref_angle - data_range, ref_angle + data_range]

        print(f"Grid geometry: ({self.grid_size} x {self.grid_size})")
        print(f"Size of segment: {self.cube_size} mm")
        print("Reference dataset:")
        print(ref_data)

        # Normalizes array to prepare for Frobenius norm calculation.
        ref_data = ref_data / np.sum(ref_data)

        angles = []
        norms = []
        # Loops through discrete range of angles and creates bin distribution.
        for t in tqdm(range(trange[0], trange[1]+1), desc="Calculating Frobenius norms"):
            caps = self.binEvents(theta=t)

            # Normalizes and flattens bin distributions and compares them with the reference dataset with the Frobenius norm of the difference.
            caps = caps / np.sum(caps)

            f_norm = np.sqrt(np.sum(np.square(ref_data - caps)))

            angles.append(t)
            norms.append(f_norm)

        # Find best angle guess via the minimum Frobenius norm of difference in the dataset.
        best_angle = angles[np.argmin(norms)]

        # Setup absolute value of sine fit.
        sin_y_shift = np.min(norms)
        sin_fit_norms = np.array([(fn - sin_y_shift) for fn in norms])

        # Careful: this assumes 'ref_angle' is within 't_range'.
        sin_fit_angles = angles[(ref_angle - trange[0] - fit_range):-(trange[1] - ref_angle - fit_range)]
        sin_fit_norms = sin_fit_norms[(ref_angle - trange[0] - fit_range):-(trange[1] - ref_angle - fit_range)]

        # Perform sine fit.
        sin_popt, sin_pcov = curve_fit(self.abs_sin, sin_fit_angles, sin_fit_norms, p0=[np.max(sin_fit_norms), np.pi/360.0, ref_angle])
        sin_y_fit = self.abs_sin(sin_fit_angles, sin_popt[0], sin_popt[1], sin_popt[2])

        print("Sine fit:", sin_popt)

        fig_width = 10.5
        fig_height = 7.3
        plt.figure(figsize=(fig_width, fig_height))
        
        # Create analysis plot.
        plt.plot(sin_fit_angles, [y + sin_y_shift for y in sin_y_fit], color="red", label=f"$|\\sin\\vartheta|$ fit (${round(sin_popt[2], 2)}" + "^\\circ$)")
        plt.plot(angles, norms, "b.", label="Frobenius norm", alpha=0.3)
        plt.axvline(ref_angle, linestyle="--", color="gray")
        plt.xlabel("$\\vartheta$ (${}^\\circ$)")
        plt.ylabel("Frobenius norm of difference")
        plt.legend(loc="upper left")

        plt.text(ref_angle + 5, 0.95 * sin_y_shift, f"$\\vartheta = {ref_angle}"+"^\\circ$")

        if save:
            plt.savefig(f"fn_analysis_plot.pdf", format="pdf", bbox_inches="tight")

        plt.show() 
                    
        return

    def testVectorSum(self):
        if self.debug:
            print("DataProcessor class method: testVectorSum")

        # Find center diagonal element of segmentation array.
        cent = int(np.floor(self.grid_size / 2.0))

        # Creates a reference dataset at a rotation of 'ref_angle' degrees.
        ref_angle = 35
        ref_data = self.binEvents(theta=ref_angle)

        x_ref = np.cos(ref_angle * np.pi / 180.0)
        y_ref = -np.sin(ref_angle * np.pi / 180.0)

        x_coord = 0
        y_coord = 0
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x_cent = (self.seg[i, j]["xbounds"][0] + self.seg[i, j]["xbounds"][1] ) / 2.0
                y_cent = (self.seg[i, j]["ybounds"][0] + self.seg[i, j]["ybounds"][1] ) / 2.0
                print(i, j, self.seg[i, j], ref_data[i, j], x_cent, y_cent)

                x_coord += ref_data[i, j] * x_cent
                y_coord += ref_data[i, j] * y_cent

        x_unit = x_coord / np.sqrt(x_coord**2 + y_coord**2)
        y_unit = y_coord / np.sqrt(x_coord**2 + y_coord**2)
        
        # Find center diag element of segmentation array.
        cent = int(np.floor(self.grid_size / 2.0))

        # Creates a reference dataset at a rotation of 'ref_angle' degrees.
        ref_angle = 45
        ref_data = self.binEvents(theta=ref_angle)

        trange = [ref_angle, ref_angle + 360]

        print(f"Grid size: ({self.grid_size} x {self.grid_size}), {self.cube_size} mm")
        print(ref_data)

        # Compare to Frobenius norm method. Normalizes array to prepare for Frobenius norm calculation.
        ref_data = ref_data / np.sum(ref_data)

        angles = []
        norms = []
        # Loops through discrete range of angles and creates bin distribution.
        for t in tqdm(range(trange[0], trange[1]+1)):
            caps = self.binEvents(theta=t)

            # Normalizes and flattens bin distributions and compares them with the reference dataset with the Frobenius norm of the difference.
            caps = caps / np.sum(caps)

            f_norm = np.sqrt(np.sum(np.square(ref_data - caps)))

            angles.append(t)
            norms.append(f_norm)

        # Find best angle guess in data.
        best_angle = angles[np.argmin(norms)]

        x_fn = np.cos(best_angle * np.pi / 180.0)
        y_fn = -np.sin(best_angle * np.pi / 180.0)

        plt.arrow(0, 0, x_ref, y_ref, color="black", length_includes_head=True,
                  head_width=0.05, head_length=0.05)
        plt.arrow(0, 0, x_unit, y_unit, color="red", length_includes_head=True,
                  head_width=0.05, head_length=0.05)
        plt.arrow(0, 0, x_ref, y_ref, color="green", length_includes_head=True,
                  head_width=0.05, head_length=0.05)
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.show()

        return

    def testNormalDist(self):
        if self.debug:
            print("DataProcessor class method: testNormalDist")

        # Parameters.
        mu = 5.0          # Mean (center)
        sigma = 1.0       # Standard deviation
        num_points = 1000 # Total number of points (must be even)
        
        # Generate half of the points normally.
        half_points = num_points // 2
        x_half = np.random.normal(mu, sigma, half_points)
        
        # Create a mirrored version of these points.
        x_mirrored = 2 * mu - x_half
        
        # Combine to get a symmetrical distribution.
        x_coords = np.concatenate((x_half, x_mirrored))
        
        # Plot the symmetrical Gaussian distribution.
        plt.hist(x_half, bins=30, density=True, alpha=0.6, color="b", edgecolor="black")
        plt.axvline(mu, color="r", linestyle="dashed", linewidth=2, label=f"Mean={mu}")
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.legend()
        plt.show()
            
        return

    def testPoissonDist(self):
        if self.debug:
            print("DataProcessor class method: testPoissonDist")

        # Example data.
        data = np.random.poisson(5, 1000)

        # Fit the Poisson distribution.
        mu = np.mean(data)
        dist = poisson(mu)

        # Plot the histogram of the data and the fitted Poisson distribution
        plt.hist(data, bins=np.arange(data.max() + 2) - 0.5, density=True, alpha=0.6, color="b", edgecolor="black", label="Data")
        x = np.arange(0, data.max() + 1)
        plt.plot(x, dist.pmf(x), "r.-", label="Fitted Poisson")
        plt.xlabel("x")
        plt.ylabel("Probability density")
        plt.legend()
        plt.show()

        return

    def binEventsSimLoop(self, rot=0):
        if self.debug:
            print("DataProcessor class method: binEventsSimLoop")

        self.initGrid()

        for x, y in zip(self.x_coords_sim, self.y_coords_sim):

            # Distance calculation in 2d.
            r = np.sqrt(x**2 + y**2)

            # Initial angle of capture.
            theta = np.arctan2(y, x)

            # Rotation calculation.
            phi = rot * np.pi / 180.0

            # Coordinate transformation.
            xprime = r * np.cos(theta - phi)
            yprime = r * np.sin(theta - phi)

            # Randomizes the rotated event in the center square segment.
            xc = xprime
            yc = yprime

            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    s = self.seg[i][j]
                    
                    if (s["xbounds"][0] < xc) and (xc < s["xbounds"][1]) and \
                       (s["ybounds"][0] < yc) and (yc < s["ybounds"][1]):
                        s["counts"] += 1
                        
        caps = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                caps[i][j] = self.seg[i][j]["counts"]

        return caps
    
    def frobeniusNormAnalysisSim(self, save=False):
        if self.debug:
            print("DataProcessor class method: frobeniusNormAnalysisSim")

        # Simulation parameters. The variable 'ref_angle' is the angle of the reference dataset. The variable 'data_range' is the angular range on either side of the reference angle to test. the range of the simulation is 'ref_data' +/- 'data_range'.
        ref_angle = 0
        data_range = 180

        # Range of the data ('ref_data' +/- 'fit_range') to use.
        fit_range = 179

        # Creates a reference dataset at a rotation of 'ref_angle' degrees.
        ref_data = self.binEventsSimLoop(rot=ref_angle)

        # Creates range of angles to test using 'ref_angle' and 'data_range' variables.
        trange = [ref_angle - data_range, ref_angle + data_range]

        print(f"Grid geometry: ({self.grid_size} x {self.grid_size})")
        print(f"Size of segment: {self.cube_size} mm")
        print("Reference dataset:")
        print(ref_data)

        # Normalizes array to prepare for Frobenius norm calculation.
        ref_data = ref_data / np.sum(ref_data)

        angles = []
        norms = []
        # Loops through discrete range of angles and creates bin distribution.
        for t in tqdm(range(trange[0], trange[1]+1), desc="Calculating Frobenius norms"):
            caps = self.binEventsSimLoop(rot=t)

            # Normalizes and flattens bin distributions and compares them with the reference dataset with the Frobenius norm of the difference.
            caps = caps / np.sum(caps)

            f_norm = np.sqrt(np.sum(np.square(ref_data - caps)))

            angles.append(t)
            norms.append(f_norm)

        # Setup sine fit.
        sin_y_shift = np.min(norms)
        sin_fit_norms = np.array([(fn - sin_y_shift) for fn in norms])

        # Careful: this assumes 'ref_angle' is within 't_range'
        sin_fit_angles = angles[(ref_angle - trange[0] - fit_range):-(trange[1] - ref_angle - fit_range)]
        sin_fit_norms = sin_fit_norms[(ref_angle - trange[0] - fit_range):-(trange[1] - ref_angle - fit_range)]

        # Perform sine fit.
        sin_popt, sin_pcov = curve_fit(self.abs_sin, sin_fit_angles, sin_fit_norms, p0=[np.max(sin_fit_norms), np.pi/360.0, ref_angle])
        sin_y_fit = self.abs_sin(sin_fit_angles, sin_popt[0], sin_popt[1], sin_popt[2])

        print("Sine fit:", sin_popt)
        
        # Create analysis plot.
        plt.plot(sin_fit_angles, [y + sin_y_shift for y in sin_y_fit], color="red")
        plt.plot(angles, norms, "b.", alpha=0.5)
        plt.xlabel("$\\vartheta$ (${}^\\circ$)")
        plt.ylabel("Frobenius norm of difference")

        if save:
            plt.savefig(f"test_fn_sim_mux{self.mux}.pdf", format="pdf", bbox_inches="tight")
        
        plt.show()
                    
        return

    def sym_2d_norm_dist(self, x, y, theta, sigma, r):
        norm_fac = 1 / (2 * np.pi * sigma**2)
        return norm_fac * np.exp(-(x**2 + y**2 + r**2) / (2 * sigma**2)) * np.exp(-(r * (x * np.cos(theta) + y * np.sin(theta))) / (sigma**2))

    def sym_2d_norm_dist_amp(self, x, y, theta, sigma, r, A):
        return A * np.exp(-(x**2 + y**2 + r**2) / (2 * sigma**2)) * np.exp(-(r * (x * np.cos(theta) + y * np.sin(theta))) / (sigma**2))

    def sym_2d_norm_dist_diff(self, x, y, theta_ref, theta_rot, sigma, r):
        theta_ref *= np.pi / 180.0
        theta_rot *= np.pi / 180.0
        return ( self.sym_2d_norm_dist(x, y, theta_ref, sigma, r) - self.sym_2d_norm_dist(x, y, theta_rot, sigma, r) )**2

    def continuousFrobeniusNorm(self, save=False):
        if self.debug:
            print("DataProcessor class method: continuousFrobeniusNorm")

        ref_angle = 0

        sigma = 30
        r = 100

        x_min, x_max = -100, 100  # Limits for x
        y_min, y_max = -100, 100  # Limits for y
        
        grid_size_sim = 32

        # Create a grid of x and y values for the test plot.
        x = np.linspace(x_min, x_max, grid_size_sim)
        y = np.linspace(y_min, y_max, grid_size_sim)

        # Create 2D grid and ompute function values on the grid.
        X, Y = np.meshgrid(x, y)
        Z = self.sym_2d_norm_dist_diff(X, Y, 0, 45, sigma, r)
        
        # Plot the data.
        plt.imshow(Z, cmap="viridis")
        plt.colorbar(label="Function Value")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

        # Perform double integration.
        angles = []
        norms = []
        discrete_norms = []
        for theta in range(ref_angle-180, ref_angle+180, 1):

            Z1 = self.sym_2d_norm_dist(X, Y, ref_angle * np.pi / 180, sigma, r)
            Z1 = Z1 / np.sum(Z1)
            Z2 = self.sym_2d_norm_dist(X, Y, theta * np.pi / 180, sigma, r)
            Z2 = Z2 / np.sum(Z2)

            dfnd = np.sqrt(np.sum(np.square(Z1 - Z2)))
            discrete_norms.append(dfnd)

            result, error = spi.dblquad(self.sym_2d_norm_dist_diff, x_min, x_max, lambda x: y_min, lambda x: y_max, args=(ref_angle, theta, sigma, r,))
            CFN = np.sqrt(result)
                        
            angles.append(theta)
            norms.append(CFN)


        # Perform sine fit.
        sin_popt, sin_pcov = curve_fit(self.abs_sin, angles, norms, p0=[np.max(norms), np.pi/360.0, ref_angle])
        sin_y_fit = self.abs_sin(angles, sin_popt[0], sin_popt[1], sin_popt[2])

        plt.plot(angles, np.array(norms) / max(norms), ".", color="m", label="Cont. sim. data")
        plt.plot(angles, np.array(discrete_norms) / max(discrete_norms), ".", color="b", label="Discrete sim. data")
        plt.plot(angles, np.array(sin_y_fit) / max(sin_y_fit), color="lightgreen", label="Abs. sine fit")
        plt.xlabel("$\\vartheta$ (${}^\\circ$)")
        plt.ylabel("CFND")
        plt.grid()
        plt.legend(loc="lower right")

        if save:
            plt.savefig(f"cfnd.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        return

    def plotTracks3D(self, event, view=(90, 270, 0), save=False):
        if self.debug:
            print("DataProcessor class method: plotTracks3D")

        if self.positronFile == None:
            print("No positron file defined...")
            return

        pos_event = self.positron_data[self.positron_data["Row"] == event]
        neu_event = self.data[self.data["Row"] == event]

        mcx_data = neu_event["mcx"]
        mcy_data = neu_event["mcy"]
        mcz_data = neu_event["mcz"]

        mcx = mcx_data.iloc[0]
        mcy = mcy_data.iloc[0]
        mcz = mcz_data.iloc[0]

        pl = len(pos_event)
        nl = len(neu_event)

        pos_x_data = pos_event["trackPosX"] - np.full(pl, mcx)
        pos_y_data = pos_event["trackPosY"] - np.full(pl, mcy)
        pos_z_data = pos_event["trackPosZ"] - np.full(pl, mcz)

        neu_x_data = neu_event["trackPosX"] - mcx_data
        neu_y_data = neu_event["trackPosY"] - mcy_data
        neu_z_data = neu_event["trackPosZ"] - mcz_data

        # Show the newly created dictionary (track data). Plots the provided event number.
        print("Plotting 3d track data...")
        fig = plt.figure()
        ax = plt.axes(projection="3d")

        # Initializes with a top-view perspective.
        ax.view_init(view[0], view[1], view[2])

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")
    
        ax.plot3D(neu_x_data, neu_y_data, neu_z_data, color="orange", label="Neutron")
        ax.plot3D(pos_x_data, pos_y_data, pos_z_data, color="red", label="Positron")

        ax.scatter(0, 0, 0, color="black", label="IBD vertex")
        ax.scatter(neu_x_data.iloc[nl-1], neu_y_data.iloc[nl-1], neu_z_data.iloc[nl-1], marker="x", color="orange", label="Capture")
        ax.scatter(pos_x_data.iloc[pl-1], pos_y_data.iloc[pl-1], pos_z_data.iloc[pl-1], marker="x", color="red", label="Annihilation")

        if save:
            plt.savefig("track_plot_3d.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        return

    def plotTracks2D(self, event, save=False):
        if self.debug:
            print("DataProcessor class method: plotTracks2D")

        if self.positronFile == None:
            print("No positron file defined...")
            return

        pos_event = self.positron_data[self.positron_data["Row"] == event]
        neu_event = self.data[self.data["Row"] == event]

        mcx_data = neu_event["mcx"]
        mcy_data = neu_event["mcy"]

        mcx = mcx_data.iloc[0]
        mcy = mcy_data.iloc[0]

        pl = len(pos_event)
        nl = len(neu_event)

        pos_x_data = pos_event["trackPosX"] - np.full(pl, mcx)
        pos_y_data = pos_event["trackPosY"] - np.full(pl, mcy)

        neu_x_data = neu_event["trackPosX"] - mcx_data
        neu_y_data = neu_event["trackPosY"] - mcy_data

        # Show the newly created dictionary (track data). Plots the provided event number.
        print("Plotting 2d track data...")
        fig = plt.figure()
        ax = plt.axes()

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    
        ax.plot(neu_x_data, neu_y_data, color="orange", label="Neutron")
        ax.plot(pos_x_data, pos_y_data, color="red", label="Positron")

        ax.scatter(0, 0, color="black", label="IBD vertex")
        ax.scatter(neu_x_data.iloc[nl-1], neu_y_data.iloc[nl-1], marker="x", color="orange", label="Capture")
        ax.scatter(pos_x_data.iloc[pl-1], pos_y_data.iloc[pl-1], marker="x", color="red", label="Annihilation")

        if save:
            plt.savefig("track_plot_2d.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        return

    def gaussian_2d(self, coords, A, mu_x, mu_y, sigma_x, sigma_y):
        x, y = coords
        return A * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))

    def CFND_eval_int(self, theta0, theta, sigma, mu):

        norm_fac = 1 / (2 * np.pi * sigma**2)
        exponent = (mu**2 * (np.cos((theta0 + theta) * np.pi / 180.0) - 1)) / (2 * sigma**2)
        return np.sqrt(norm_fac * (1 - np.exp(exponent)))

    def normalDistFit(self, save=False):
        if self.debug:
            print("DataProcessor class method: normalDistFit")

        # Must be integer number of grid.
        grid_size = 25

        # Cube size in mm
        cube_size = 10

        max_c = 125

        x_coords = self.coords["x"]
        y_coords = self.coords["y"]

        l = (cube_size * grid_size) / 2.0

        # Create the figure and polar axes
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        hist = ax.hist2d(x_coords, y_coords, bins=grid_size, cmap="gray_r", range=[[-l, l], [-l, l]], rasterized=True)

        x_centers = np.linspace(-l + 0.5 * cube_size, l - 0.5 * cube_size, grid_size)
        y_centers = np.linspace(-l + 0.5 * cube_size, l - 0.5 * cube_size, grid_size)
        X, Y = np.meshgrid(x_centers, y_centers)

        # Flatten for fitting.
        x_data = X.ravel()
        y_data = Y.ravel()
        z_data = hist[0].T.ravel()

        # Initial parameter estimates: amplitude (A), means (mu_x, mu_y), stds (sigma_x, sigma_y)
        A_guess = np.max(hist[0])
        mu_x_guess, mu_y_guess = np.mean(x_coords), np.mean(y_coords)
        sigma_x_guess, sigma_y_guess = np.std(x_coords), np.std(y_coords)
        initial_guess = (A_guess, mu_x_guess, mu_y_guess, sigma_x_guess, sigma_y_guess)

        # Fit the 2D Gaussian to the histogram data.
        popt, _ = curve_fit(self.gaussian_2d, (x_data, y_data), z_data, p0=initial_guess)

        # Extract the fitted parameters.
        A_fit, mu_x_fit, mu_y_fit, sigma_x_fit, sigma_y_fit = popt

        # Print the fitted parameters.
        print(f"Fitted Amplitude: {A_fit:.2f}")
        print(f"Normal dist. amplitude: {1 / (2 * np.pi * sigma_x_fit * sigma_y_fit)}")
        print(f"Frac: {A_fit / (len(x_coords) * cube_size**2)}")
        print(f"Fitted Mean (μ_x, μ_y): ({mu_x_fit:.2f}, {mu_y_fit:.2f})")
        print(f"Fitted Std Devs (σ_x, σ_y): ({sigma_x_fit:.2f}, {sigma_y_fit:.2f})")

        hist[3].set_clim(vmin=0, vmax=max_c)
        
        cbar = fig.colorbar(hist[3], label="Counts")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_title("Capture histogram")
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        
        ax.set_aspect('equal', adjustable='box')

        if save:
            plt.savefig(f"c_normal_dist_counts_plot.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        # Plot continuous 2D Gaussian distribution.
        n_points = 128

        # Create a grid of points in the specified region
        x = np.linspace(-l, l, n_points)
        y = np.linspace(-l, l, n_points)
        X, Y = np.meshgrid(x, y)

        # Compute the values of the function at each point in the grid.
        Z = self.gaussian_2d((X, Y), A_fit, mu_x_fit, mu_y_fit, sigma_x_fit, sigma_y_fit)

        # Create the figure and polar axes.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        img = ax.imshow(Z, extent=[-l, l, -l, l], origin='lower', cmap='gray_r', aspect='equal')

        img.set_clim(vmin=0, vmax=max_c)
        
        cbar = fig.colorbar(img, label="Function value")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_title("Gaussian distribution")
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)

        if save:
            plt.savefig(f"c_normal_dist_fit_plot.pdf", format="pdf", bbox_inches="tight")

        plt.show()
        
        return

    def CFNDAnalysis(self, save=False):
        if self.debug:
            print("DataProcessor class method: CFNDAnalysis")
        
        # COmpares four different functions: (1) RATPAC2 data FND, (2) Sampled normal distribution FND, (3) CFND, and (4) CFND evaluated integral in Mathematica.

        # Must be integer number of grid.
        grid_size = 11

        # Cube size in mm.
        cube_size = 30

        max_c = 125

        x_coords = self.coords["x"]
        y_coords = self.coords["y"]

        n = len(x_coords)

        l = (cube_size * grid_size) / 2.0

        # Create the figure and polar axes.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        hist = ax.hist2d(x_coords, y_coords, bins=grid_size, cmap="gray_r", range=[[-l, l], [-l, l]], rasterized=True)

        x_centers = np.linspace(-l + 0.5 * cube_size, l - 0.5 * cube_size, grid_size)
        y_centers = np.linspace(-l + 0.5 * cube_size, l - 0.5 * cube_size, grid_size)
        X, Y = np.meshgrid(x_centers, y_centers)

        # Flatten for fitting.
        x_data = X.ravel()
        y_data = Y.ravel()
        z_data = hist[0].T.ravel()

        # Initial parameter estimates: amplitude (A), means (mu_x, mu_y), stds (sigma_x, sigma_y).
        A_guess = np.max(hist[0])
        mu_x_guess, mu_y_guess = np.mean(x_coords), np.mean(y_coords)
        sigma_x_guess, sigma_y_guess = np.std(x_coords), np.std(y_coords)
        initial_guess = (A_guess, mu_x_guess, mu_y_guess, sigma_x_guess, sigma_y_guess)

        # Fit the 2D Gaussian to the histogram data.
        popt, _ = curve_fit(self.gaussian_2d, (x_data, y_data), z_data, p0=initial_guess)

        # Extract the fitted parameters.
        A_fit, mu_x_fit, mu_y_fit, sigma_x_fit, sigma_y_fit = popt

        # Print the fitted parameters.
        print(f"Fitted Amplitude: {A_fit:.2f}")
        print(f"Normal dist. amplitude: {1 / (2 * np.pi * sigma_x_fit * sigma_y_fit)}")
        print(f"Frac: {A_fit / (len(x_coords) * cube_size**2)}")
        print(f"Fitted Mean (μ_x, μ_y): ({mu_x_fit:.2f}, {mu_y_fit:.2f})")
        print(f"Fitted Std Devs (σ_x, σ_y): ({sigma_x_fit:.2f}, {sigma_y_fit:.2f})")

        hist[3].set_clim(vmin=0, vmax=max_c)
        
        cbar = fig.colorbar(hist[3], label="Counts")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_title("Capture histogram")
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        
        ax.set_aspect("equal", adjustable="box")

        if save:
            plt.savefig(f"c_normal_dist_counts_plot.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        angles = range(-180, 180)
        norms = []

        ref = np.histogram2d(y_coords, x_coords, bins=grid_size, range=[[-l, l], [-l, l]])[0]
        ref = ref / np.sum(ref)

        # Calculate FND of RATPAC2 data.
        for theta in angles:

            x_rot, y_rot = self.rotateCoords(x_coords, y_coords, theta)
            
            hist = np.histogram2d(y_rot, x_rot, bins=grid_size, range=[[-l, l], [-l, l]])[0]
            hist = hist / np.sum(hist)

            FND = np.sqrt(np.sum(np.square(ref - hist)))

            norms.append(FND)

        FND_angles = angles
        FND_norms = norms

        # Plot continuous 2d Gaussian distribution.
        n_points = grid_size

        # Create a grid of points in the specified region.
        x = np.linspace(-l, l, n_points)
        y = np.linspace(-l, l, n_points)
        X, Y = np.meshgrid(x, y)

        # Compute the values of the function at each point in the grid.
        Z = self.gaussian_2d((X, Y), A_fit, mu_x_fit, mu_y_fit, sigma_x_fit, sigma_y_fit)

        # Create the figure and polar axes.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        img = ax.imshow(Z, extent=[-l, l, -l, l], origin="lower", cmap="gray_r", aspect="equal")

        img.set_clim(vmin=0, vmax=max_c)
        
        cbar = fig.colorbar(img, label="Function value")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_title("Gaussian distribution")
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)

        if save:
            plt.savefig(f"c_normal_dist_fit_plot.pdf", format="pdf", bbox_inches="tight")

        plt.show()

        angles = range(-180, 180)
        norms = []

        ref = self.sym_2d_norm_dist_amp(X, Y, 0, (sigma_x_fit + sigma_y_fit) / 2.0, np.sqrt(mu_x_fit**2 + mu_y_fit**2), A_fit)
        ref = ref / np.sum(ref)

        # Calculate FND of RATPAC2 data.
        for theta in angles:
            hist = self.sym_2d_norm_dist_amp(X, Y, theta * np.pi/180.0, (sigma_x_fit + sigma_y_fit) / 2.0, np.sqrt(mu_x_fit**2 + mu_y_fit**2), A_fit)
            hist = hist / np.sum(hist)

            FND = np.sqrt(np.sum(np.square(ref - hist)))

            norms.append(FND)

        FND_sampled_angles = angles
        FND_sampled_norms = norms

        angles = range(-180, 180)
        norms = []
        
        for theta in angles:
            result, error = spi.dblquad(self.sym_2d_norm_dist_diff, -l, l, lambda x: -l, lambda x: l, args=(0, theta, (sigma_x_fit + sigma_y_fit) / 2.0, np.sqrt(mu_x_fit**2 + mu_y_fit**2),))
            CFND = np.sqrt(result)
            norms.append(CFND)

        CFND_angles = angles
        CFND_norms = norms
        
        angles = np.array(range(-180, 180))
        norms = self.CFND_eval_int(0, angles, (sigma_x_fit + sigma_y_fit) / 2.0, np.sqrt(mu_x_fit**2 + mu_y_fit**2))

        CFND_eval_angles = angles
        CFND_eval_norms = norms

        plt.plot(FND_angles, FND_norms, "r.", label="FND of RATPAC2 data")
        plt.plot(FND_sampled_angles, FND_sampled_norms, "gx", label="FND of sampled Gauss")
        plt.plot(CFND_angles, cube_size * np.array(CFND_norms), "b+", label="CFND")
        plt.plot(CFND_eval_angles, cube_size * np.array(CFND_eval_norms), "m-", label="CFND evaluated expression")
        plt.legend(loc="upper right")
        plt.ylabel("Norm")
        plt.xlabel("$\\vartheta$ (${}^\\circ$)")
        plt.show()
        
        return

    def cmapPlot(self, save=False):
        if self.debug:
            print("DataProcessor class method: cmapPlot")
            
        # Create the figure and polar axes.
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot()

        l = (100 if self.kind == "capture" else 10)

        hist = ax.hist2d(self.coords["x"], self.coords["y"], bins=64, cmap="gray_r", range=[[-l, l], [-l, l]], rasterized=False)
        fig.colorbar(hist[3], label="Counts")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_title(self.kind.replace("-", " "))
        
        ax.set_xlim(-l, l)
        ax.set_ylim(-l, l)
        
        ax.set_aspect("equal", adjustable='box')

        if save:
            plt.savefig(f"cmap_plot_{self.kind}.pdf", format="pdf", bbox_inches="tight")

        plt.show()
                    
        return
