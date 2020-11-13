import os
from glob import glob
from natsort import natsorted
import numpy as np
import pandas as pd
import argparse
import matplotlib
from matplotlib.widgets import Button
matplotlib.use('TkAgg')  # MUST BE CALLED BEFORE IMPORTING plt
import matplotlib.pyplot as plt


class ResultsSet:
    def __init__(self, pattern):
        self.cutoff = 8.0
        self.folders = natsorted(glob(pattern))
        self.n = len(self.folders)
        self.frames = []
        self.results = []
        self.pocket_names = [[] for _ in range(self.n)]
        self.volumes = np.zeros(shape=self.n, dtype=np.float_)
        return

    def parse(self):
        for f in self.folders:
            try:
                self.frames.append(glob(os.path.join(f, "*pdb"))[0])
            except IndexError:
                print(f, "does not contain a pdb file")
            try:
                summary = glob(os.path.join(f, "*rept"))[0]
                self.results.append(pd.read_csv(summary))
            except IndexError:
                self.results.append(None)
        return

    def sum_pockets(self):
        self.volumes = np.zeros(shape=self.n, dtype=np.float_)
        self.pocket_names = [[] for _ in range(self.n)]
        for j in range(self.n):
            if self.results[j] is None:
                continue
            for i in range(self.results[j].shape[0]):
                if self.results[j].distance[i] < self.cutoff:
                    self.volumes[j] += self.results[j].volume[i]
                    self.pocket_names[j].append(self.results[j].name[i])
        return

    def update(self, event):
        if event.button == 1:
            self.cutoff += 0.5
            self.sum_pockets()
        plt.clf()
        plt.plot(np.arange(self.n), self.volumes, "c-")
        plt.title(self.cutoff)
        plt.draw()
        return

    def inc(self, event):
        if event.button == 1:
            self.cutoff += 0.5
            self.sum_pockets()
        self.figdata.set_ydata(self.volumes)
        self.ax.set_title(self.cutoff)
        self.ax.set_ylim((0, 1.05*np.max(self.volumes)))
        plt.draw()
        return

    def dcr(self, event):
        if event.button == 1:
            self.cutoff -= 0.5
            self.sum_pockets()
        self.figdata.set_ydata(self.volumes)
        self.ax.set_title(self.cutoff)
        self.ax.set_ylim((0, 1.05 * np.max(self.volumes)))
        plt.draw()
        return

    def opt_cutoff(self):
        self.sum_pockets()
        plt.rcParams.update({'font.size': 14})
        f, self.ax = plt.subplots()
        self.ax.set_title(self.cutoff)
        self.ax.set_ylim((0, 1.05 * np.max(self.volumes)))
        self.ax.set_ylabel("Volume / A$^3$")
        self.figdata, = self.ax.plot(np.arange(self.n), self.volumes, "o-")
        self.ax.xaxis.set_visible(False)
        axi = plt.axes([0.60, 0.0, 0.18, 0.07])
        axd = plt.axes([0.78, 0.0, 0.18, 0.07])
        bi = Button(axi, 'Increase')
        bi.on_clicked(self.inc)
        bd = Button(axd, 'Decrease')
        bd.on_clicked(self.dcr)
        plt.tight_layout()
        plt.show()
        delattr(self, "figdata")
        delattr(self, "ax")
        return


class Pocket:
    def __init__(self, name, volume, distance):
        self.name = name
        self.volume = float(volume)
        self.distance = float(distance)
        return

    def __str__(self):
        return self.name


def read_out(filename):
    volume = []
    pocket = []
    with open(filename, "r") as fin:
        for l in fin:
            try:
                ll = l.split()
                volume.append(float(ll[3]))
                pocket.append([])
                for i in range(4, len(ll)):
                    pocket[-1].append(ll[i])
            except IndexError:
                pass
    return np.array(volume), pocket


def plot_volume(volume, name, fout=None):
    plt.rcParams.update({'font.size': 14})
    f, a = plt.subplots()
    plt.title(name)
    plt.ylabel("Volume / A$^3$")
    a.plot(np.arange(volume.shape[0]), volume, "c-")
    a.xaxis.set_visible(False)
    if fout is None:
        plt.show()
    return


parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file")
parser.add_argument("-p", "--pattern")
args = parser.parse_args()
if args.file is not None:
    name = args.file.split(os.sep)[-1].split('.')[0]
    v, p = read_out(args.file)
    plot_volume(v, name)
elif args.pattern is not None:
    rs = ResultsSet(args.pattern)
    rs.parse()
    rs.opt_cutoff()
print("")
