import os
from datetime import datetime
from glob import glob
from natsort import natsorted
import numpy as np
import pandas as pd
import argparse
import mdtraj as md
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('TkAgg')  # MUST BE CALLED BEFORE IMPORTING plt
import matplotlib.pyplot as plt
from pyvol.spheres import Spheres


class ResultsSet:
    def __init__(self, pattern):
        self.cutoff = 3.0
        self.wrkdr = os.path.dirname(pattern)
        self.folders = natsorted(glob(pattern))
        self.n = len(self.folders)
        self.xval = None
        self.frames = []
        self.results = []
        self.pocket_names = [[] for _ in range(self.n)]
        self.pockets = []
        self.volumes = np.zeros(shape=self.n, dtype=np.float_)
        self.ref_sel = []
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
            try:
                self.xval = np.loadtxt(os.path.join(self.wrkdr, "xval.dat"))
            except IOError:
                pass
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
        for i in range(len(self.volumes)):
            if self.volumes[i] == 0:
                self.volumes[i] = float("NaN")
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
        self.ax.set_ylim((0, 1.05*np.nanmax(self.volumes)))
        plt.draw()
        return

    def dcr(self, event):
        if event.button == 1:
            self.cutoff -= 0.5
            self.sum_pockets()
        self.figdata.set_ydata(self.volumes)
        self.ax.set_title(self.cutoff)
        self.ax.set_ylim((0, 1.05 * np.nanmax(self.volumes)))
        plt.draw()
        return

    def opt_cutoff(self):
        self.sum_pockets()
        plt.rcParams.update({'font.size': 14})
        f, self.ax = plt.subplots()
        self.ax.set_title(self.cutoff)
        self.ax.set_ylim((0, 1.05 * np.nanmax(self.volumes)))
        self.ax.set_ylabel("Volume / A$^3$")
        self.figdata, = self.ax.plot(np.arange(self.n), self.volumes, "o-")
        if self.xval is None:
            self.ax.xaxis.set_visible(False)
        else:
            self.ax.set_xticks([0, 24, 49, 74, 99])
            self.ax.set_xticklabels(["{:.1f}".format(self.xval[0]), "{:.1f}".format(self.xval[24]),
                                     "{:.1f}".format(self.xval[49]), "{:.1f}".format(self.xval[74]),
                                     "{:.1f}".format(self.xval[99])])
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

    def identify(self, ID, index):

        return

    def get_pocketID(self, index):
        frame = md.load(self.frames[index])
        refs = np.empty(shape=(len(self.ref_sel), 3), dtype=np.float_)
        for i in range(len(self.ref_sel)):
            refs[i] = md.compute_center_of_mass(frame.atom_slice(frame.top.select(self.ref_sel[i])))
        refs = 10 * refs
        pocketID = np.empty(shape=(self.results[index].shape[0], refs.shape[0]))
        for i in range(self.results[index].shape[0]):
            p = Spheres(spheres_file="{0:s}.obj".format(os.path.join(self.folders[index], self.results[index].name[i])))
            for j in range(refs.shape[0]):
                nearest = p.nearest_coord_to_external(refs[j])
                pocketID[i][j] = np.linalg.norm(refs[j] - nearest)
        return pocketID

    def process(self):
        for i in [0]:
            pid = self.get_pocketID(i)

        return

    def write_pml(self):
        """
        This function is specific to the SARS-CoV-2 helicase visualization
        """
        tmpl = "/mnt/data/covid/pyvol/PCA/visualization_template.pse"
        with open(os.path.join(self.wrkdr, "pyvol_{:%Y%m%d%H%M}.pml".format(datetime.now())), "w") as fout:
            fout.write("load {:s}\n".format(tmpl))
            for i in range(self.n):
                fout.write("load {0:s}, frame{1:03d}\n".format(self.frames[i], i))
                # fout.write("align frame{:03d}, struct\n".format(i))
                for j in range(len(self.pocket_names[i])):
                    fout.write("load {0:s}.xyz\n".format(os.path.join(self.folders[i], self.pocket_names[i][j])))
                    fout.write("color cyan, {0:s}; hide everything, {0:s}; show surface, {0:s}\n".format(
                        self.pocket_names[i][j]))
            fout.write("select domain1, frame* and i. 1-240\n")
            fout.write("select RecA1, frame* and i. 241-442\n")
            fout.write("select RecA2, frame* and i. 443-601\n")
            fout.write("color deepteal, domain1 and elem C\n")
            fout.write("color yellow, RecA1 and elem C\n")
            fout.write("color hotpink, RecA2 and elem C\n")
            fout.write("align struct, frame000;\n")
            fout.write("zoom struct;\n")
            fout.write("mset 1x{:d};\n".format(self.n))
            for i in range(self.n):
                enable_pockets = ""
                for j in range(len(self.pocket_names[i])):
                    enable_pockets += "enable {:s}; ".format(self.pocket_names[i][j])
                fout.write("disable all; enable struct; enable frame{0:03d}; {1:s}scene s{0:03d}, store;\n".format(
                    i, enable_pockets))
                fout.write("mview store, {0:d}, scene=s{0:03d};\n".format(i))
        return

    def write_vis_pml(self):
        """
        This function is specific to the SARS-CoV-2 helicase visualization. Resaving movie frames after adjustment.
        """
        with open(os.path.join(self.wrkdr, "pyvol_movie.pml".format(datetime.now())), "w") as fout:
            fout.write("mset; rewind\n")
            fout.write("mset 1x{:d};\n".format(self.n))
            for i in range(self.n):
                enable_pockets = ""
                for j in range(len(self.pocket_names[i])):
                    enable_pockets += "enable {:s}; ".format(self.pocket_names[i][j])
                fout.write("disable all; enable struct; enable frame{0:03d}; {1:s}scene s{0:03d}, store;\n".format(
                    i, enable_pockets))
                fout.write("mview store, {0:d}, scene=s{0:03d};\n".format(i))
        return

    def write_RNA_pml(self):
        """
        This function is specific to the SARS-CoV-2 helicase visualization, with the extra assumption, that the
        largest pocket is the RNA
        """
        with open(os.path.join(self.wrkdr, "pyvol_largest_{:%Y%m%d%H%M}.pml".format(datetime.now())), "w") as fout:
            for i in range(self.n):
                if self.results[i] is not None:
                    fout.write("load {0:s}.xyz, largest{1:03d}\n".format(
                        os.path.join(self.folders[i], self.results[i].name[0]), i))
            fout.write("hide everything, largest*; show surface, largest*\n")
            fout.write("zoom struct;\n")
            for i in range(self.n):
                fout.write("scene s{0:03d}, recall; enable largest{0:03d}; scene s{0:03d}, update;\n".format(i))
                fout.write("mview store, {0:d}, scene=s{0:03d};\n".format(i))
        return

    def plot_selected(self, fout=None):
        plt.rcParams.update({'font.size': 14})
        f, a = plt.subplots()
        plt.ylabel("Volume / A$^3$")
        if self.xval is None:
            a.xaxis.set_visible(False)
        else:
            a.set_xticks([0, 24, 49, 74, 99])
            a.set_xticklabels(["{:.1f}".format(self.xval[0]), "{:.1f}".format(self.xval[24]),
                               "{:.1f}".format(self.xval[49]), "{:.1f}".format(self.xval[74]),
                               "{:.1f}".format(self.xval[99])])
        a.plot(np.arange(self.n), self.volumes, "o-", label="ATP")
        plt.legend(loc="best")
        plt.tight_layout()
        if fout is None:
            plt.show()
        else:
            plt.savefig(fout, dpi=300)
            np.savetxt(fout.replace(".png", ".txt"), self.volumes)
        return

    def plot_selected_and_largest(self, fout=None):
        plt.rcParams.update({'font.size': 14})
        f, a = plt.subplots()
        plt.ylabel("Volume / A$^3$")
        if self.xval is None:
            a.xaxis.set_visible(False)
        else:
            a.set_xticks([0, 24, 49, 74, 99])
            a.set_xticklabels(["{:.1f}".format(self.xval[0]), "{:.1f}".format(self.xval[24]),
                               "{:.1f}".format(self.xval[49]), "{:.1f}".format(self.xval[74]),
                               "{:.1f}".format(self.xval[99])])
        a.plot(np.arange(self.n), self.volumes, "o-", label="ATP")
        largest = [r.volume[0] if r is not None else float("NaN") for r in self.results]
        a.plot(np.arange(self.n), largest, "o-", label="largest")
        a.set_ylim([0, np.nanmin([np.nanmax(largest), 3800])+200])
        plt.legend(loc="best")
        plt.tight_layout()
        if fout is None:
            plt.show()
        else:
            plt.savefig(fout, dpi=300)
        return


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pattern")
parser.add_argument("--fig")
args = parser.parse_args()
ref_selection = ["resSeq <= 240", "240 < resSeq <= 445", "445 < resSeq"]
if args.pattern is not None:
    rs = ResultsSet(args.pattern)
    rs.parse()
    rs.ref_sel = ref_selection
    rs.get_pocketID(0)
    # rs.opt_cutoff()
    # rs.write_pml()
    # rs.write_RNA_pml()
    # rs.write_vis_pml()
    # rs.plot_selected(args.fig)
print("")
