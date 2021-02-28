import os
import time
from datetime import datetime
from glob import glob
from multiprocessing import Pool
from contextlib import closing
from natsort import natsorted
import numpy as np
# from numba import guvectorize, float32
from scipy.stats.stats import pearsonr
import pandas as pd
import seaborn as sns
import argparse
import pickle
import mdtraj as md
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib.widgets import Button
import matplotlib
matplotlib.use('TkAgg')  # MUST BE CALLED BEFORE IMPORTING plt
import matplotlib.pyplot as plt
from pyvol.spheres import Spheres


# @guvectorize([(float32[:], float32[:], float32)], '(n),(n)->()')#, nopython=True)
# def dist_v(pid, gp, res):
#     res = np.linalg.norm(pid - gp)
#     print(res)
#     return


def dist(pid, gp):
    return np.linalg.norm(pid - gp)


class ResultsSet:
    def __init__(self, *args, **kwargs):
        self.cutoff = 3.0
        try:
            # self.wrkdr = os.path.dirname(kwargs["pattern"])
            self.wrkdr = os.getcwd()
            self.folders = natsorted(glob(kwargs["pattern"]))
        except KeyError:
            self.wrkdr = os.curdir
            self.folders = []
        self.n = len(self.folders)
        self.pocket_names = [[] for _ in range(self.n)]
        self.volumes = np.zeros(shape=self.n, dtype=np.float_)
        self.xval = None
        self.frames = []
        self.results = []
        self.pocket_legend = [] # marked for deprecation
        self.pocket_IDs = None
        self.all_volumes = None
        self.ref_sel = None
        self.ref_poc = None
        self.n_clusters = None
        return

    def save(self, fname="resultset.p"):
        fout = open(fname, "wb")
        pickle.dump(self, fout)
        fout.close()
        return

    def load(self, fname="resultset.p"):
        fout = open(fname, "rb")
        loaded = pickle.load(fout)
        fout.close()
        return loaded

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

    def parse_reference(self, fname):
        self.ref_poc = pd.read_csv(fname, sep=';', header=None).to_numpy()
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

    def get_pocketID(self, index):
        """
        Generates distances to pocket for identification.
        """
        frame = md.load(self.frames[index])
        print(self.frames[index])
        refs = np.empty(shape=(len(self.ref_sel), 3), dtype=np.float_)
        for i in range(len(self.ref_sel)):
            if "CA" in self.ref_sel[i]:
                refs[i] = frame.xyz[0][frame.top.select(self.ref_sel[i])[0]]
            else:
                refs[i] = md.compute_center_of_mass(frame.atom_slice(frame.top.select(self.ref_sel[i])))
        refs = 10 * refs
        try:
            pocketID = np.empty(shape=(self.results[index].shape[0], refs.shape[0]), dtype=np.float32)
            for i in range(self.results[index].shape[0]):
                p = Spheres(spheres_file="{0:s}.obj".format(os.path.join(self.folders[index], self.results[index].name[i])))
                for j in range(refs.shape[0]):
                    nearest = p.nearest_coord_to_external(refs[j])
                    pocketID[i][j] = np.linalg.norm(refs[j] - nearest)
            return pocketID
        except AttributeError:
            return []

    def prepare_for_clustering(self):
        all = []
        self.n_clusters = 2
        for i in range(self.n):
            for pid in self.get_pocketID(i):
                all.append(pid)
            pid = self.get_pocketID(i)
            for j in range(pid.shape[0]):
                if j == self.n_clusters:
                    self.n_clusters += 1
                all.append(pid[j])
                self.pocket_legend.append((i, j))
        self.pocket_IDs = np.array(all)
        return

    def opt_cluster(self, krange=range(2, 16), plot=False):
        """
        Optimises the number of clusters.
        """
        # km = KMeans(n_clusters=8).fit(self.pocket_IDs)
        # https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        Sum_of_squared_distances = []
        for k in krange:
            km = KMeans(n_clusters=k)
            km = km.fit(self.pocket_IDs)
            Sum_of_squared_distances.append(km.inertia_)
        if plot:
            plt.plot(krange, Sum_of_squared_distances, 'bx-')
            plt.xlabel('Number of clusters')
            plt.ylabel('Sum of squared distances')
            plt.show()
        # https://realpython.com/k-means-clustering-python/#choosing-the-appropriate-number-of-clusters
        # A list holds the silhouette coefficients for each k
        silhouette_coefficients = []
        # Notice you start at 2 clusters for silhouette coefficient
        for k in krange:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(self.pocket_IDs)
            score = silhouette_score(self.pocket_IDs, kmeans.labels_)
            silhouette_coefficients.append(score)
        if plot:
            plt.plot(krange, silhouette_coefficients)
            plt.xticks(krange)
            plt.xlabel("Number of Clusters")
            plt.ylabel("Silhouette Coefficient")
            plt.show()
        self.n_clusters = krange[silhouette_coefficients.index(max(silhouette_coefficients))]
        return

    def plot_cluster(self):
        """
        Plots clustered ID vectors in the first 3 dimensions.
        """
        km = KMeans(n_clusters=self.n_clusters)
        km = km.fit(self.pocket_IDs)
        centroids = km.cluster_centers_
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pocket_IDs[:, 0], self.pocket_IDs[:, 1], self.pocket_IDs[:, 2],
                   c=km.labels_.astype(float), s=50, alpha=0.5)
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', s=50)
        plt.show()
        return

    def identify_clusters(self, plot=False):
        self.all_volumes = np.zeros(shape=(self.n, self.n_clusters), dtype=np.float_)
        km = KMeans(n_clusters=self.n_clusters)
        km = km.fit(self.pocket_IDs)
        for i in range(self.pocket_IDs.shape[0]):
            self.all_volumes[self.pocket_legend[i][0]][km.labels_[i]] +=\
                self.results[self.pocket_legend[i][0]].volume[self.pocket_legend[i][1]]
        self.all_volumes[self.all_volumes == 0] = 'nan'
        if plot:
            f, ax = plt.subplots()
            x = range(self.n)
            for i in range(self.n_clusters):
                ax.plot(x, self.all_volumes[:, i], "o-", label="Cluster {:d}".format(i + 1))
            plt.xlabel("Frame")
            plt.ylabel("Volume / A$^3$")
            plt.legend(loc="best")
            plt.show()
        return

    def process(self):
        """
        Processed all frames for pocket IDs.
        """
        # how to flatten:
        # v = p.reshape((1020, 3))
        # w = v[~np.isnan(v).any(axis=1)]
        # all = []
        # for i in range(self.n):
        #     all.append(self.get_pocketID(i))
        if self.ref_sel is None:
            frame = md.load(self.frames[0])
            # self.ref_sel = ["resid {0:d} to {1:d}".format(i, i + 9) for i in range(0, frame.n_residues, 10)]
            self.ref_sel = ["resid {0:d} and name CA".format(i) for i in range(2, frame.n_residues, 10)]
        if int(self.n/args.threads) == 0:
            chunk = 1
        else:
            chunk = int(self.n/args.threads)
        # for i in range(self.n):
        #     rv = self.get_pocketID(i)
        with closing(Pool(processes=int(args.threads))) as pool:
            all = pool.map(self.get_pocketID, range(self.n), chunk)
            pool.terminate()
        b = np.zeros(shape=(len(all), len(max(all, key=lambda x: len(x))), len(self.ref_sel)), dtype=np.float32)
        for i, j in enumerate(all):
            try:
                b[i][0:len(j)] = j
            except ValueError:
                pass
        b[b == 0] = 'nan'
        self.pocket_IDs = b
        return

    def grid(self, bins=10, plot=False, cutoff_factor=0.5, keep_bin_ratio=0.25):
        all_volumes = []
        dimensions = self.pocket_IDs.shape[2]
        boundaries = [np.nanmin(self.pocket_IDs, axis=(0, 1)), np.nanmax(self.pocket_IDs, axis=(0, 1))]
        lspace = [np.linspace(boundaries[0][i], boundaries[1][i], bins) for i in range(len(boundaries[0]))]
        mesh = np.array(np.meshgrid(*lspace), dtype=np.float32)
        cutoff = dist(mesh[:, 0, 0, 0], mesh[:, 1, 1, 1]) * cutoff_factor
        mesh = mesh.swapaxes(0, dimensions).reshape(bins**dimensions, 3)
        pockets = []
        for i in range(mesh.shape[0]):
            dists = np.empty(shape=(self.n, self.pocket_IDs.shape[1]), dtype=np.float32)
            for j in range(self.n):
                for k in range(self.pocket_IDs.shape[1]):
                    dists[j][k] = dist(self.pocket_IDs[j][k], mesh[i])
            min_dist = np.nanmin(dists, axis=1)
            if np.nanmin(min_dist) < cutoff:
                if len(np.where(min_dist < cutoff)[0]) > keep_bin_ratio*self.n:
                    print(mesh[i])
                    idx = np.where(dists < cutoff)
                    pockets.append(idx)
                    vol = []
                    for j in range(self.n):
                        if j in idx[0]:
                            k = np.where(idx[0] == j)[0][0]
                            vol.append(self.results[j].volume[idx[1][k]])
                        else:
                            vol.append('nan')
                    all_volumes.append(vol)
        all_volumes = np.array(all_volumes, dtype=np.float32)
        cov = np.zeros(shape=(all_volumes.shape[0], all_volumes.shape[0]))
        for i in range(all_volumes.shape[0]):
            for j in range(i):
                x = all_volumes[i]
                y = all_volumes[j]
                bad = ~np.logical_or(np.isnan(x), np.isnan(y))
                cov[i][j] = pearsonr(np.compress(bad, x), np.compress(bad, y))[0]
        groups = [0]
        for i in range(1, all_volumes.shape[0]):
            for j in range(i):
                FOUND = False
                if cov[i][j] > 0.8:
                    groups.append(groups[j])
                    FOUND = True
                    break
            if not FOUND:
                groups.append(np.max(groups) + 1)
        groups = np.array(groups)
        filtered_volumes = []
        for i in range(np.max(groups) + 1):
            filtered_volumes.append(np.nanmean(all_volumes[np.where(groups == i)[0], :], axis=0))
        filtered_volumes = np.array(filtered_volumes)
        if plot:
            f, ax = plt.subplots()
            x = range(self.n)
            for i in range(filtered_volumes.shape[0]):
                ax.plot(x, filtered_volumes[i], "o-", label="Cluster {:d}".format(i + 1))
            plt.xlabel("Frame")
            plt.ylabel("Volume / A$^3$")
            plt.legend(loc="best")
            plt.show()
        return

    def write_pml(self, fname="pyvol", stride=1, spec="helicase"):
        """
        This function was specific to the SARS-CoV-2 helicase visualization.
        The generalization is underway.
        """
        if spec == "helicase":
            tmpl = "/mnt/data/covid/pyvol/PCA/visualization_template.pse"
        with open(os.path.join(self.wrkdr, "{0:s}_{1:%Y%m%d%H%M}.pml".format(fname, datetime.now())), "w") as fout:
            if spec == "helicase":
                fout.write("load {:s}\n".format(tmpl))
            for i in range(0, self.n, stride):
                fout.write("load {0:s}, frame{1:03d}\n".format(self.frames[i], i))
                if spec == "helicase":
                    fout.write("super frame{:03d}, struct\n".format(i))
                else:
                    fout.write("super frame{:03d}, frame001\n".format(i))
                for j in range(len(self.pocket_names[i])):
                    fout.write("load {0:s}.xyz, frame{1:d}p{2:d}\n".format(os.path.join(self.folders[i],
                                                                                        self.pocket_names[i][j]), i, j))
                    fout.write(
                        "color cyan, frame{0:d}p{1:d}; hide everything, frame{0:d}p{1:d}; show surface, frame{0:d}p{1:d}\n".format(
                            i, j))
                    fout.write("matrix_copy frame{0:03d}, frame{0:d}p{1:d};\n".format(i, j))
            if spec == "helicase":
                fout.write("select domain1, frame* and i. 1-240\n")
                fout.write("select RecA1, frame* and i. 241-442\n")
                fout.write("select RecA2, frame* and i. 443-601\n")
                fout.write("color deepteal, domain1 and elem C\n")
                fout.write("color yellow, RecA1 and elem C\n")
                fout.write("color hotpink, RecA2 and elem C\n")
            fout.write("zoom frame*;\n")
            fout.write("mset; rewind\n")
            fout.write("mset 1x{:d};\n".format(int(self.n/stride)))
            k = 0
            for i in range(0, self.n, stride):
                enable_pockets = ""
                for j in range(len(self.pocket_names[i])):
                    enable_pockets += "enable frame{0:d}p{1:d};".format(i, j)
                fout.write("disable all; enable struct; enable frame{0:03d}; {1:s}scene s{0:03d}, store;\n".format(
                    i, enable_pockets))
                fout.write("mview store, {1:d}, scene=s{0:03d};\n".format(i, k-1))
                k += 1
        return

    def write_vis_pml(self, fname="pyvol_movie", stride=1):
        """
        This function is specific to the SARS-CoV-2 helicase visualization. Resaving movie frames after adjustment.
        """
        with open(os.path.join(self.wrkdr, "{:s}.pml".format(fname)), "w") as fout:
            fout.write("mset; rewind\n")
            fout.write("mset 1x{:d};\n".format(int(self.n/stride)))
            k = 0
            for i in range(0, self.n, stride):
                enable_pockets = ""
                for j in range(len(self.pocket_names[i])):
                    enable_pockets += "enable frame{0:d}p{1:d};".format(i, j)
                fout.write("disable all; enable struct; enable frame{0:03d}; {1:s}scene s{0:03d}, store;\n".format(
                    i, enable_pockets))
                fout.write("mview store, {1:d}, scene=s{0:03d};\n".format(i, k))
                k += 1
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

    def plot_selected(self, fout=None, label="ATP"):
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
        a.plot(np.arange(self.n), self.volumes, "o-", label=label)
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

    def follow_pocket(self, spec, identifiers=6, mode="single", stride=1):
        if self.ref_poc is None:
            raise ValueError("No pocket is selected to be tracked.")
        if self.pocket_IDs is None:
            self.process()
        for r in self.ref_poc:
            if identifiers == "all":
                PIDs = self.pocket_IDs
            else:
                PIDs = self.select_ID(r, identifiers, plot=False)
            ref = PIDs[r[0]][r[1]]
            dists = np.empty(shape=(self.n, PIDs.shape[1]), dtype=np.float32)
            for j in range(self.n):
                for k in range(PIDs.shape[1]):
                    dists[j][k] = dist(PIDs[j][k], ref)
            if mode == "single":
                min_dist = np.nanmin(dists, axis=1)
                sns.displot(min_dist[~np.isnan(min_dist)], kde=True, binwidth=0.3)
                plt.show()
            elif mode == "multiple":
                flat_dist = dists.flatten()
                sns.displot(flat_dist[~np.isnan(flat_dist)], kde=True, binwidth=0.3)
                plt.show()
            while True:
                ans = input("Please enter a distance cutoff:")
                try:
                    cutoff = float(ans)
                    break
                except ValueError:
                    pass
            # reinitialize selected pockets
            self.pocket_names = [[] for _ in range(self.n)]
            self.volumes = np.zeros(shape=self.n, dtype=np.float_)
            if mode == "single":
                for i in range(self.n):
                    if min_dist[i] < cutoff:
                        self.pocket_names[i].append(self.results[i].
                                                    name[np.where(dists[i] == np.nanmin(dists[i]))[0][0]])
                        self.volumes[i] = self.results[i].volume[np.where(dists[i] == np.nanmin(dists[i]))[0][0]]
            elif mode == "multiple":
                for i in range(self.n):
                    for j in range(dists.shape[1]):
                        if dists[i][j] < cutoff:
                            self.pocket_names[i].append(self.results[i].name[j])
                            self.volumes[i] += self.results[i].volume[j]
            # self.volumes = np.where(self.volumes == 0, 199, self.volumes)
            self.volumes = np.where(self.volumes > 10000, np.nan, self.volumes)
            self.write_pml(fname="pocket_tracking_{0:d}_{1:d}".format(r[0], r[1]), stride=stride, spec=spec)
            self.write_vis_pml(fname="pocket_vis_{0:d}_{1:d}".format(r[0], r[1]), stride=stride)
            self.plot_selected(fout="pocket_{0:d}_{1:d}_{2:.1f}.png".format(r[0], r[1], cutoff),
                               label="{0:d}_{1:d}_{2:.1f}".format(r[0], r[1], cutoff))
        return

    def select_ID(self, r, dim, plot=False):
        refID = self.pocket_IDs[r[0]][r[1]]
        idx = refID.argsort()[:dim]
        selected = self.pocket_IDs[:, :, idx]
        if plot:
            frame = md.load(self.frames[r[0]])
            refs = np.empty(shape=(len(self.ref_sel), 3), dtype=np.float32)
            for i in range(len(self.ref_sel)):
                if "CA" in self.ref_sel[i]:
                    refs[i] = frame.xyz[0][frame.top.select(self.ref_sel[i])[0]]
                else:
                    refs[i] = md.compute_center_of_mass(frame.atom_slice(frame.top.select(self.ref_sel[i])))
            refs = 10 * refs
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(refs[:, 0], refs[:, 1], refs[:, 2], s=50, alpha=0.5)
            ax.scatter(refs[idx, 0], refs[idx, 1], refs[idx, 2], c='red', s=50)
            plt.show()
        return selected


parser = argparse.ArgumentParser()
parser.add_argument("-p", "--pattern")
parser.add_argument("--fig")
parser.add_argument("-t", "--threads", type=int, default=4)
parser.add_argument("--ps", type=int, default=100, help="Stride for pml printing.")
parser.add_argument("-s", "--selected_pockets",
                    help="Zero based index pairs <WINDOW>;<POCKET> for selecting reference in a file")
parser.add_argument("-S", "--specific", default=None, help="Name a predefined topic specification.")
parser.add_argument("-m", "--mode", default="single", help="Pocket tracking mode {single|multiple}")
parser.add_argument("-l", "--load", help="Processed result-set file, i.e. 'ref_ca.p'")
args = parser.parse_args()
start = time.time()
# ref_selection = ["resSeq <= 240", "240 < resSeq <= 445", "445 < resSeq"]
# ref_selection = ["resSeq <= 100", "100 < resSeq <= 200", "200 < resSeq <= 300",
#                  "300 < resSeq <= 400", "400 < resSeq <= 500", "500 < resSeq"]
if args.pattern is not None:
    rs = ResultsSet(pattern=args.pattern)
else:
    rs = ResultsSet()
# rs.ref_sel = ref_selection
if args.load is None:
    rs.parse()
    rs.process()
else:
    rs = rs.load(fname=args.load)
rs.save(fname="ref_ca.p")
rs.parse_reference(args.selected_pockets)
rs.follow_pocket(args.specific, mode=args.mode, identifiers='all', stride=args.ps)

# rs.opt_cluster()
# rs.n_clusters = 10
# rs.plot_cluster()
# rs.identify_clusters(plot=True)

# rs.grid(plot=True)
# rs.opt_cutoff()
# rs.write_pml()
# rs.write_RNA_pml()
# rs.write_vis_pml()
# rs.plot_selected(args.fig)
end =time.time()
print(end - start)
