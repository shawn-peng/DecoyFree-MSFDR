import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from scipy import stats
import numpy as np
from . import skew_normal
# from . import normal_gpu
from . import normal
from .constraints import *
from .param_binary_search import *
import sympy
import time
from .myutils import *
# from plotter import *

from collections import defaultdict
from typing import *
import multiprocessing as mp

SN = skew_normal.SkewNormal
N = normal.Normal


class PlotWrapper:
    def __init__(self, func):
        # self.static = {}
        self.func = func

    def __call__(self, *args):
        cmd = args[0]
        if cmd == 'set_static':
            name, val = args[1:]
            self.static[name] = val
        elif cmd == 'plot':
            X = self.static['X']
            self.func(X, *args[1:])


def values_except(X, ex):
    x = X.reshape([-1])
    x = x[x != ex]
    return x


class MixtureModelBase:
    missing_value_placeholder = 0

    def __init__(self, plot_func=None, title=None, **kwargs):
        self.plot_func = plot_func
        # if not plot_pipe:
        #     plot_pipe, plotter_pipe = mp.Pipe()
        #     self.plotter = ProcessPlotter((self._plot))
        #     self.plot_process = mp.Process(
        #         target=self.plotter, args=(plotter_pipe,), daemon=True)
        #     self.plot_process.start()
        # self.plot_pipe = plot_pipe
        self.title = title
        self.plotting_X = None
        self.fig = None
        self.x_upper_bound_scale = 1.1

    def frozen(self):
        return AttrObj({
            **{
                k: deepcopy(self.__getattribute__(k))
                for k in ['binwidth', 'plotstep', 'n_samples', 'weights', 'comps', 'all_comps', 'starting_pos',
                          'xrange', 'plotting_xrange',
                          'cdf_curve', 'ecdf_curve',
                          'fdr_curve',
                          'delta_cdf', 'fdr_thres', 'title']
            }})

    def __del__(self):
        # self.plot_pipe.send(None)
        pass

    def to_json(self):
        model = {}
        for cname, dist in self.all_comps.items():
            param = {
                'mu': dist.mu,
                'sigma': dist.sigma,
                'lambda': dist.alpha,
            }
            model[cname] = param

        weights = []
        for i in range(self.n_samples):
            weights.append([])
            for cname, w in self.weights[i].items():
                weights[i][cname] = w.get()
        model['weights'] = weights
        return model

    @staticmethod
    def json_from_frozen(frozen_model):
        model = {}
        for cname, dist in frozen_model.all_comps.items():
            if cname == 'NA':
                continue
            param = {
                'mu':     dist.mu,
                'sigma':  dist.sigma,
                'lambda': dist.alpha,
            }
            model[cname] = param

        weights = []
        for i in range(frozen_model.n_samples):
            weights.append({})
            for cname, w in frozen_model.weights[i].items():
                weights[i][cname] = w.get()
        model['weights'] = weights
        return model

    def init_range(self, X):
        xmax = np.nanmax(X)
        xmin = np.nanmin(values_except(X, self.missing_value_placeholder))
        self.n_xsamples = 200
        xstep = (xmax - xmin) / self.n_xsamples
        # x = np.arange(xmin, xmax + self.plotstep, self.plotstep)
        self.xrange = np.arange(xmin, xmin + (xmax - xmin) * self.x_upper_bound_scale + xstep, xstep)

        xmax = np.nanmax(X)
        xmin = np.nanmin(np.quantile(values_except(X, self.missing_value_placeholder), 0.005))
        self.plotstep = (xmax - xmin) / self.plot_sample
        self.plotting_xrange = np.arange(xmin, xmax + self.plotstep, self.plotstep)

        self.binwidth = (xmax - xmin) / self.nbins
        self.X = X

    def cdf(self, x):
        res = np.zeros_like(x)
        for cname, cdist in self.comps[0].items():
            res += self.weights[0][cname] * cdist.cdf(x)
        return res

    @property
    def delta_cdf(self):
        # x = np.sort(self.X[:, 0])
        x = self.plotting_xrange
        cdf = self.cdf(x)
        ecdf = empirical_cdf(x)
        return np.trapz(np.abs(ecdf - cdf), x)

    @property
    def cons_satisfied(self):
        return self.check_constraints()

    @property
    def fdr_thres(self):
        return self.fdr_thres_score(self.xrange, 0.01)

    @property
    def fdr_curve(self):
        return self.fdr(self.plotting_xrange)

    @property
    def cdf_curve(self):
        return self.cdf(self.plotting_xrange)

    @property
    def ecdf_curve(self):
        x = np.sort(self.X[:, 0])
        ecdf = empirical_cdf(x)
        return np.interp(self.plotting_xrange, x, ecdf)

    # def plot(self, X, lls, slls, finished=False):
    #     send = self.plot_pipe.send
    #
    #     m = TimeMeter()
    #     if X is not self.plotting_X:
    #         self.plotting_X = X
    #         data = ('set_static_pos_arg', 0, copy(X))
    #         send(data)
    #     data = ('plot', None, lls, slls, self.frozen())  # None is a placeholder for static arg
    #     print('Sending plot data')
    #     send(data)
    #     print('done', m.read())
    def plot(self, X, lls, slls):
        if self.plot_func:
            return self.plot_func(X, lls, slls, self.frozen())
        if not self.fig:
            self.fig = plt.figure(figsize=(16, 9))
            plt.ion()
        fig = self.fig
        return self._plot(X, lls, slls, self.frozen(), fig)

    @staticmethod
    def _plot(X, lls, slls, frozen_model, fig=None):
        # print('frozen_model', frozen_model)
        # print('frozen_model starting_pos', frozen_model.starting_pos)
        # frozen_model = AttrObj(frozen_model)
        xmax = frozen_model.plotting_xrange[-1]
        xmin = frozen_model.plotting_xrange[0]
        # xmin = x.min()
        # xmax = x.max()
        # x = np.arange(xmin, xmax + frozen_model.plotstep, frozen_model.plotstep)
        x = frozen_model.plotting_xrange
        bins = np.arange(xmin, xmax + frozen_model.binwidth, frozen_model.binwidth)
        # distc = {}
        axs = []
        idmap = defaultdict(lambda: len(idmap))
        nsub = frozen_model.n_samples + (1 if lls is not None else 0)
        # fig.clf()
        if not fig.axes:
            if nsub == 1:
                axs = np.array([fig.subplots(nsub, 1)])
            else:
                axs = fig.subplots(nsub, 1)
        else:
            axs = fig.axes
            axs.pop().remove()

        """need to rescale the weights for second scores"""
        for i in range(len(frozen_model.comps)):
            if i == 0:
                reweight_scale = 1
            else:
                reweight_scale = 1 / (1 - frozen_model.weights[i]['NA'].get())

            # print(f'plotting score {i + 1}')
            # xi = X[:, i]
            # ax = fig.subplot(nsub, 1, i + 1, label=f's{i + 1}')
            ax = axs[i]
            ax.cla()
            yi = np.zeros_like(x, dtype=float)
            legends = []
            # cnts, _ = np.histogram(X[:, i], bins=bins, density=True)
            # ymax = cnts.max()
            for j, (cname, cdist) in enumerate(frozen_model.comps[i].items()):
                # ax.plot(cdist.pdf(x), linestyle='--')
                if cname == 'NA':
                    continue
                yj = frozen_model.weights[i][cname] * reweight_scale * cdist.pdf(x)
                yi += yj
                ax.plot(x, yj, c='C%d' % idmap[cname])
                legends.append(cname)
            ymax = yi.max()
            for j, (cname, cdist) in enumerate(frozen_model.comps[i].items()):
                if cname == 'NA':
                    continue
                scdist = frozen_model.starting_pos.comps[i][cname]
                # print(f'plotting component {cname}')
                str_params = f'{frozen_model.weights[i][cname]:.2f}' \
                             f' {cname}: m={cdist.mu:.2f}({scdist.mu:.2f}),' \
                             f' s={cdist.sigma:.2f}({scdist.sigma:.2f}),' \
                             f' a={cdist.alpha:.2f}({scdist.alpha:.2f})'
                ax.text(0.05, (1 - 0.1 * (j + 1)), str_params, transform=ax.transAxes)
            # print('draw mixture')
            ax.plot(x, yi, c='C%d' % idmap['mixture'])
            legends.append(f'mixture{i + 1}')
            # print('draw hist')
            """remove zeros when hist second scores"""
            if i == 0:
                ax.hist(X[:, i], bins=bins, density=True, facecolor='w', ec='k', lw=1)
            else:
                nonzero = X[:, i]
                nonzero = nonzero[nonzero != 0]
                ax.hist(nonzero, bins=bins, density=True, facecolor='w', ec='k', lw=1)

            # fig.bars()
            # h = info.hist[i]
            # fig.bar(h[1][:-1], h[0], facecolor='none', edgecolor='k')
            # print('draw texts')
            ax.text(0.76, 0.9, f'll = {slls[i]:.5f}', transform=ax.transAxes)
            ax.text(0.76, 0.8, f'$\delta_{{cdf}}$ = {frozen_model.delta_cdf:.5f}', transform=ax.transAxes)
            # print('draw legends')
            ax.legend(legends)

        if lls is not None:
            """ Plot ll """
            # ax = fig.subplot(nsub, 1, nsub, label=f'll_curve')
            ax = axs[-1]
            ax.cla()
            ax.plot(lls[1:])
            if lls:
                ax.text(0.05, 0.9, f'll = {lls[-1]:.5f}', transform=ax.transAxes)

        """ Plot FDR & Title """
        ax = fig.axes[0]
        blended_transform = transforms.blended_transform_factory(
            ax.transData,  # Data coordinates for x
            ax.transAxes   # Axes coordinates for y
        )
        # plt.axes(ax)
        try:
            fdr1 = frozen_model.fdr_thres
            ax.axvline(fdr1)
            ax.text(fdr1, 0.1, '$\leftarrow$ 1% FDR threshold', transform=blended_transform)

            ax2 = ax.twinx()
            ax2.cla()
            ax2.set_ylim(0, 1)

            cmap = plt.get_cmap('Set2')
            ax2.plot(x, frozen_model.cdf_curve, c=cmap(1))
            ax2.plot(x, frozen_model.ecdf_curve, c=cmap(2))
            ax2.plot(x, frozen_model.fdr_curve, c=cmap(3))
        except:
            print('failed to plot fdr')

        if frozen_model.title:
            ax.set_title(frozen_model.title)
        plt.pause(0.01)

    def log_likelihood(self, X):
        # ll = np.log(self.likelihood(X))
        # ll = list(map(np.log, self.likelihood(X)))
        # for lli in ll:
        #     lli[np.isinf(lli)] = lli[~np.isinf(lli)].min()
        # return np.mean(list(map(lambda x: x.mean(), ll)))
        return np.mean(self.sep_log_likelihood(X))

    def sep_log_likelihood(self, X):
        ll = list(map(np.log, self.likelihood(X)))
        for lli in ll:
            lli[np.isinf(lli)] = lli[~np.isinf(lli)].min()
        return list(map(lambda x: x.mean(), ll))

    def likelihood(self, X):
        X = np.array(X)
        n, d = X.shape
        assert d == 2
        # pj = list(map(lambda c: np.zeros((n, len(c))), self.comps))
        pj = []
        p = []
        for i in range(len(self.comps)):
            ws = self.weights[i]
            if i == 0:
                reweight_scale = 1
                nonzero = X[:, i]
            else:
                reweight_scale = 1 / (1 - self.weights[i]['NA'].get())
                nonzero = X[:, i][X[:, i] != -10000]
            pj.append(np.zeros((len(nonzero), len(self.comps[i]))))
            for j, (cname, cdist) in enumerate(self.comps[i].items()):
                # pj[i][:, j] = ws[cname] * cdist.pdf(X[:, i])
                if cname == 'NA':
                    continue
                pj[i][:, j] = ws[cname] * reweight_scale * cdist.pdf(nonzero)
            p0 = np.sum(pj[i], 1)
            p.append(p0)
        return p
