import time, tqdm
from pathlib import Path
import numpy as np
from scipy import signal, spatial, stats

from skimage import measure
from matplotlib.collections import LineCollection

import matplotlib as mpl

mpl.rcParams["path.simplify"] = False
mpl.rcParams["agg.path.chunksize"] = 20000  # or 0 to disable

from matplotlib import (
    pyplot as plt,
    rc,
    colors as mcolors,
    patches as mppatches,
    lines as mplines,
    cm,
)

from .neuron_tracking import Tracking

from catan.core.plot_utils import (
    plot_with_confidence,
    # get_shift_and_flow,
    # build_remap_from_shift_and_flow,
)


class TrackingAnalysis(Tracking):
    """
    this specifies the different plot functions:

      all plots have inputs sv, suffix to specify saving behavior

    #   1. plot_fit_results
    #       inputs:
    #         model
    #         times
    #       creates interactive plot of joint model results

      2. plot_model
          creates general overview of model results and matching performance compared to guess based on nearest neighbours

      3. plot_fit_parameters
        MERGE THIS INTO #1

      4. plot_count_histogram
          inputs:
            times
          plots the histogram for different populations in 2- and 3D

      5. plot_something
          plots 3D visualization of matching probability

      6. plot_matches
          inputs:
            s_ref
            s
          plots neuron footprints of 2 sessions, colorcoded by whether they are matched, or not

      7. plot_neuron_numbers
          shows sessions in which each neuron is active
          ADJUST ACCORDING TO PLOTTING SCRIPT TO SHOW PHD LIKE FIGURE

      8. plot_registration
          shows distribution of match probabilities and 2nd best probability per match (how much "confusion" could there be?)
    """

    def calculate_RoC(self, steps, times=0):
        # key_counts = 'counts' if self.params['correlation_model']=='shifted' else 'counts_unshifted'

        counts = self.scale_counts(times)
        X, Y = np.meshgrid(
            self.params["arrays"]["distance"], self.params["arrays"]["correlation"]
        )

        p_steps = np.linspace(0, 1, steps + 1)

        rates = {"tp": {}, "tn": {}, "fp": {}, "fn": {}, "cumfrac": {}}

        for key in rates.keys():
            rates[key] = {
                "joint": np.zeros(steps),
                "distance": np.zeros(steps),
                "correlation": np.zeros(steps),
            }

        nTotal = counts[..., 0].sum()
        for key in ["joint", "distance", "correlation"]:
            if key == "joint":
                f_same = self.get_f_same("joint")
                p_same = self.model["p_same"]["joint"]
            elif key == "distance":
                f_same = self.get_f_same("distance")
                p_same = f_same(self.params["arrays"]["distance"])
            elif key == "correlation":
                f_same = self.get_f_same("correlation")
                p_same = f_same(self.params["arrays"]["correlation"])
            else:
                raise ValueError(f"Invalid key: {key}")

            for i, p in enumerate(p_steps[:-1]):

                idxes_negative = p_same < p
                idxes_positive = p_same >= p

                if key == "joint":

                    tp = counts[idxes_positive, 1].sum()
                    tn = counts[idxes_negative, 2].sum()
                    fp = counts[idxes_positive, 2].sum()
                    fn = counts[idxes_negative, 1].sum()

                    # print(f"p_thr: {p:.2f}, TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
                    rates["cumfrac"]["joint"][i] = (
                        counts[idxes_negative, 0].sum() / nTotal
                    )
                elif key == "distance":

                    tp = counts[idxes_positive, :, 1].sum()
                    tn = counts[idxes_negative, :, 2].sum()
                    fp = counts[idxes_positive, :, 2].sum()
                    fn = counts[idxes_negative, :, 1].sum()

                    rates["cumfrac"]["distance"][i] = (
                        counts[idxes_negative, :, 0].sum() / nTotal
                    )
                else:

                    tp = counts[:, idxes_positive, 1].sum()
                    tn = counts[:, idxes_negative, 2].sum()
                    fp = counts[:, idxes_positive, 2].sum()
                    fn = counts[:, idxes_negative, 1].sum()

                    rates["cumfrac"]["correlation"][i] = (
                        counts[:, idxes_negative, 0].sum() / nTotal
                    )

                rates["tp"][key][i] = tp / (fn + tp)
                rates["tn"][key][i] = tn / (fp + tn)
                rates["fp"][key][i] = fp / (fp + tn)
                rates["fn"][key][i] = fn / (fn + tp)

        return p_steps, rates

    def find_confusion_candidates(
        self, confusion_distance=5, ct_max=3, use_plotly=True
    ):

        import ipywidgets as widgets

        # import anywidget

        clusters = getattr(self, self.cluster_field)

        cm_mean = np.nanmean(clusters["cm"], axis=1)
        cm_dists = spatial.distance.squareform(spatial.distance.pdist(cm_mean))

        confusion_candidates = np.where(
            np.logical_and(cm_dists > 0, cm_dists < confusion_distance)
        )

        print(confusion_candidates)
        print(len(confusion_candidates[0]) // 2, " candidates found")

        # Filter and prepare candidates for slider
        valid_candidates = []
        for i, j in zip(*confusion_candidates):
            if j > i:
                continue
            assignments = clusters["assignments"][(i, j), :].T
            confused_sessions = np.where(np.all(assignments >= 0, axis=1))[0]
            valid_candidates.append((i, j, assignments, confused_sessions))

        print(len(valid_candidates), " valid candidates after filtering")
        if len(valid_candidates) == 0:
            print("No valid candidates found")
            return

        # Create initial figure with first candidate
        if len(valid_candidates) == 0:
            return None

        if use_plotly:
            fig = go.FigureWidget()

        else:
            fig_ = plt.figure(figsize=(6, 6), dpi=150)
            fig = fig_.add_subplot(111, projection="3d")

        slider = widgets.IntSlider(
            min=0,
            max=len(valid_candidates) - 1,
            value=0,
            step=1,
            description="Candidate",
        )
        debug = widgets.Output()

        # Generate plot for first candidate
        def generate_candidate_plot(update):

            candidate_idx = update["new"]
            print(f"slider moved to {candidate_idx}")
            i, j, assignments, confused_sessions = valid_candidates[candidate_idx]
            occ = assignments >= 0
            nOcc = occ.sum(axis=0)

            if use_plotly:
                with debug:
                    print(f"input data from slider update: {update}")
                    print(
                        f"Candidate {candidate_idx}: Clusters {i}, {j} | Occurrences: {nOcc} | Confused sessions: {confused_sessions}"
                    )

                with fig.batch_update():
                    fig.data = []  # Clear previous data

                    fig.update_layout(
                        # title=f"candidate idx {candidate_idx}",  # | Confused sessions: {confused_count}",
                        title=f"Clusters {i}, {j} | Occurrences: {nOcc}",  # | Confused sessions: {confused_count}",
                        height=700,
                    )

            # Plot footprints (add their traces to the figure)
            self.plot_footprints(i, fp_color="black", ax_in=fig, use_plotly=use_plotly)
            col = "green" if len(confused_sessions) == 0 else "red"
            self.plot_footprints(
                j,
                fp_color=col,
                ax_in=fig,
                use_plotly=use_plotly,
            )

            # return fig
            # return nOcc, i, j, len(confused_sessions), assignments

        # if use_plotly:
        slider.observe(generate_candidate_plot, names="value")
        # Create initial plot with first candidate
        generate_candidate_plot({"new": 0})

        # # Create slider steps
        # steps = []
        # for idx in range(len(valid_candidates)):
        #     i, j, _, _ = valid_candidates[idx]
        #     step = {
        #         "args": [
        #             [idx],
        #             {
        #                 "frame": {"duration": 0, "redraw": True},
        #                 "mode": "immediate",
        #                 "transition": {"duration": 0},
        #             },
        #         ],
        #         "label": f"Candidate {idx}: Clusters {i}, {j}",
        #         "method": "skip",
        #     }
        #     steps.append(step)

        # sliders = [
        #     {
        #         "active": 0,
        #         "steps": steps,
        #         "x": 0.1,
        #         "len": 0.9,
        #         "xanchor": "left",
        #         "y": 0,
        #         "yanchor": "top",
        #     }
        # ]

        # fig.update_layout(sliders=sliders)

        # Add custom callback using Dash or update via relayout
        if use_plotly:
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=[],
                    )
                ]
            )
            display(slider, fig, debug)
        else:
            display(slider, debug)
        # Store reference to generate_candidate_plot for dynamic updates
        # ax.generate_candidate_plot = generate_candidate_plot
        # ax.valid_candidates = valid_candidates

        # fig.show()

        # return fig

    # def plot_fit_results(self, sv=False, suffix="", times=0):

    #     arrays = self.params["arrays"]

    #     fig = plt.figure(figsize=(6, 4), dpi=150)
    #     ax_r = fig.add_subplot(221)
    #     ax_r.plot(
    #         arrays["distance"],
    #         self.model["pdf"]["distance_same"] * self.model["parameters"]["p_same"],
    #         color="tab:green",
    #         label="f_r_same",
    #     )
    #     ax_r.plot(
    #         arrays["distance"],
    #         self.model["pdf"]["distance_diff"]
    #         * (1 - self.model["parameters"]["p_same"]),
    #         color="tab:red",
    #         label="f_r_diff",
    #     )
    #     ax_r_p_same = ax_r.twinx()
    #     ax_r_p_same.plot(
    #         arrays["distance"],
    #         self.model["p_same"]["distance"],
    #         color="tab:blue",
    #         label="p_same",
    #     )
    #     ax_r.legend()
    #     ax_r.spines[["right", "top"]].set_visible(False)

    #     ax_c = fig.add_subplot(222)
    #     ax_c.plot(
    #         arrays["correlation"],
    #         self.model["pdf"]["correlation_same"] * self.model["parameters"]["p_same"],
    #         color="tab:green",
    #         label="f_c_same",
    #     )
    #     ax_c.plot(
    #         arrays["correlation"],
    #         self.model["pdf"]["correlation_diff"]
    #         * (1 - self.model["parameters"]["p_same"]),
    #         color="tab:red",
    #         label="f_c_diff",
    #     )
    #     ax_c_p_same = ax_c.twinx()
    #     ax_c_p_same.plot(
    #         arrays["correlation"],
    #         self.model["p_same"]["correlation"],
    #         color="tab:blue",
    #         label="p_same",
    #     )
    #     ax_c.legend()

    #     ax_fsame = fig.add_subplot(224, projection="3d")
    #     # CC, RR = np.meshgrid(arrays["correlation"], arrays["distance"])
    #     CC, RR = np.meshgrid(
    #         arrays["correlation"][self.params["nbins"] // 2 :],
    #         arrays["distance"][: self.params["nbins"] // 2],
    #     )
    #     ax_fsame.plot_surface(
    #         CC,
    #         RR,
    #         self.model["p_same"]["joint"][
    #             : self.params["nbins"] // 2, self.params["nbins"] // 2 :
    #         ],
    #         cmap=plt.cm.RdYlGn,
    #     )
    #     ax_fsame.view_init(elev=30, azim=40)
    #     plt.setp(ax_fsame, xlabel="correlation", ylabel="distance", zlabel="$p_{same}$")
    #     plt.tight_layout()
    #     plt.show()

    #     # fig = plt.figure(figsize=(3, 2.75), dpi=300)
    #     # ax_fsame = fig.add_subplot(111, projection="3d")
    #     # # CC, RR = np.meshgrid(arrays["correlation"], arrays["distance"])
    #     # CC, RR = np.meshgrid(
    #     #     arrays["correlation"][self.params["nbins"] // 2 :],
    #     #     arrays["distance"][: self.params["nbins"] // 2],
    #     # )
    #     # ax_fsame.plot_surface(
    #     #     CC,
    #     #     RR,
    #     #     self.model["p_same"]["joint"][
    #     #         : self.params["nbins"] // 2, self.params["nbins"] // 2 :
    #     #     ],
    #     #     cmap=plt.cm.RdYlGn,
    #     # )
    #     # ax_fsame.view_init(elev=20, azim=140)
    #     # plt.setp(
    #     #     ax_fsame,
    #     #     xlabel="similarity",
    #     #     ylabel="distance",
    #     #     zlabel="$p_{same}$",
    #     #     zticks=[],
    #     # )

    #     # plt.tight_layout(w_pad=0.2, h_pad=0.2)
    #     # plt.show()

    def plot_model(self, sv=False, suffix="", times=0):

        rc("font", size=10)
        rc("axes", labelsize=12)
        rc("xtick", labelsize=8)
        rc("ytick", labelsize=8)

        counts = self.scale_counts(times)
        nbins = self.params["nbins"]

        arrays = self.params["arrays"]
        X, Y = np.meshgrid(arrays["correlation"], arrays["distance"])

        fig = plt.figure(figsize=(8, 4), dpi=150)
        ax_phase = fig.add_subplot((0.3, 0.13, 0.2, 0.4))
        # add_number(fig, ax_phase, order=1, offset=[-250, 200])
        # ax_phase.imshow(self.model[key_counts][:,:,0],extent=[0,1,0,self.params['neighbor_distance']],aspect='auto',clim=[0,0.25*self.model[key_counts][:,:,0].max()],origin='lower')
        cmap = plt.cm.RdYlGn

        ### --------------------------------------------- ###
        ### ------------- NN ratio plot ----------------- ###
        ### --------------------------------------------- ###
        NN_ratio = counts[:, :, 1] / counts[:, :, 0]
        NN_ratio = cmap(NN_ratio)
        NN_ratio[..., -1] = np.minimum(counts[..., 0] / (np.max(counts) / 5.0), 1)

        im_ratio = ax_phase.imshow(
            NN_ratio,
            extent=[
                *arrays["correlation_bounds"][[0, -1]],
                *arrays["distance_bounds"][[0, -1]],
            ],  # type: ignore
            aspect="auto",
            clim=[0, 0.5],
            origin="lower",
        )
        nlev = 3
        # col = (np.ones((nlev,3)).T*np.linspace(0,1,nlev)).T
        p_levels = ax_phase.contour(
            X,
            Y,
            self.model["p_same"]["joint"],
            levels=[0.05, 0.5, 0.95],
            colors="k",
            linestyles=[":", "--", "-"],
            linewidths=1.0,
        )
        plt.setp(
            ax_phase,
            # xlim=arrays["correlation_bounds"][[0, -1]],
            xlim=[0.5,1],
            ylim=arrays["distance_bounds"][[0, -1]],
            xlabel="correlation",
            ylabel="distance",
        )
        ax_phase.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=True,
            labelbottom=False,
            labeltop=True,
        )
        ax_phase.tick_params(
            axis="y",
            which="both",
            left=True,
            right=True,
            labelright=False,
            labelleft=True,
        )
        ax_phase.yaxis.set_label_position("right")
        ax_phase.xaxis.set_label_coords(0.5, -0.15)
        ax_phase.yaxis.set_label_coords(1.15, 0.5)

        im_ratio.cmap = cmap

        cbaxes = fig.add_subplot((0.32, 0.47, 0.07, 0.03))
        cbar = plt.colorbar(im_ratio, cax=cbaxes, orientation="horizontal")
        plt.setp(cbar.ax, xticks=[0, 0.5], xticklabels=["nNN", "NN"])

        ### --------------------------------------------- ###
        ### ----------- distance distr plot ------------- ###
        ### --------------------------------------------- ###
        ax_dist = fig.add_subplot((0.05, 0.13, 0.2, 0.4))

        distance_step = np.diff(arrays["distance_bounds"])[0]
        ax_dist.barh(
            arrays["distance"],
            counts[..., 0].sum(1),
            distance_step,
            facecolor="k",
            alpha=0.7,
            orientation="horizontal",
        )
        ax_dist.barh(
            arrays["distance"],
            counts[..., 2].sum(1),
            distance_step,
            facecolor="salmon",
            alpha=0.7,
        )
        ax_dist.barh(
            arrays["distance"],
            counts[..., 1].sum(1),
            distance_step,
            facecolor="lightgreen",
            alpha=0.7,
        )
        ax_dist.invert_xaxis()
        # h_d_move = ax_dist.bar(arrays['distance'],np.zeros(nbins),arrays['distance_step'],facecolor='k')

        N_same = self.model["parameters"]["p_same"] * counts[..., 0].sum()
        N_diff = (1 - self.model["parameters"]["p_same"]) * counts[..., 0].sum()

        dr = arrays["distance_bounds"][-1] / self.params["nbins"]

        ax_dist.plot(
            self.model["pdf"]["distance_same"] * dr * N_same
            + self.model["pdf"]["distance_diff"] * dr * N_diff,
            arrays["distance"],
            "k",
            alpha=0.7,
        )
        ax_dist.plot(
            self.model["pdf"]["distance_same"] * dr * N_same,
            arrays["distance"],
            "tab:green",
            ls=":",
        )
        ax_dist.plot(
            self.model["pdf"]["distance_diff"] * dr * N_diff,
            arrays["distance"],
            "tab:red",
            ls=":",
        )

        ax_dist_p_same = ax_dist.twiny()
        ax_dist_p_same.plot(
            self.model["p_same"]["distance"],
            arrays["distance"],
            color="tab:blue",
            label="p_same",
        )
        plt.setp(ax_dist_p_same, xlabel="$p_{same}$", xlim=[1.1, 0])
        ax_dist_p_same.spines[["left","bottom"]].set_visible(False)

        plt.setp(ax_dist, ylim=[0, self.params["neighbor_distance"]], xlabel="counts")
        ax_dist.spines[["top","left"]].set_visible(False)
        ax_dist.tick_params(
            axis="y",
            which="both",
            left=False,
            right=True,
            labelright=False,
            labelleft=False,
        )

        ### --------------------------------------------- ###
        ### ----------- correlation distr plot ---------- ###
        ### --------------------------------------------- ###
        ax_corr = plt.axes([0.3, 0.63, 0.2, 0.325])
        ax_corr.bar(
            arrays["correlation"],
            counts[..., 0].sum(0),
            1 / nbins,
            facecolor="k",
            alpha=0.7,
        )
        ax_corr.bar(
            arrays["correlation"],
            counts[..., 2].sum(0),
            1 / nbins,
            facecolor="salmon",
            alpha=0.7,
        )
        ax_corr.bar(
            arrays["correlation"],
            counts[..., 1].sum(0),
            1 / nbins,
            facecolor="lightgreen",
            alpha=0.7,
        )

        dc = arrays["correlation_bounds"][-1] / self.params["nbins"]
        # p_same = self.model["parameter"]["p_same"] #counts[..., 1].sum()
        # norm_same = dc * self.model["parameter"]["p_same"] * counts[..., 0].sum()
        # norm_diff = dc * (1 - self.model["parameter"]["p_same"]) * counts[..., 0].sum()
        ax_corr.plot(
            arrays["correlation"],
            self.model["pdf"]["correlation_same"] * dc * N_same
            + self.model["pdf"]["correlation_diff"] * dc * N_diff,
            "k",
            alpha=0.7,
        )
        ax_corr.plot(
            arrays["correlation"],
            self.model["pdf"]["correlation_same"] * dc * N_same,
            "tab:green",
        )
        ax_corr.plot(
            arrays["correlation"],
            self.model["pdf"]["correlation_diff"] * dc * N_diff,
            "tab:red",
        )

        ax_corr.set_ylabel("counts")
        ax_corr.set_xlim([0.5, 1])
        ax_corr.spines["right"].set_visible(False)
        ax_corr.spines["top"].set_visible(False)
        ax_corr.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=False,
            labeltop=False,
        )

        ax_corr_p_same = ax_corr.twinx()
        ax_corr_p_same.plot(
            arrays["correlation"],
            self.model["p_same"]["correlation"],
            color="tab:blue",
            label="p_same",
        )
        plt.setp(ax_corr_p_same, ylabel="$p_{same}$", ylim=[0,1.1])
        ax_corr_p_same.spines[["left","top"]].set_visible(False)

        ### =============================================== ###
        ### ===================== p same ================== ###
        ### =============================================== ###

        ax_fsame = fig.add_subplot(133, projection="3d")
        # CC, RR = np.meshgrid(arrays["correlation"], arrays["distance"])
        CC, RR = np.meshgrid(
            arrays["correlation"][self.params["nbins"] // 2 :],
            arrays["distance"][: self.params["nbins"] // 2],
        )
        ax_fsame.plot_surface(
            CC,
            RR,
            self.model["p_same"]["joint"][
                : self.params["nbins"] // 2, self.params["nbins"] // 2 :
            ],
            cmap=plt.cm.RdYlGn,
        )
        ax_fsame.view_init(elev=30, azim=40)
        plt.setp(ax_fsame, xlabel="correlation", ylabel="distance", zlabel="$p_{same}$")
        
        
        # ### --------------------------------------------- ###
        # ### ---------- RoC & further stats plot --------- ###
        # ### --------------------------------------------- ###
        # p_steps, rates = self.calculate_RoC(100)

        # ax_cum = plt.axes([0.675, 0.7, 0.3, 0.225])
        # # add_number(fig, ax_cum, order=2)

        # uncertain = {}
        # idx_low = np.where(p_steps > 0.05)[0][0]
        # idx_high = np.where(p_steps < 0.95)[0][-1]
        # for key in rates["cumfrac"].keys():

        #     if key == "joint" and (rates["cumfrac"][key][idx_low] > 0.01) & (
        #         rates["cumfrac"][key][idx_high] < 0.99
        #     ):
        #         ax_cum.fill_between(
        #             [rates["cumfrac"][key][idx_low], rates["cumfrac"][key][idx_high]],
        #             [0, 0],
        #             [1, 1],
        #             facecolor="y",
        #             alpha=0.5,
        #         )
        #     uncertain[key] = (
        #         rates["cumfrac"][key][idx_high] - rates["cumfrac"][key][idx_low]
        #     )  # /(1-rates['cumfrac'][key][idx_high+1])

        # # print(uncertain)
        # # print(rates["cumfrac"]["joint"])
        # # print(p_steps)
        # ax_cum.axhline(0.05, color="b", linestyle=":")
        # ax_cum.axhline(0.95, color="b", linestyle="-")

        # ax_cum.plot(
        #     np.append(rates["cumfrac"]["joint"], 1),
        #     p_steps,
        #     "grey",
        #     label="Joint",
        # )

        # plt.setp(ax_cum, xlim=[0, 1], ylabel="$p_{same}$", xlabel="cumulative fraction")
        # ax_cum.spines[["right", "top"]].set_visible(False)

        # ax_uncertain = plt.axes((0.75, 0.825, 0.05, 0.1))
        # ax_uncertain.bar(3, uncertain["joint"], facecolor="k")
        # plt.setp(ax_uncertain, xticks=[], xticklabels=[])
        # ax_uncertain.spines[["right", "top"]].set_visible(False)
        # ax_uncertain.set_title("uncertain fraction", fontsize=10)

        # idx = np.where(p_steps == 0.3)[0]

        # ax_RoC = plt.axes((0.675, 0.13, 0.125, 0.3))
        # # add_number(fig, ax_RoC, order=3)

        # key_model = "joint"
        # ax_RoC.plot(rates["fp"][key_model], rates["tp"][key_model], "k", label="Joint")
        # ax_RoC.plot(rates["fp"][key_model][idx], rates["tp"][key_model][idx], "kx")

        # plt.setp(ax_RoC, ylabel="true positive", xlabel="false positive")
        # ax_RoC.spines[["right", "top"]].set_visible(False)

        # ax_fp = plt.axes((0.925, 0.13, 0.05, 0.1))
        # ax_fp.bar(1, rates["fp"]["distance"][idx], facecolor="k")
        # ax_fp.bar(2, rates["fp"]["correlation"][idx], facecolor="k")
        # ax_fp.bar(3, rates["fp"][key_model][idx], facecolor="k")

        # plt.setp(ax_fp, xlim=[0.5, 3.5], ylim=[0, 0.05], xticks=[], xticklabels=[])

        # # ax_fp.set_xticklabels(['Dist.','Joint'],rotation=60,fontsize=10)

        # ax_fp.spines[["right", "top"]].set_visible(False)
        # ax_fp.set_ylabel("false pos.", fontsize=10)

        # ax_tp = plt.axes((0.925, 0.33, 0.05, 0.1))
        # # add_number(fig, ax_tp, order=4, offset=[-100, 25])
        # # ax_tp.bar(2,rates['tp']['distance'][idx],facecolor='k')
        # ax_tp.bar(1, rates["tp"]["distance"][idx], facecolor="k")
        # ax_tp.bar(2, rates["tp"]["correlation"][idx], facecolor="k")
        # ax_tp.bar(3, rates["tp"][key_model][idx], facecolor="k")

        # plt.setp(ax_tp, xlim=[0.5, 3.5], ylim=[0.4, 1], xticks=[], xticklabels=[])
        # ax_tp.spines[["right", "top"]].set_visible(False)
        # # ax_tp.set_ylabel('fraction',fontsize=10)
        # ax_tp.set_ylabel("true pos.", fontsize=10)

        plt.tight_layout()
        plt.show(block=False)
        if sv:
            ext = "png"
            path = Path(
                self.params["pathMouse"],
                f"Sheintuch_matching_{self.params['correlation_model']}{suffix}.{ext}",
            )
            plt.savefig(path, format=ext, dpi=150)

    def plot_count_histogram(self, times=0):

        counts = self.scale_counts(times).astype(np.float32)
        arrays = self.params["arrays"]

        plt.figure(figsize=(6, 4), dpi=150)
        ax = plt.subplot(224, projection="3d")
        X, Y = np.meshgrid(arrays["correlation"], arrays["distance"])
        NN_ratio = counts[:, :, 1] / counts[:, :, 0]
        cmap = plt.cm.RdYlGn
        NN_ratio = cmap(NN_ratio)
        ax.plot_surface(X, Y, counts[:, :, 0], facecolors=NN_ratio)
        ax.view_init(30, -120)

        plt.setp(ax, xlabel="correlation", ylabel="distance", zlabel="# pairs")

        im_opts = {
            "extent": [
                *self.params["arrays"]["correlation_bounds"][[0, -1]],
                *self.params["arrays"]["distance_bounds"][[0, -1]],
            ],
            "aspect": "auto",
            "origin": "lower",
        }
        title_opts = {
            "y": 1,
            "pad": -14,
            # "color": "white",
            "color": "black",
            "fontweight": "bold",
            "fontsize": 10,
        }
        counts[counts == 0] = np.nan
        ax2 = plt.subplot(223)
        ax2.imshow(counts[..., 0], **im_opts)
        ax2.set_title("all counts", **title_opts)

        ax3 = plt.subplot(221)
        ax3.imshow(counts[..., 1], **im_opts)
        ax3.set_title("nearest neighbours", **title_opts)

        ax4 = plt.subplot(222)
        ax4.imshow(counts[..., 2], **im_opts)
        ax4.set_title("non-nearest neighbours", **title_opts)
        # plt.tight_layout()
        plt.show(block=False)

    def plot_matches(
        self,
        s_ref,
        s,
        color_s_ref="coral",
        color_s="lightgreen",
        level=0.2,
        p_thr=[0.5, 0.3],
    ):
        """
        TODO:
        * rewrite function, such that it calculates and plots footprint matching for 2 arbitrary sessions (s,s_ref)
        * write function description and document code properly
        * optimize plotting, so it doesn't take forever
        """

        if isinstance(s_ref, int) and isinstance(s, int):
            print("using existing registration results...")

            # cluster_field = self.cluster_field

            ref_data = self.sessions[s_ref]
            A_ref = ref_data.A
            Cn_ref = ref_data.Cn

            this_data = self.sessions[s]
            A = this_data.A
            Cn = this_data.Cn

        else:
            raise NotImplementedError(
                "plotting matches for arbitrary sessions is not yet implemented"
            )
            assert isinstance(s_ref, (str, Path)) and isinstance(
                s, (str, Path)
            ), "sessions must be specified as paths if no existing registration results are used"

            cluster_field = "compare_clusters"

            self.reset_registration(storage_struct=cluster_field)

            ref_data = self.get_data(s_ref)
            self.register_neurons(from_data=ref_data, p_thr=p_thr)
            A_ref = ref_data.A
            Cn_ref = ref_data.Cn

            this_data = self.get_data(s, alignment_template=ref_data.Cn)
            self.register_neurons(from_data=this_data, p_thr=p_thr)
            A = this_data.A
            Cn = this_data.Cn

            s_ref, s = 0, 1

            # session_alignment = getattr(self, storage_struct[0])

        # print(this_data.nA, ref_data.nA)
        # clusters = getattr(self, cluster_field)

        dims = this_data.dims

        assignments = self.assignments
        # print(assignments)
        matched_c = np.all(assignments[:, (s_ref, s)] >= 0, axis=1)
        matched_ref = assignments[matched_c, s_ref]
        matched_this = assignments[matched_c, s]
        # print('matched: ',matched_c.sum())
        n_matched = matched_c.sum()

        non_matched_c = (assignments[:, s_ref] >= 0) & (assignments[:, s] < 0)
        non_matched_ref = assignments[non_matched_c, s_ref]
        # print('non_matched 1: ',non_matched_c.sum())
        n_non_matched_ref = non_matched_c.sum()

        non_matched_c = (assignments[:, s_ref] < 0) & (assignments[:, s] >= 0)
        non_matched_this = assignments[non_matched_c, s]
        # print('non_matched 1: ',non_matched_c.sum())
        n_non_matched_this = non_matched_c.sum()

        print(
            f"matched: {n_matched}, non-matched in ref: {n_non_matched_ref}, non-matched in this: {n_non_matched_this}"
        )

        print("plotting...")
        t_start = time.time()

        Cn_plt = np.zeros(dims + (3,))
        Cn_plt[..., 0] = normalize_array(Cn_ref)
        Cn_plt[..., 1] = normalize_array(Cn)

        fig = plt.figure(figsize=(8, 6), dpi=150)

        ax = fig.add_subplot(111)
        ax.imshow(Cn_plt, origin="lower")

        contour_opts = {
            "linewidths": 0.8,
            "alpha": 0.8,
            "colors": color_s_ref,
        }
        all_segs = []
        for a in A_ref[:, matched_ref].T:
            all_segs.extend(get_contours(a, level))

        lc = LineCollection(all_segs, **contour_opts)
        lc.set_rasterized(True)
        ax.add_collection(lc)

        all_segs = []
        for a in A_ref[:, non_matched_ref].T:
            all_segs.extend(get_contours(a, level))
        lc = LineCollection(all_segs, **contour_opts, linestyles="--")
        lc.set_rasterized(True)
        ax.add_collection(lc)

        print("first half done: %5.3f" % (time.time() - t_start))
        contour_opts["colors"] = color_s
        contour_opts["linewidths"] = 0.5

        all_segs = []
        for a in A[:, matched_this].T:
            all_segs.extend(get_contours(a, level))
        lc = LineCollection(all_segs, **contour_opts)
        lc.set_rasterized(True)
        ax.add_collection(lc)

        all_segs = []
        for a in A[:, non_matched_this].T:
            all_segs.extend(get_contours(a, level))
        # add_lc_chunks(ax, all_segs, chunk=100, **contour_opts)
        lc = LineCollection(all_segs, **contour_opts, linestyles="--")
        lc.set_rasterized(True)
        ax.add_collection(lc)

        for c in ax.collections:
            c.set_clip_on(False)
        ax.legend(
            handles=[
                mppatches.Patch(color=color_s_ref, label="reference session"),
                mppatches.Patch(color=color_s, label="session"),
                mplines.Line2D(
                    [0], [0], color="k", linestyle="-", label=f"matched ({n_matched})"
                ),
                mplines.Line2D(
                    [0],
                    [0],
                    color="k",
                    linestyle="--",
                    label=f"non-matched ({n_non_matched_ref}/{n_non_matched_this})",
                ),
            ],
            loc="lower left",
            framealpha=0.9,
        )
        ax.set_aspect("equal")
        ax.autoscale()
        plt.setp(ax, xlabel="x [px]", ylabel="y [px]")
        print("done. time taken: %5.3f" % (time.time() - t_start))
        # fig.savefig("debug.pdf", dpi=300)

        plt.show(block=False)

    def plot_neuron_numbers(self):

        ### plot occurence of neurons
        colors = [(1, 0, 0, 0), (1, 0, 0, 1)]
        RedAlpha = mcolors.LinearSegmentedColormap.from_list("RedAlpha", colors, N=2)
        colors = [(0, 0, 0, 0), (0, 0, 0, 1)]
        BlackAlpha = mcolors.LinearSegmentedColormap.from_list(
            "BlackAlpha", colors, N=2
        )

        # session_alignment = getattr(self, self.results[0])
        # clusters = getattr(self, self.cluster_field)

        session_status = self.classify_sessions()
        nS = session_status.sum()
        cluster_status = self.classify_components(border_margin=2.0)
        nC = cluster_status.sum()
        print(f"{nC} clusters, {nS} sessions")

        idxes = self.assignments >= 0

        fig = plt.figure(figsize=(6, 4), dpi=150)

        ax_oc = fig.add_subplot((0.1, 0.15, 0.25, 0.6))
        ax_oc2 = ax_oc.twinx()
        ax_oc.imshow(
            idxes & cluster_status[:, None],
            cmap=BlackAlpha,
            aspect="auto",
            interpolation="none",
        )

        ax_oc2.imshow(
            idxes & (~cluster_status[:, None]),
            cmap=RedAlpha,
            alpha=0.5,
            aspect="auto",
            interpolation="none",
        )
        # ax_oc.imshow(clusters['p_matched'],cmap='binary',aspect='auto')
        ax_oc.set_xlabel("session")
        ax_oc.set_ylabel("neuron ID")

        ax = fig.add_subplot([0.1, 0.75, 0.25, 0.2])
        ax.axhline(nC, color="k", lw=0.5, ls="--")
        ax.plot(
            range(len(session_status)),
            idxes.sum(0),
            "ro",
            alpha=0.2,
            markersize=0.5,
        )
        ax.plot(
            range(len(session_status)),
            (idxes & cluster_status[:, None]).sum(0),
            "ko",
            markersize=1,
        )
        plt.setp(
            ax,
            # xlabel="session",
            ylabel="# neurons",
            xlim=[0, nS],
            ylim=[0, nC * 1.05],
            xticks=[],
        )

        ax = plt.axes([0.35, 0.15, 0.1, 0.6])
        ax.plot(
            (idxes & cluster_status[:, None]).sum(1),
            range(len(cluster_status)),
            # np.linspace(0, nC, nC),
            "ko",
            markersize=0.5,
        )
        ax.invert_yaxis()
        plt.setp(ax, ylim=[nC, 0], yticks=[], xlabel="occurence")

        ax = plt.axes([0.35, 0.75, 0.1, 0.2])
        ax.hist(
            idxes.sum(1),
            range(len(session_status)),
            color="r",
            cumulative=True,
            density=True,
            histtype="step",
        )
        ax.hist(
            (idxes & cluster_status[:, None]).sum(1),
            range(len(session_status)),
            color="k",
            alpha=0.5,
            cumulative=True,
            density=True,
            histtype="step",
        )
        ax.set_xticks([])
        # ax.set_yticks([])
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylim([0, 1])
        # ax.set_ylabel('# neurons')
        ax.spines["top"].set_visible(False)
        # ax.spines['right'].set_visible(False)

        # ext = 'png'
        # path = pathcat([self.params['pathMouse'],'Figures/Sheintuch_registration_score_stats_raw_%s_%s.%s'%(self.params['correlation_model'],suffix,ext)])
        # plt.savefig(path,format=ext,dpi=300)

    def plot_p_match(self, suffix="", sv=False):

        rc("font", size=10)
        rc("axes", labelsize=12)
        rc("xtick", labelsize=8)
        rc("ytick", labelsize=8)

        # clusters = getattr(self, self.cluster_field)

        idxes = self.assignments >= 0

        fig = plt.figure(figsize=(4, 1.5), dpi=150)
        ax_sc1 = fig.add_subplot((0.1, 0.3, 0.35, 0.65))

        ax = ax_sc1.twinx()
        ax.hist(
            self.tracking["p_matched"][idxes, 1].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:red",
            alpha=0.3,
        )
        # ax.invert_yaxis()
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

        ax = ax_sc1.twiny()
        ax.hist(
            self.tracking["p_matched"][idxes, 0].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            orientation="horizontal",
            alpha=0.3,
        )
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)

        ax_sc1.plot(
            self.tracking["p_matched"][idxes, 1].flat,
            self.tracking["p_matched"][idxes, 0].flat,
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc1.plot([0, 1], [0, 1], "--", color="tab:red", lw=0.5)
        ax_sc1.plot([0, 0.45], [0.5, 0.95], "--", color="tab:orange", lw=1)
        ax_sc1.plot([0.45, 1], [0.95, 0.95], "--", color="tab:orange", lw=1)
        plt.setp(
            ax_sc1,
            ylim=[0.5, 1],
            ylabel="$p^{\\asterisk}$",
            xlabel="max($p\\backslash p^{\\asterisk}$)",
        )
        ax_sc1.spines[["top", "right"]].set_visible(False)

        # match vs max
        # idxes &= idx_pm

        # avg matchscore per cluster, min match score per cluster, ...
        ax_sc2 = plt.axes([0.6, 0.3, 0.35, 0.65])
        # plt.hist(np.nanmean(clusters['p_matched'],1),np.linspace(0,1,51))
        ax = ax_sc2.twinx()
        ax.hist(
            np.nanmin(self.tracking["p_matched"][..., 0], 1),
            np.linspace(0, 1, 51),
            facecolor="tab:red",
            alpha=0.3,
        )
        ax.set_yticks([])
        ax.spines[["top", "right"]].set_visible(False)

        ax = ax_sc2.twiny()
        ax.hist(
            np.nanmean(self.tracking["p_matched"][..., 0], axis=1),
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            orientation="horizontal",
            alpha=0.3,
        )
        ax.set_xticks([])
        ax.spines[["top", "right"]].set_visible(False)

        ax_sc2.plot(
            np.nanmin(self.tracking["p_matched"][..., 0], 1),
            np.nanmean(self.tracking["p_matched"][..., 0], axis=1),
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc2.set_ylim(0.5, 1)
        ax_sc2.set_xlabel("min($p^{\\asterisk}$)")
        ax_sc2.set_ylabel("$\\left\\langle p^{\\asterisk} \\right\\rangle$")
        ax_sc2.spines[["top", "right"]].set_visible(False)

        ### plot positions of neurons
        # plt.tight_layout()
        plt.show(block=False)

        if sv:
            ext = "png"
            path = Path(
                self.params["pathMouse"],
                f"Figures/Sheintuch_registration_score_stats_{self.params['correlation_model']}_{suffix}.{ext}",
            )
            plt.savefig(path, format=ext, dpi=300)

    def plot_cluster_stats(self):

        print("### Plotting ROI and cluster statistics of matching ###")

        # clusters = getattr(self, self.cluster_field)

        nC, nSes = self.assignments.shape
        active = self.assignments >= 0
        # print(active)

        centroids = self.build_centroids()

        idx_unsure = self.tracking["p_matched"][..., 0] < 0.95

        fig = plt.figure(figsize=(7, 4), dpi=150)

        nDisp = 20
        ax_3D = plt.subplot(221, projection="3d")

        n_arr = np.random.choice(np.where(active.sum(1) > 5)[0], nDisp)
        cmap = cm.get_cmap("tab20")
        ax_3D.set_prop_cycle(color=cmap.colors)
        for n in n_arr:
            # centroids = np.zeros((nSes, 2))
            # for s in range(nSes):
            #     fp_id = self.assignments[n, s]
            #     if fp_id>=0:
            #         centroids[s, :] = self.sessions[s].centroids[fp_id, :]
            ax_3D.scatter(
                centroids[n,:, 0],
                centroids[n,:, 1],
                np.arange(nSes),
                s=0.5,
            )  # linewidth=2)
        plt.setp(
            ax_3D,
            xlim=[0, 512 * self.params["pxtomu"]],
            ylim=[0, 512 * self.params["pxtomu"]],
            xlabel="x [$\\mu$m]",
            ylabel="y [$\\mu$m]",
            zlabel="session",
        )
        ax_3D.invert_zaxis()

        ax_proxy = fig.add_axes((0.1, 0.925, 0.01, 0.01))
        # add_number(fig, ax_proxy, order=1, offset=[-50, 25])
        ax_proxy.spines[["top", "right", "bottom", "left"]].set_visible(False)
        # pl_dat.remove_frame(ax_proxy)
        plt.setp(ax_proxy, xticks=[], yticks=[])

        # ax = plt.subplot(243)
        ax = fig.add_axes((0.65, 0.65, 0.125, 0.275))
        # add_number(fig, ax, order=2, offset=[-50, 25])
        dx = np.diff(centroids[..., 0], axis=1) * self.params["pxtomu"]
        ax.hist(
            dx.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dx[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        plt.setp(
            ax,
            xlabel="$\\Delta$x [$\\mu$m]",
            ylabel="density",
            yticks=[],
        )
        ax.spines[["top", "left", "right"]].set_visible(False)

        # ax = plt.subplot(244)
        ax = fig.add_axes((0.8, 0.65, 0.125, 0.275))
        dy = np.diff(centroids[..., 1], axis=1) * self.params["pxtomu"]
        ax.hist(
            dy.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dy[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        plt.setp(ax, xlabel="$\\Delta$y [$\\mu$m]", yticks=[])
        ax.spines[["top", "left", "right"]].set_visible(False)

        ax = fig.add_axes([0.73, 0.85, 0.075, 0.05])
        ax.hist(
            dx.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dx[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        plt.setp(ax, yticks=[])
        # ax.set_xlabel('$\\Delta$x [$\\mu$m]',fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)

        ax = fig.add_axes((0.88, 0.85, 0.075, 0.05))
        ax.hist(
            dy.flatten(), np.linspace(-10, 10, 101), facecolor="tab:blue", alpha=0.5
        )
        ax.hist(
            dy[idx_unsure[:, 1:]].flatten(),
            np.linspace(-10, 10, 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        # ax.set_xlabel('$\\Delta$y [$\\mu$m]',fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)
        plt.setp(ax, yticks=[])

        ROI_diff = np.full((nC, nSes, 2), np.nan)
        com_ref = np.full((nC, 2), np.nan)
        for n in range(nC):
            s_ref = np.where(active[n, :])[0]
            if len(s_ref) > 0:
                com_ref[n, :] = centroids[n, s_ref[0], :]
                ROI_diff[n, : nSes - s_ref[0], :] = (
                    centroids[n, s_ref[0] :, :] - com_ref[n, :]
                )
                # print('neuron %d, first session: %d, \tposition: (%.2f,%.2f)'%(n,s_ref[0],com_ref[n,0],com_ref[n,1]))

        ax_mv = fig.add_axes((0.1, 0.11, 0.35, 0.3))
        # add_number(fig, ax_mv, order=3, offset=[-75, 50])
        # ROI_diff = (self.results['cm'].transpose(1,0,2)-self.results['cm'][:,0,:]).transpose(1,0,2)#*cluster.para['pxtomu']
        # for n in range(nC):
        # ROI_diff[n,:]
        # ROI_diff = (self.results['cm'].transpose(1,0,2)-com_ref).transpose(1,0,2)#*cluster.para['pxtomu']
        ROI_diff_abs = np.array(
            [np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2) for x in ROI_diff]
        )
        # ROI_diff_abs[~cluster.status[...,1]] = np.nan

        for n in n_arr:
            ax_mv.plot(
                range(nSes), ROI_diff_abs[n, :], linewidth=0.5, color=[0.6, 0.6, 0.6]
            )
        ax_mv.plot(
            range(nSes),
            ROI_diff_abs[n, :] * np.nan,
            linewidth=0.5,
            color=[0.6, 0.6, 0.6],
            label="displacement",
        )

        plot_with_confidence(
            ax_mv,
            range(nSes),
            np.nanmean(ROI_diff_abs, 0),
            np.nanstd(ROI_diff_abs, 0),
            col="tab:red",
            ls="-",
            label="average",
        )
        plt.setp(ax_mv, xlabel="session", ylabel="$\\Delta$d [$\\mu$m]", ylim=[0, 11])
        ax_mv.legend(fontsize=10)
        ax_mv.spines[["top", "right"]].set_visible(False)

        idx_c_unsure = idx_unsure.any(1)

        ax_mv_max = fig.add_axes((0.6, 0.11, 0.35, 0.325))
        # add_number(fig, ax_mv_max, order=4, offset=[-75, 50])
        ROI_max_mv = np.nanmax(ROI_diff_abs, 1)
        ax_mv_max.hist(
            ROI_max_mv,
            np.linspace(0, 20, 41),
            facecolor="tab:blue",
            alpha=0.5,
            label="certain",
        )
        ax_mv_max.hist(
            ROI_max_mv[idx_c_unsure],
            np.linspace(0, 20, 41),
            facecolor="tab:red",
            alpha=0.5,
            label="uncertain",
        )
        ax_mv_max.set_xlabel("max($\\Delta$d) [$\\mu$m]")
        ax_mv_max.set_ylabel("# cluster")
        ax_mv_max.legend(fontsize=10)

        ax_mv_max.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.show(block=False)

        # if sv:
        # pl_dat.save_fig('ROI_positions')

    def build_centroids(self):

        centroids = np.full(self.assignments.shape + (2,), np.nan)

        for s,session in enumerate(self.sessions):
            active_neurons = np.where(self.assignments[:, s] >= 0)[0]
            fp_ids = self.assignments[active_neurons, s]
            centroids[active_neurons, s, :] = session.centroids[fp_ids, :]

        return centroids

    def plot_match_statistics(self, s, s_ref=None):

        print("### Plotting matching score statistics ###")

        # print(
        #     "now add example how to calculate footprint correlation(?), sketch how to fill cost-matrix"
        # )
        if s_ref is None:
            s_ref = s - 1
        margins = 20

        ### ------------------------------------------------------------- ###
        ### -------------- Load and preprocess data --------------------- ###
        ### ------------------------------------------------------------- ###
        # clusters = getattr(self, self.cluster_field)

        active = self.assignments >= 0

        ref_data = self.sessions[s_ref]
        this_data = self.sessions[s]
        

        A_ref = ref_data.A
        Cn_ref = ref_data.Cn

        A = this_data.A
        Cn = this_data.Cn
        dims = this_data.dims

        centroids = self.build_centroids()

        D_ROIs = spatial.distance.squareform(
            spatial.distance.pdist(centroids[:, s_ref, :])
        )
        np.fill_diagonal(D_ROIs, np.nan)

        idx_dense = np.where(
            (np.sum(D_ROIs < margins, 1) <= 8) & active[:, s_ref] & active[:, s]
        )[0]
        c = np.random.choice(idx_dense)

        n = int(self.assignments[c, s_ref])

        fig = plt.figure(figsize=(7, 4), dpi=150)
        props = dict(boxstyle="round", facecolor="w", alpha=0.8)

        ## plot ROIs from a single session
        n_close = self.assignments[D_ROIs[c, :] < margins * 1.5, s_ref]

        x, y = centroids[c, s_ref, :].astype("int")

        ax_ROIs1 = fig.add_axes((0.05, 0.55, 0.25, 0.4))
        # add_number(fig, ax_ROIs1, order=1, offset=[-25, 25])
        Cn_ref = normalize_array(Cn_ref)
        # Cn_tmp = Cn_ref[y - margins : y + margins, x - margins : x + margins]

        ax_ROIs1.imshow(Cn_ref, origin="lower", clim=[0, 1])
        An = normalize_array(A_ref[..., n].toarray()).reshape(dims)
        for nn in n_close:
            cc = np.where(self.assignments[:, s_ref] == nn)[0]
            # print(cc, nn)
            ax_ROIs1.contour(
                normalize_array(A_ref[..., nn].toarray()).reshape(dims),
                [0.2],
                colors="w",
                linestyles="--",
                linewidths=1,
            )
        ax_ROIs1.contour(An, [0.2], colors="w", linewidths=3)

        # sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
        # ax_ROIs1.add_artist(sbar)
        plt.setp(
            ax_ROIs1,
            xlim=[x - margins, x + margins],
            ylim=[y - margins, y + margins],
            xticklabels=[],
            yticklabels=[],
        )
        ax_ROIs1.text(
            x - margins + 3, y + margins - 5, "Session s", bbox=props, fontsize=10
        )

        D_ROIs_cross = spatial.distance.cdist(
            centroids[:, s_ref, :], centroids[:, s, :]
        )
        n_close = self.assignments[D_ROIs_cross[c, :] < margins * 2, s]

        ## plot ROIs of session 2 compared to one of session 1
        ax_ROIs2 = fig.add_axes((0.35, 0.55, 0.25, 0.4))
        # add_number(fig, ax_ROIs2, order=2, offset=[-25, 25])

        Cn = normalize_array(Cn)
        ax_ROIs2.imshow(Cn, origin="lower", clim=[0, 1])
        n_match = self.assignments[c, s]
        for nn in n_close:
            cc = np.where(self.assignments[:, s] == nn)
            if not (nn == n_match):  # & (cluster.stats['SNR'][cc,s+1]>3):
                ax_ROIs2.contour(
                    normalize_array(A[..., nn].toarray()).reshape(dims),
                    [0.2],
                    colors="r",
                    linestyles="--",
                    linewidths=1,
                )
        ax_ROIs2.contour(An, [0.2], colors="w", linewidths=3)
        ax_ROIs2.contour(
            normalize_array(A[..., n_match].toarray()).reshape(dims),
            [0.2],
            colors="g",
            linewidths=3,
        )
        plt.setp(
            ax_ROIs2,
            xlim=[x - margins, x + margins],
            ylim=[y - margins, y + margins],
            xticklabels=[],
            yticklabels=[],
        )

        ax_ROIs2.text(
            x - margins + 3, y + margins - 5, "Session s+1", bbox=props, fontsize=10
        )

        ax_zoom1 = fig.add_axes((0.075, 0.125, 0.225, 0.275))
        # add_number(fig, ax_zoom1, order=3, offset=[-50, 25])
        ax_zoom1.hist(
            D_ROIs.flatten(), np.linspace(0.0, 15.0, 31), facecolor="k", density=True
        )
        plt.setp(
            ax_zoom1,
            xlabel="distance [$\\mu$m]",
            ylabel="counts",
            yticks=[],
        )
        ax_zoom1.spines[["top", "left", "right"]].set_visible(False)

        ax = fig.add_axes((0.1, 0.345, 0.075, 0.125))
        plt.hist(
            D_ROIs.flatten(),
            np.linspace(0, np.sqrt(2 * 512**2), 101),
            facecolor="k",
            density=True,
        )
        ax.set_xlabel("d [$\\mu$m]", fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        D_matches = np.copy(D_ROIs_cross.diagonal())
        np.fill_diagonal(D_ROIs_cross, np.nan)

        ax_zoom2 = fig.add_axes((0.35, 0.125, 0.225, 0.275))
        # add_number(fig, ax_zoom2, order=4, offset=[-50, 25])
        ax_zoom2.hist(
            D_ROIs_cross.flatten(),
            np.linspace(0, 15, 31),
            facecolor="tab:red",
            alpha=0.5,
        )
        ax_zoom2.hist(
            D_ROIs.flatten(),
            np.linspace(0, 15, 31),
            facecolor="k",
            edgecolor="k",
            histtype="step",
        )
        ax_zoom2.hist(
            D_matches, np.linspace(0, 15, 31), facecolor="tab:green", alpha=0.5
        )
        ax_zoom2.set_xlabel("distance [$\\mu$m]")
        ax_zoom2.spines[["top", "left", "right"]].set_visible(False)
        ax_zoom2.set_yticks([])

        ax = fig.add_axes((0.38, 0.345, 0.075, 0.125))
        ax.hist(
            D_ROIs_cross.flatten(),
            np.linspace(0, np.sqrt(2 * 512**2), 101),
            facecolor="tab:red",
            alpha=0.5,
        )
        ax.hist(
            D_matches,
            np.linspace(0, np.sqrt(2 * 512**2), 101),
            facecolor="tab:green",
            alpha=0.5,
        )
        ax.set_xlabel("d [$\\mu$m]", fontsize=10)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set_yticks([])

        plt.show(block=False)
        return

        ax = plt.axes([0.7, 0.775, 0.25, 0.125])  # ax_sc1.twinx()
        add_number(fig, ax, order=5, offset=[-75, 50])
        ax.hist(
            cluster.stats["match_score"][:, :, 0].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            alpha=1,
            label="$p^*$",
        )
        ax.hist(
            cluster.stats["match_score"][:, :, 1].flat,
            np.linspace(0, 1, 51),
            facecolor="tab:orange",
            alpha=1,
            label="max($p\\backslash p^*$)",
        )
        # ax.invert_yaxis()
        ax.set_xlim([0, 1])
        ax.set_yticks([])
        ax.set_xlabel("p")
        ax.legend(
            fontsize=8, bbox_to_anchor=[0.3, 0.2], loc="lower left", handlelength=1
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        ax_sc1 = plt.axes([0.7, 0.45, 0.25, 0.125])
        add_number(fig, ax_sc1, order=6, offset=[-75, 50])
        # ax = plt.axes([0.925,0.85,0.225,0.05])#ax_sc1.twiny()
        # ax.set_xticks([])
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)

        ax_sc1.plot(
            cluster.stats["match_score"][:, :, 1].flat,
            cluster.stats["match_score"][:, :, 0].flat,
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc1.plot([0, 1], [0, 1], "--", color="tab:red", lw=1)
        # ax_sc1.plot([0,0.45],[0.5,0.95],'--',color='tab:blue',lw=2)
        # ax_sc1.plot([0.45,1],[0.95,0.95],'--',color='tab:blue',lw=2)
        ax_sc1.set_ylabel("$p^{\\asterisk}$")
        ax_sc1.set_xlabel("max($p\\backslash p^*$)")
        ax_sc1.set_xlim([0, 1])
        ax_sc1.set_ylim([0.5, 1])
        ax_sc1.spines["top"].set_visible(False)
        ax_sc1.spines["right"].set_visible(False)

        ax_sc2 = plt.axes([0.7, 0.125, 0.25, 0.125])
        add_number(fig, ax_sc2, order=7, offset=[-75, 50])
        # plt.hist(np.nanmean(self.results['p_matched'],1),np.linspace(0,1,51))
        ax = ax_sc2.twinx()
        ax.hist(
            np.nanmin(cluster.stats["match_score"][:, :, 0], 1),
            np.linspace(0, 1, 51),
            facecolor="tab:red",
            alpha=0.3,
        )
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = ax_sc2.twiny()
        ax.hist(
            np.nanmean(cluster.stats["match_score"][:, :, 0], axis=1),
            np.linspace(0, 1, 51),
            facecolor="tab:blue",
            orientation="horizontal",
            alpha=0.3,
        )
        ax.set_xticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax_sc2.plot(
            np.nanmin(cluster.stats["match_score"][:, :, 0], 1),
            np.nanmean(cluster.stats["match_score"][:, :, 0], axis=1),
            ".",
            markeredgewidth=0,
            color="k",
            markersize=1,
        )
        ax_sc2.set_xlabel("min($p^{\\asterisk}$)")
        ax_sc2.set_ylabel("$\\left\\langle p^{\\asterisk} \\right\\rangle$")
        ax_sc2.set_xlim([0.5, 1])
        ax_sc2.set_ylim([0.5, 1])
        ax_sc2.spines["top"].set_visible(False)
        ax_sc2.spines["right"].set_visible(False)

        # ax = plt.subplot(248)
        # ax.plot([0,1],[0,1],'--',color='r')
        # ax.scatter(cluster.stats['match_score'][:,:,0],cluster.stats['match_score'][:,:,1],s=1,color='k')
        # ax.set_xlim([0.3,1])
        # ax.set_ylim([-0.05,1])
        # ax.yaxis.tick_right()
        # ax.yaxis.set_label_position("right")
        # ax.set_xlabel('matched score',fontsize=14)
        # ax.set_ylabel('2nd best score',fontsize=14)
        # pl_dat.remove_frame(ax,['top'])
        #
        # ax = plt.subplot(244)
        # #ax.hist(cluster.sessions['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='r',alpha=0.5)
        # ax.hist(cluster.stats['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,label='match score')
        # pl_dat.remove_frame(ax,['left','right','top'])
        # ax.yaxis.set_label_position("right")
        # #ax.yaxis.tick_right()
        # ax.set_xlim([0.3,1])
        # ax.set_xticks([])
        # ax.set_ylabel('density',fontsize=14)
        # ax.legend(loc='upper left',fontsize=10)
        #
        # ax = plt.subplot(247)
        # ax.hist(cluster.stats['match_score'][...,1].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5,density=True,orientation='horizontal',label='2nd best score')
        # #ax.hist(cluster.sessions['match_score'][...,0].flatten(),np.linspace(0,1,101),facecolor='k',alpha=0.5)
        # pl_dat.remove_frame(ax,['left','bottom','top'])
        # ax.set_ylim([-0.05,1])
        # ax.set_xlim([1.2,0])
        # ax.set_yticks([])
        # ax.legend(loc='upper right',fontsize=10)
        # ax.set_xlabel('density',fontsize=14)

        plt.tight_layout()
        plt.show(block=False)

        if sv:
            pl_dat.save_fig("match_stats")

    def plot_alignment_statistics(self, s_ref=0, s_this=1):

        print("### Plotting session alignment procedure and statistics ###")

        """
            Careful: shift is not displayed properly:
            currently, shift between corrected backgrounds is calculated (thus max @(0,0))
            and shift arrow shows shift wrt all (already corrected) reference sessions, not the actual raw one
        """


        nC, nS = self.assignments.shape
        self.classify_sessions()
        active = self.assignments >= 0

        # s = s-1

        # com_mean = np.nanmean(clusters["cm"], 1)

        # self.alignment_template = session_alignment["template"]

        # path = session_alignment["file_paths"][s_ref]
        # A, Cn, quality, trace = self.get_data(path=path)
        # ref_data = self.preprocess_data(
        #     A, Cn, quality, trace, align_to_reference=True
        # )
        str_idx = False
        ref_data = self.sessions[s_ref]
        this_data = self.sessions[s_this]

        # path = session_alignment["file_paths"][s_this]
        # A, Cn, quality, trace = self.get_data(path=path)
        # this_data = self.preprocess_data(A, Cn, quality, trace, align_to_reference=True)
        # dims = ref_data.Cn.shape
        dims = this_data.dims

        W = stats.norm.pdf(range(dims[0]), dims[0] / 2, dims[0] / (0.5 * 1.96))
        W /= W.sum()
        W = np.sqrt(np.diag(W))
        # x_w = np.dot(W,x)

        y = np.hstack([np.ones((512, 1)), np.arange(512).reshape(512, 1)])
        y_w = np.dot(W, y)
        x = np.hstack([np.ones((512, 1)), np.arange(512).reshape(512, 1)])
        x_w = np.dot(W, x)
        # pathSession1 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%1])
        # ROIs1_ld = loadmat(pathSession1)
        # print(self.paths["neuron_detection"][s_])
        # ROIs1_ld = load_data(self.paths["neuron_detection"][s_])
        # print(ROIs1_ld.keys())

        # Cn = np.array(ROIs1_ld["A"].sum(1).reshape(dims))
        # # Cn = ROIs1_ld['Cn'].T
        # Cn -= Cn.min()
        # Cn /= Cn.max()
        # # if self.data[_s]['remap']['transposed']:
        # # Cn2 = Cn2.T
        # # dims = Cn.shape

        # p_vals = np.zeros((cluster.meta['nSes'],4))*np.nan
        p_vals = np.zeros((nS, 2)) * np.nan
        # fig1 = plt.figure(figsize=(7,5),dpi=pl_dat.sv_opt['dpi'])
        fig = plt.figure(figsize=(10, 3), dpi=150)
        for s in tqdm.tqdm(
            np.where(self.alignment_status)[0][1:]
        ):  # cluster.meta['nSes'])):

            com_silent = self.union.centroids[~active[:, s], :]
            com_active = self.union.centroids[active[:, s], :]

            if self.sessions[s].remap.flow:
                # pathSession2 = pathcat([cluster.meta['pathMouse'],'Session%02d/results_redetect.mat'%(s+1)])
                # ROIs2_ld = load_dict_from_hdf5(self.paths['sessions'][s])

                # Cn2 = np.array(ROIs2_ld['A'].sum(1).reshape(dims))
                # Cn2 = ROIs2_ld['Cn']
                # Cn2 -= Cn2.min()
                # Cn2 /= Cn2.max()
                # if self.data[s]['remap']['transpose']:
                #     Cn2 = Cn2.T
                # print('adjust session position')

                # t_start = time.time()
                (x_shift, y_shift), flow, corr, corr_zscored = get_shift_and_flow(
                    ref_data.Cn, this_data.Cn, dims, projection=None
                )
                # (x_shift,y_shift) = cluster.sessions['shift'][s,:]
                # flow = cluster.sessions['flow_field'][s,...]

                # x_remap = (x_grid - x_shift + flow[...,0])
                # y_remap = (y_grid - y_shift + flow[...,1])

                # flow = self.data[s]['remap']['flow']
                # try:
                x_remap, y_remap = build_remap_from_shift_and_flow(
                    dims,
                    self.sessions[s].remap.shift,
                    self.sessions[s].remap.flow,
                )

                flow_w_y = np.dot(self.sessions[s].remap.flow[0, ...], W)
                y0, res, rank, tmp = np.linalg.lstsq(y_w, flow_w_y)
                dy = -y0[0, :] / y0[1, :]
                idx_out = (dy > 512) | (dy < 0)
                r_y = stats.linregress(np.where(~idx_out), dy[~idx_out])
                tilt_ax_y = r_y.intercept + r_y.slope * range(512)

                # print((res**2).sum())
                res_y = np.sqrt(((tilt_ax_y - dy) ** 2).sum()) / dims[0]
                # print('y: %.3f'%(np.sqrt(((tilt_ax_y-dy)**2).sum())/dims[0]))

                flow_w_x = np.dot(self.sessions[s].remap.flow[1, ...], W)
                x0, res, rank, tmp = np.linalg.lstsq(x_w, flow_w_x)
                dx = -x0[0, :] / x0[1, :]
                idx_out = (dx > 512) | (dx < 0)
                r_x = stats.linregress(np.where(~idx_out), dx[~idx_out])
                tilt_ax_x = r_x.intercept + r_x.slope * range(512)
                # print(r_x)
                # print('x:')
                # print((res**2).sum())
                # print('x: %.3f'%(np.sqrt(((tilt_ax_x-dx)**2).sum())/dims[0]))
                res_x = np.sqrt(((tilt_ax_x - dx) ** 2).sum()) / dims[0]
                r = r_y if (res_y < res_x) else r_x
                d = dy if (res_y < res_x) else dx
                tilt_ax = r.intercept + r.slope * range(512)

                # com_PCs = com_mean[cluster.status[cluster.stats['cluster_bool'],s,2],:]

                dist_mean = np.abs(
                    (r.slope * com_mean[:, 0] - com_mean[:, 1] + r.intercept)
                    / np.sqrt(r.slope**2 + 1**2)
                )
                dist_silent = np.abs(
                    (r.slope * com_silent[:, 0] - com_silent[:, 1] + r.intercept)
                    / np.sqrt(r.slope**2 + 1**2)
                )
                dist_active = np.abs(
                    (r.slope * com_active[:, 0] - com_active[:, 1] + r.intercept)
                    / np.sqrt(r.slope**2 + 1**2)
                )

                r_silent = stats.ks_2samp(dist_silent, dist_mean)
                r_active = stats.ks_2samp(dist_active, dist_mean)

                p_vals[s, :] = [r_silent.statistic, r_active.statistic]
            # except:
            #     pass
            # print('time (KS): %.3f'%(time.time()-t_start))
            if s == s_this:

                # ROIs2_ld = load_data(self.paths["neuron_detection"][s])

                Cn_ref = ref_data.Cn
                Cn_ref -= Cn_ref.min()
                Cn_ref /= Cn_ref.max()

                Cn_this = this_data.Cn
                Cn_this -= Cn_this.min()
                Cn_this /= Cn_this.max()
                # # if self.data[s]['remap']['transposed']:
                # #     Cn2 = Cn2.T

                props = dict(boxstyle="round", facecolor="w", alpha=0.8)

                ax_im1 = fig.add_subplot([0.1, 0.325, 0.225, 0.6])
                # add_number(fig, ax_im1, order=1, offset=[-50, -5])
                im_col = np.zeros((512, 512, 3))
                im_col[:, :, 0] = Cn_ref
                ax_im1.imshow(im_col, origin="lower", aspect="auto")
                ax_im1.text(50, 430, "Session %d" % s_this, bbox=props, fontsize=8)
                ax_im1.set_xticks([])
                ax_im1.set_yticks([])

                im_col = np.zeros((512, 512, 3))
                im_col[:, :, 1] = Cn_this

                ax_im2 = fig.add_subplot([0.05, 0.175, 0.225, 0.6])
                ax_im2.imshow(im_col, origin="lower", aspect="auto")
                ax_im2.text(50, 430, "Session %d" % s_ref, bbox=props, fontsize=8)
                ax_im2.set_xticks([])
                ax_im2.set_yticks([])
                # ax_im2.set_xlabel('x [px]',fontsize=14)
                # ax_im2.set_ylabel('y [px]',fontsize=14)
                # sbar = ScaleBar(530.68/512 *10**(-6),location='lower right')
                # ax_im2.add_artist(sbar)

                ax_sShift = fig.add_subplot([0.4, 0.175, 0.225, 0.6])
                # add_number(fig, ax_sShift, order=2)
                cbaxes = fig.add_subplot([0.4, 0.8, 0.075, 0.04])

                C = signal.convolve(
                    Cn_ref - Cn_ref.mean(),
                    Cn_this[::-1, ::-1] - Cn_this.mean(),
                    mode="same",
                ) / (np.prod(dims) * Cn_ref.std() * Cn_this.std())
                C -= np.percentile(C, 95)
                C /= C.max()
                im = ax_sShift.imshow(
                    C,
                    origin="lower",
                    extent=[-dims[0] / 2, dims[0] / 2, -dims[1] / 2, dims[1] / 2],
                    cmap="jet",
                    clim=[0, 1],
                )

                cb = fig.colorbar(im, cax=cbaxes, orientation="horizontal")
                cbaxes.xaxis.set_label_position("top")
                cbaxes.xaxis.tick_top()
                cb.set_ticks([0, 1])
                cb.set_ticklabels(["low", "high"])
                cb.set_label("corr.", fontsize=10)
                ax_sShift.arrow(
                    0,
                    0,
                    float(self.sessions[s].remap.shift[0]),
                    float(self.sessions[s].remap.shift[1]),
                    head_width=1.5,
                    head_length=2,
                    color="k",
                    width=0.1,
                    length_includes_head=True,
                )
                ax_sShift.text(
                    -13,
                    -13,
                    f"shift: ({self.sessions[s].remap.shift[0]:.2f},{self.sessions[s].remap.shift[1]:.2f})",
                    size=10,
                    ha="left",
                    va="bottom",
                    color="k",
                    bbox=props,
                )

                # ax_sShift.colorbar()
                ax_sShift.set_xlim([-15, 15])
                ax_sShift.set_ylim([-15, 15])
                ax_sShift.set_xlabel("$\\Delta x [\\mu m]$")
                ax_sShift.set_ylabel("$\\Delta y [\\mu m]$")

                ax_sShift_all = fig.add_subplot([0.55, 0.6, 0.11, 0.25])
                for ss in range(1, nS):
                    if self.alignment_status[ss]:
                        col = [0.6, 0.6, 0.6]
                    else:
                        col = "tab:red"
                    try:
                        ax_sShift_all.arrow(
                            0,
                            0,
                            *self.sessions[ss].remap.shift,
                            color=col,
                            linewidth=0.5,
                        )
                    except:
                        pass
                ax_sShift_all.arrow(
                    0,
                    0,
                    *self.sessions[s].remap.shift,
                    color="k",
                    linewidth=0.5,
                )
                ax_sShift_all.yaxis.set_label_position("right")
                ax_sShift_all.yaxis.tick_right()
                ax_sShift_all.xaxis.set_label_position("top")
                ax_sShift_all.xaxis.tick_top()
                ax_sShift_all.set_xlim([-15, 15])

                ax_sShift_all.set_ylim([-15, 15])
                # ax_sShift_all.set_xlabel('x [px]',fontsize=10)
                # ax_sShift_all.set_ylabel('y [px]',fontsize=10)

                idxes = 50
                # tx = dims[0]/2 - 1
                # ty = tilt_ax_y[int(tx)]
                # ax_OptFlow = plt.axes([0.8, 0.625, 0.175, 0.25])
                # # add_number(fig, ax_OptFlow, order=3)

                # x_grid, y_grid = np.meshgrid(
                #     np.arange(0.0, dims[0]).astype(np.float32),
                #     np.arange(0.0, dims[1]).astype(np.float32),
                # )

                # ax_OptFlow.set_xlim([0, dims[0]])
                # ax_OptFlow.set_ylim([0, dims[1]])
                # ax_OptFlow.set_xlabel("$x [\\mu m]$")
                # ax_OptFlow.set_ylabel("$y [\\mu m]$")

                ax_sShifted = fig.add_subplot([0.7, 0.175, 0.225, 0.6])
                # add_number(fig, ax_sShifted, order=6, offset=[-5, 25])
                im_col = np.zeros((512, 512, 3))
                im_col[:, :, 0] = Cn_ref
                im_col[:, :, 1] = Cn_this
                ax_sShifted.imshow(im_col, origin="lower")
                ax_sShifted.text(125, 510, "aligned sessions", bbox=props, fontsize=10)
                ax_sShifted.set_xticks([])
                ax_sShifted.set_yticks([])

                # ax_scatter = plt.axes([0.1, 0.125, 0.2, 0.3])
                # # add_number(fig, ax_scatter, order=4)
                # ax_scatter.scatter(com_silent[:, 0], com_silent[:, 1], s=0.7, c="k")
                # ax_scatter.scatter(
                #     com_active[:, 0], com_active[:, 1], s=0.7, c="tab:orange"
                # )
                # # x_ax = np.linspace(0,dims[0]-1,dims[0])
                # # y_ax = n[0]/n[1]*(p[0]-x_ax) + p[1] + n[2]/n[1]*p[2]
                # if self.sessions[s].remap.flow:
                #     ax_scatter.plot(
                #         np.linspace(0, dims[0] - 1, dims[0]),
                #         tilt_ax,
                #         "-",
                #         color="tab:green",
                #     )
                #     ax_hist.hist(
                #         dist_silent,
                #         np.linspace(0, 400, 51),
                #         facecolor="k",
                #         alpha=0.5,
                #         density=True,
                #         label="silent",
                #     )
                #     ax_hist.hist(
                #         dist_active,
                #         np.linspace(0, 400, 51),
                #         facecolor="tab:orange",
                #         alpha=0.5,
                #         density=True,
                #         label="active",
                #     )

                # # ax_scatter.plot(x_ax,y_ax,'k-')
                # ax_scatter.set_xlim([0, dims[0]])
                # ax_scatter.set_ylim([0, dims[0]])
                # ax_scatter.set_xlabel("x [$\\mu$m]")
                # ax_scatter.set_ylabel("y [$\\mu$m]")

                # # x_grid, y_grid = np.meshgrid(np.arange(0., dims[0]).astype(np.float32),
                # # np.arange(0., dims[1]).astype(np.float32))

                # ax_hist = plt.axes([0.4, 0.125, 0.3, 0.3])
                # # add_number(fig, ax_hist, order=5, offset=[-50, 25])
                # # ax_hist.hist(dist_mean,np.linspace(0,400,21),facecolor='k',alpha=0.5,density=True,label='all neurons')

                # ax_hist.legend(loc="lower left", fontsize=8)
                # ax_hist.set_ylabel("density")
                # ax_hist.set_yticks([])
                # ax_hist.set_xlabel("distance from axis [$\\mu$m]")
                # ax_hist.set_xlim([0, 400])
                # ax_hist.spines[["top", "right"]].set_visible(False)
        # except:
        # pass

        # ax_p = plt.axes([0.525, 0.325, 0.125, 0.125])
        # ax_p.axhline(0.01, color="k", linestyle="--")
        # ax_p.plot(
        #     np.where(self.alignment_status)[0],
        #     p_vals[self.alignment_status, 0],
        #     "k",
        #     linewidth=0.5,
        # )
        # ax_p.plot(
        #     np.where(self.alignment_status)[0],
        #     p_vals[self.alignment_status, 1],
        #     "tab:orange",
        #     linewidth=0.5,
        # )
        # # ax_p.plot(np.where(self.alignment_status)[0],p_vals[self.alignment_status],'b')
        # # ax_p.plot(np.where(self.alignment_status)[0],p_vals[self.alignment_status,2],'--',color=[0.6,0.6,0.6])
        # # ax_p.plot(np.where(self.alignment_status)[0],p_vals[self.alignment_status,3],'g--')
        # ax_p.set_yscale("log")
        # ax_p.xaxis.set_label_position("top")
        # ax_p.yaxis.set_label_position("right")
        # ax_p.tick_params(
        #     axis="y",
        #     which="both",
        #     left=False,
        #     right=True,
        #     labelright=True,
        #     labelleft=False,
        # )
        # ax_p.tick_params(
        #     axis="x",
        #     which="both",
        #     top=True,
        #     bottom=False,
        #     labeltop=True,
        #     labelbottom=False,
        # )
        # # ax_p.xaxis.tick_top()
        # # ax_p.yaxis.tick_right()
        # ax_p.set_xlabel("session")
        # ax_p.set_ylim([10 ** (-4), 1])
        # # ax_p.set_ylim([1,0])
        # ax_p.set_ylabel(
        #     "p-value", fontsize=8, rotation="horizontal", labelpad=-5, y=-0.2
        # )
        # ax_p.spines[["bottom", "left"]].set_visible(False)
        # ax_p.tick_params(axis='x',which='both',top=True,bottom=False,labeltop=True,labelbottom=False)

        plt.tight_layout()
        plt.show(block=False)
        # if sv:
        #     pl_dat.save_fig('session_align')

    def plot_footprints(self, c, fp_color="r", ax_in=None, use_plotly=False):
        """
        plots footprints of neuron c across all sessions in 3D view
        """
        # print(
        #     "uhm... appears to be broken: are footprint locations not corrected for shift?"
        # )
        # session_alignment = getattr(self, self.results[0])
        # clusters = getattr(self, self.cluster_field)

        dims = self.sessions[0].dims
        nC, nS = self.clusters["assignments"].shape

        X = np.arange(0, dims[0])
        Y = np.arange(0, dims[1])
        X, Y = np.meshgrid(X, Y)

        if use_plotly:

            if ax_in is None:
                ax = go.Figure()
            else:
                ax = ax_in

            for s in range(nS):
                idx = self.clusters["assignments"][c, s]
                # print("footprint:", s, idx)
                if idx < 0:
                    continue

                A = normalize_array(self.sessions[s].A[:, idx].toarray()).reshape(dims)
                A[A < 0.2] = np.nan

                ax.add_trace(
                    go.Surface(
                        x=X,
                        y=Y,
                        z=A + s,
                        colorscale=[[0, fp_color], [1, fp_color]],
                        showscale=False,
                        name=f"Session {s}",
                    )
                )

            margin = 15
            com = (
                np.nanmean(self.clusters["cm"][c, ...], axis=0) / self.params["pxtomu"]
            )
            ax.update_layout(
                scene=dict(
                    xaxis=dict(range=[com[0] - margin, com[0] + margin]),
                    yaxis=dict(range=[com[1] - margin, com[1] + margin]),
                    zaxis=dict(range=[-1, nS + 1]),
                )
            )
            if ax_in is None:
                ax.show()

        else:
            if ax_in is None:
                fig, ax = plt.subplots(ncols=1, subplot_kw={"projection": "3d"})
            else:
                ax = ax_in

            for s in range(nS):
                idx = self.clusters["assignments"][c, s]
                # print("footprint:", s, idx)
                if idx < 0:
                    continue

                A = normalize_array(self.sessions[s].A[:, idx].toarray()).reshape(dims)
                A[A < 0.2] = np.nan

                ax.plot_surface(
                    X,
                    Y,
                    A + s,
                    linewidth=0,
                    antialiased=False,
                    rstride=5,
                    cstride=5,
                    color=fp_color,
                )

            margin = 15
            com = (
                np.nanmean(self.clusters["cm"][c, ...], axis=0) / self.params["pxtomu"]
            )
            plt.setp(
                ax,
                xlim=[com[0] - margin, com[0] + margin],
                ylim=[com[1] - margin, com[1] + margin],
                zlim=[-1, nS + 1],
            )
            if ax_in is None:
                plt.show(block=False)


def normalize_array(a):
    a_min, a_max = np.percentile(a[a > 0], [1, 99])
    a = (a - a_min) / (a_max - a_min)
    a = np.clip(a, 0, 1)
    return a


def get_contours(a, level, dims=(512, 512)):
    a = normalize_array(a.toarray()).reshape(dims)
    segs = []
    for c in measure.find_contours(a, level):
        segs.append(c[:, ::-1])
    return segs
