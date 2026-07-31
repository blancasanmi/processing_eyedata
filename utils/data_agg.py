from typing import List
from utils.gazedata import GazeData
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_pres_real_maps(sentence_order_path: str):
    """Load presentation_index <-> real_index maps directly from sentence_order.csv."""
    df = pd.read_csv(sentence_order_path)
    pres_to_real = dict(zip(df["presentation_index"], df["real_index"]))
    real_to_pres = dict(zip(df["real_index"], df["presentation_index"]))
    return pres_to_real, real_to_pres


class DataAggregator:
    META_COLS = ["real_index", "Uncertainty", "Resolution", "Unc Valence", "Res Valence"]

    METRICS = [
        "n_fixations_sentence1",
        "n_fixations_sentence2",
        "horizontal_regressions_sentence1",
        "vertical_regressions_sentence1",
        "horizontal_regressions_sentence2",
        "vertical_regressions_sentence2",
        "horizontal_regressions_sentence1_after_sentence2",
        "vertical_regressions_sentence1_after_sentence2",
        "cross_sentence_regressions",
        "gaze_samples_sentence1_after_sentence2",
    ]

    def __init__(
    self,
    gazedata: List[GazeData],
    sentence_order_paths: List[str],
    df_pos: pd.DataFrame,
    sentence_metadata_path: str = None,
    participant_ids: List[str] = None,
    ):
        if len(gazedata) != len(sentence_order_paths):
            raise ValueError(
                f"gazedata ({len(gazedata)}) and sentence_order_paths "
                f"({len(sentence_order_paths)}) must be the same length and aligned by participant."
            )

        if participant_ids is not None and len(participant_ids) != len(gazedata):
            raise ValueError(
                f"participant_ids ({len(participant_ids)}) must match gazedata length ({len(gazedata)})."
            )

        self.gazedata = gazedata
        self.df_pos = df_pos
        self.participant_ids = participant_ids  # e.g. ["01", "03", "04", ...], aligned with gazedata by index

        self.sentence_metadata = None
        if sentence_metadata_path is not None:
            self.sentence_metadata = self.load_sentence_metadata(sentence_metadata_path)

        self.maps = [load_pres_real_maps(path) for path in sentence_order_paths]

        self.df_pos_per_participant = self._build_df_pos_per_participant()

    def get_df_pos(self, participant_idx: int) -> pd.DataFrame:
        return self.df_pos_per_participant[participant_idx]

    def _build_df_pos_per_participant(self) -> List[pd.DataFrame]:
        translated = []
        for i, (pres_to_real, real_to_pres) in enumerate(self.maps):
            df = self.df_pos.copy()
            df["presentation_index"] = df["real_index"].map(real_to_pres)

            unmapped = df[df["presentation_index"].isna()]
            if not unmapped.empty:
                print(f"  WARNING: participant {i} has {len(unmapped)} unmapped rows in df_pos")

            df["presentation_index"] = df["presentation_index"].astype(int)
            translated.append(df)
        return translated

    def load_sentence_metadata(self, path: str) -> pd.DataFrame:
        """real_index is derived from row order — sentences_newconditions.xlsx
        has no explicit index column."""
        df = pd.read_excel(path)
        df.index.name = "real_index"
        df = df.reset_index()
        print(f"[load_sentence_metadata] Loaded {len(df)} sentences")
        return df

    def compute_gaze_fix_stats(self) -> pd.DataFrame:
        rows_all = []

        for i, gaze in enumerate(self.gazedata):
            df_pos_translated = self.get_df_pos(i)
            pres_to_real, _ = self.maps[i]

            gaze.gaze_on_screen(df_pos_translated)
            stats = gaze.gaze_fix_stats(df_pos_translated)

            stats["real_index"] = stats["presentation"].map(pres_to_real)
            stats["participant"] = i
            stats["participant_id"] = self.participant_ids[i] if self.participant_ids is not None else i

            rows_all.append(stats)

        df_all = pd.concat(rows_all).reset_index(drop=True)

        if self.sentence_metadata is not None:
            df_all = df_all.merge(self.sentence_metadata[self.META_COLS], on="real_index", how="left")
            missing = df_all[df_all["Uncertainty"].isna()]
            if not missing.empty:
                print(f"  WARNING: {len(missing)} rows in df_all have no metadata")

        print(f"[DataAggregator] Combined df shape: {df_all.shape}")
        return df_all

    def aggregate_by_sentence(self) -> pd.DataFrame:
        df_all = self.compute_gaze_fix_stats()

        agg_dict = {}
        for m in self.METRICS:
            agg_dict[f"mean_{m}"] = (m, "mean")
            agg_dict[f"std_{m}"] = (m, "std")
        agg_dict["n_participants"] = ("participant", "count")

        result = df_all.groupby("real_index").agg(**agg_dict).reset_index()

        if self.sentence_metadata is not None:
            result = result.merge(self.sentence_metadata[self.META_COLS], on="real_index", how="left")
            missing = result[result["Uncertainty"].isna()]
            if not missing.empty:
                print(f"  WARNING: {len(missing)} sentences have no metadata")

        return result

    def plot_boxplot_grid_by_condition_per_participant(
        self,
        df_all: pd.DataFrame,
        metric: str,
        ylabel: str = None,
    ):
        """Boxplot grid: one subplot per Uncertainty x Resolution combination,
        each with 4 boxplots covering all Unc Valence x Res Valence combinations.

        Each box shows the distribution ACROSS PARTICIPANTS of that participant's
        average metric value within the condition cell — i.e. for each participant,
        average `metric` over all sentences they saw in that condition, then plot
        the spread of those per-participant averages. Sentence-level variability
        is collapsed away; participant-level variability is what's shown.
        """
        if metric not in self.METRICS:
            raise ValueError(f"Unknown metric '{metric}'. Must be one of: {self.METRICS}")

        ylabel = ylabel or metric

        group_cols = ["participant", "Uncertainty", "Resolution", "Unc Valence", "Res Valence"]
        df = df_all.dropna(subset=[metric] + group_cols)

        # Step 1: collapse sentences -> one value per (participant, condition cell)
        per_participant = (
            df.groupby(group_cols)[metric]
            .mean()
            .reset_index()
        )

        combos = (
            per_participant[["Uncertainty", "Resolution"]]
            .drop_duplicates()
            .sort_values(["Uncertainty", "Resolution"])
            .itertuples(index=False, name=None)
        )
        combos = list(combos)

        n = len(combos)
        ncols = min(2, n) if n > 1 else 1
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False, sharey=True)
        axes_flat = axes.flatten()

        bar_specs = [
            ("pos", "pos", "Unc+ / Res+", "mediumseagreen"),
            ("pos", "neg", "Unc+ / Res−", "seagreen"),
            ("neg", "pos", "Unc− / Res+", "salmon"),
            ("neg", "neg", "Unc− / Res−", "firebrick"),
        ]
        bar_labels = [spec[2] for spec in bar_specs]
        bar_colors = [spec[3] for spec in bar_specs]

        def get_vals(sub, unc_val_prefix, res_val_prefix):
            rows = sub[
                sub["Unc Valence"].str.lower().str.startswith(unc_val_prefix)
                & sub["Res Valence"].str.lower().str.startswith(res_val_prefix)
            ]
            return rows[metric].values  # one value per participant

        subplot_data = []
        all_vals = []
        for unc, res in combos:
            sub = per_participant[(per_participant["Uncertainty"] == unc) & (per_participant["Resolution"] == res)]
            box_vals = [get_vals(sub, u, r) for u, r, _, _ in bar_specs]
            subplot_data.append((unc, res, box_vals))
            for v in box_vals:
                all_vals.extend(v)

        y_max = max(all_vals) * 1.1 if len(all_vals) else 1
        y_min = min(0, min(all_vals) * 1.1) if len(all_vals) else 0

        for ax_idx, (unc, res, box_vals) in enumerate(subplot_data):
            ax = axes_flat[ax_idx]
            bp = ax.boxplot(box_vals, labels=bar_labels, showmeans=True, patch_artist=True)
            for patch, color in zip(bp["boxes"], bar_colors):
                patch.set_facecolor(color)
                patch.set_edgecolor("black")
                patch.set_linewidth(0.5)
            ax.set_ylim(y_min, y_max)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Uncertainty: {unc} | Resolution: {res}")
            ax.tick_params(axis="x", rotation=15)

        for j in range(len(combos), len(axes_flat)):
            axes_flat[j].axis("off")

        fig.suptitle(f"{ylabel} by condition ({metric}) — per participant", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig


    def plot_boxplot_grid_by_condition_by_sentence(
            self,
            df_by_sentence: pd.DataFrame,
            metric: str,
            use_mean: bool = True,
            ylabel: str = None,
        ):
            """Boxplot grid: one subplot per Uncertainty x Resolution combination,
            each with 4 boxplots covering all Unc Valence x Res Valence combinations.

            Each box shows the distribution ACROSS SENTENCES of that sentence's
            across-participant average metric value — i.e. participant variability
            is collapsed away (already averaged out in df_by_sentence), and what's
            shown is how much sentences differ from each other within the condition.
            """
            if metric not in self.METRICS:
                raise ValueError(f"Unknown metric '{metric}'. Must be one of: {self.METRICS}")

            col = f"{'mean' if use_mean else 'std'}_{metric}"
            if col not in df_by_sentence.columns:
                raise ValueError(f"Column '{col}' not found — did you run aggregate_by_sentence()?")

            ylabel = ylabel or col

            df = df_by_sentence.dropna(
                subset=[col, "Uncertainty", "Resolution", "Unc Valence", "Res Valence"]
            )

            combos = (
                df[["Uncertainty", "Resolution"]]
                .drop_duplicates()
                .sort_values(["Uncertainty", "Resolution"])
                .itertuples(index=False, name=None)
            )
            combos = list(combos)

            n = len(combos)
            ncols = min(2, n) if n > 1 else 1
            nrows = int(np.ceil(n / ncols))

            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False, sharey=True)
            axes_flat = axes.flatten()

            bar_specs = [
                ("pos", "pos", "Unc+ / Res+", "mediumseagreen"),
                ("pos", "neg", "Unc+ / Res−", "seagreen"),
                ("neg", "pos", "Unc− / Res+", "salmon"),
                ("neg", "neg", "Unc− / Res−", "firebrick"),
            ]
            bar_labels = [spec[2] for spec in bar_specs]
            bar_colors = [spec[3] for spec in bar_specs]

            def get_vals(sub, unc_val_prefix, res_val_prefix):
                rows = sub[
                    sub["Unc Valence"].str.lower().str.startswith(unc_val_prefix)
                    & sub["Res Valence"].str.lower().str.startswith(res_val_prefix)
                ]
                return rows[col].values

            subplot_data = []
            all_vals = []
            for unc, res in combos:
                sub = df[(df["Uncertainty"] == unc) & (df["Resolution"] == res)]
                box_vals = [get_vals(sub, u, r) for u, r, _, _ in bar_specs]
                subplot_data.append((unc, res, box_vals))
                for v in box_vals:
                    all_vals.extend(v)

            y_max = max(all_vals) * 1.1 if len(all_vals) else 1
            y_min = min(0, min(all_vals) * 1.1) if len(all_vals) else 0

            for ax_idx, (unc, res, box_vals) in enumerate(subplot_data):
                ax = axes_flat[ax_idx]
                bp = ax.boxplot(box_vals, labels=bar_labels, showmeans=True, patch_artist=True)
                for patch, color in zip(bp["boxes"], bar_colors):
                    patch.set_facecolor(color)
                    patch.set_edgecolor("black")
                    patch.set_linewidth(0.5)
                ax.set_ylim(y_min, y_max)
                ax.set_ylabel(ylabel)
                ax.set_title(f"Uncertainty: {unc} | Resolution: {res}")
                ax.tick_params(axis="x", rotation=15)

            for j in range(len(combos), len(axes_flat)):
                axes_flat[j].axis("off")

            fig.suptitle(f"{ylabel} by condition ({metric}) — by sentence, across participants", fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            return fig