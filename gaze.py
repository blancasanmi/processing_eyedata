import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import SCREEN_H, SCREEN_W, FONT_SIZE, CHAR_W, LINE_H

class CatchTrials:
    def __init__(self, path):
        self.path = path
        self.df = pd.read_csv(path)  

    def get_catch_idx(self, idx):
        """Return all rows where catch_index == idx."""
        return self.df[self.df["catch_index"] == idx]  

    def get_by_sentence_position(self, position):
        """Return all rows where sentence_position == position."""
        return self.df[self.df["sentence_position"] == position]


    def percent_correct(self):
        """Return proportion of correct responses (handles bool or string 'true'/'false')."""
        correct = self.df["correct"]
        # Fix: normalise to bool regardless of dtype
        if correct.dtype == object:
            correct = correct.str.strip().str.lower() == "true"
        return (correct.sum() / len(correct)) # Fix: divide by count, not last value

    def mean_rt(self):
        """Return mean reaction time across all trials."""
        return round(self.df["rt"].mean(), 2)


    def rt_stats(self):
        """Return median, IQR, and full quartile breakdown across all trials."""
        rt = self.df["rt"]
        return pd.Series({
            "median" : rt.median(),
            "q1"     : rt.quantile(0.25),
            "q3"     : rt.quantile(0.75),
            "iqr"    : rt.quantile(0.75) - rt.quantile(0.25),
            "min"    : rt.min(),
            "max"    : rt.max(),
        }).round(2)


    def response_distribution(self):
        """Return value counts for response_label."""
        return self.df["response_label"].value_counts()

    def confusion_matrix(self):
        """Cross-tab of correct_response vs response_label."""
        return pd.crosstab(
            self.df["correct_response"],
            self.df["response_label"],
            rownames=["correct_response"],
            colnames=["response_label"]
        )

    def rt_by_correctness(self):
        """Return RT distribution stats split by correct vs incorrect responses."""
        df = self.df.copy()
        if df["correct"].dtype == object:
            df["correct"] = df["correct"].str.strip().str.lower() == "true"
        
        stats = df.groupby("correct")["rt"].describe(
            percentiles=[0.25, 0.5, 0.75]
        )[["count", "min", "25%", "50%", "75%", "max"]].rename(columns={
            "25%": "q1", "50%": "median", "75%": "q3"
        }).round(2)

        stats.index = stats.index.map({True: "correct", False: "incorrect"})
        stats.index.name = "response"
        return stats

    def unique_sentences(self):
        """Return the list of unique sentences used."""
        return self.df["sentence"].unique()

    def cath_trials_nr(self):
        """Return how many trials each sentence appears in."""
        return self.df["catch_index"].value_counts()

    def summary(self):
        """Print a quick overview of the dataset."""
        print(f"Total trials     : {len(self.df)}")
        print(f"Unique sentences : {self.df['sentence'].nunique()}")
        print(f"Percent correct  : {self.percent_correct():.1%}")
        print(f"Mean RT          : {self.mean_rt():.1f} ms")
        print(f"\nRT by correctness:\n{self.rt_by_correctness()}")
        print(f"Catch types      : \n{self.df['catch_type'].value_counts()}")


class GazeData:
    def __init__(self, path):
        self.path = path
        try:
            self.df = pd.read_csv(path, sep=None, engine="python")
        except Exception:
            self.df = pd.read_csv(path, sep="\t")

        self.df.columns = self.df.columns.str.strip()

        if "USER" not in self.df.columns:
            alt = next((c for c in self.df.columns if c.lower() in {"user", "event", "msg", "trigger"}), None)
            if alt is not None:
                self.df = self.df.rename(columns={alt: "USER"})
            else:
                raise KeyError("Expected a USER-like event column in gaze data, but none was found.")

        self.gaze_reg = None

    # ── Column groups ────────────────────────────────────────────────────────

    BEST_POG   = ["BPOGX", "BPOGY", "BPOGV"]
    LEFT_POG   = ["LPOGX", "LPOGY", "LPOGV"]
    RIGHT_POG  = ["RPOGX", "RPOGY", "RPOGV"]
    FIXED_POG  = ["FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV"]
    LEFT_EYE   = ["LEYEX", "LEYEY", "LEYEZ", "LPCX", "LPCY", "LPD", "LPS", "LPV", "LPUPILD", "LPUPILV"]
    RIGHT_EYE  = ["REYEX", "REYEY", "REYEZ", "RPCX", "RPCY", "RPD", "RPS", "RPV", "RPUPILD", "RPUPILV"]
    CURSOR     = ["CX", "CY", "CS"]

    # ── Accessors ────────────────────────────────────────────────────────────

    def fixations(self):
        """Return only valid fixation rows (FPOGV == 1)."""
        return self.df[self.df["FPOGV"] == 1][self.FIXED_POG].reset_index(drop=True)

    def best_pog(self):
        """Return best-eye point of gaze (valid rows only)."""
        return self.df[self.df["BPOGV"] == 1][["TIME", "BPOGX", "BPOGY"]].reset_index(drop=True)

    def left_eye(self):
        """Return all left eye data."""
        return self.df[["TIME"] + self.LEFT_EYE]

    def right_eye(self):
        """Return all right eye data."""
        return self.df[["TIME"] + self.RIGHT_EYE]

    def pupil_data(self):
        """Return pupil diameter and validity for both eyes over time."""
        return self.df[["TIME", "LPUPILD", "LPV", "RPUPILD", "RPV"]]

    def cursor_data(self):
        """Return cursor position and state over time."""
        return self.df[["TIME", "CX", "CY", "CS"]]

    def user_events(self):
        """Return rows where a USER event was logged."""
        return self.df[self.df["USER"].notna()][["TIME", "USER"]].reset_index(drop=True)

    # ── Stats ────────────────────────────────────────────────────────────────

    def fixation_stats(self):
        """Summary stats on fixation duration and position."""
        fix = self.fixations()
        return pd.Series({
            "n_fixations"    : fix["FPOGID"].nunique(),
            "mean_duration"  : round(fix["FPOGD"].mean(), 2),
            "median_duration": round(fix["FPOGD"].median(), 2),
            "mean_x"         : round(fix["FPOGX"].mean(), 2),
            "mean_y"         : round(fix["FPOGY"].mean(), 2),
        })

    def pupil_stats(self):
        """Mean and median pupil diameter per eye (valid samples only)."""
        left  = self.df[self.df["LPV"] == 1]["LPD"]
        right = self.df[self.df["RPV"] == 1]["RPD"]
        return pd.DataFrame({
            "left" : [round(left.mean(), 2),  round(left.median(), 2),  left.count()],
            "right": [round(right.mean(), 2), round(right.median(), 2), right.count()],
        }, index=["mean", "median", "n_valid"])
    
    # ── Triggers ─────────────────────────────────────────────────────────────

    def get_trigger(self, trigger):
        """Return all rows matching a specific trigger string."""
        return self.df[self.df["USER"] == trigger][["CNT", "TIME", "USER"]].reset_index(drop=True)

    def calibration_start(self):
        return self.get_trigger("CALIBRATION_START")

    def experiment_end(self):
        return self.get_trigger("EXPERIMENT_END")

    def fixation_onsets(self, pres=None):
        """Return all fixation onset triggers, or a specific presentation if pres is given."""
        if pres is not None:
            return self.get_trigger(f"FIXATION_ONSET_PRES{pres}")
        return self.df[self.df["USER"].str.startswith("FIXATION_ONSET", na=False)][["CNT", "TIME", "USER"]].reset_index(drop=True)

    def sentence_onsets(self, pres=None):
        """Return all sentence onset triggers, or a specific presentation if pres is given."""
        if pres is not None:
            return self.get_trigger(f"SENTENCE_ONSET_PRES{pres}")
        return self.df[self.df["USER"].str.startswith("SENTENCE_ONSET", na=False)][["CNT", "TIME", "USER"]].reset_index(drop=True)

    def sentence_offsets(self, pres=None):
        """Return all sentence offset triggers, or a specific presentation if pres is given."""
        if pres is not None:
            return self.get_trigger(f"SENTENCE_OFFSET_PRES{pres}")
        return self.df[self.df["USER"].str.startswith("SENTENCE_OFFSET", na=False)][["CNT", "TIME", "USER"]].reset_index(drop=True)

    def gaze_during_pres(self, pres):
        """Return all gaze samples between SENTENCE_ONSET and SENTENCE_OFFSET for a given presentation."""
        onset  = self.sentence_onsets(pres)
        offset = self.sentence_offsets(pres)
        if onset.empty or offset.empty:
            raise ValueError(f"No onset/offset found for presentation {pres}")
        t_start = onset["TIME"].values[0]
        t_end   = offset["TIME"].values[0]
        return self.df[(self.df["TIME"] >= t_start) & (self.df["TIME"] <= t_end)].reset_index(drop=True)

    def gaze_during_fixation(self, pres):
        """Return all gaze samples between FIXATION_ONSET and SENTENCE_ONSET for a given presentation."""
        fixation = self.fixation_onsets(pres)
        onset    = self.sentence_onsets(pres)
        if fixation.empty or onset.empty:
            raise ValueError(f"No fixation/onset found for presentation {pres}")
        t_start = fixation["TIME"].values[0]
        t_end   = onset["TIME"].values[0]
        return self.df[(self.df["TIME"] >= t_start) & (self.df["TIME"] <= t_end)].reset_index(drop=True)

    def all_triggers(self):
        """Return a tidy table of all triggers in order."""
        return self.df[self.df["USER"].notna()][["CNT", "TIME", "USER"]].reset_index(drop=True)

    def n_presentations(self):
        """Infer the number of presentations from the sentence onset triggers."""
        onsets = self.sentence_onsets()
        nums = onsets["USER"].str.extract(r"PRES(\d+)").astype(int)
        return nums[0].max() + 1  # 0-indexed
    
    def get_saccades(self, pres):
        """Return all saccade events (FPOGS == 1)."""
        fixation = self.fixation_onsets(pres)
        onset    = self.sentence_onsets(pres)
        if fixation.empty or onset.empty:
            raise ValueError(f"No fixation/onset found for presentation {pres}")
        t_start = fixation["TIME"].values[0]
        t_end   = onset["TIME"].values[0]
        return self.df[(self.df["FPOGS"] == 1) & (self.df["TIME"] >= t_start) & (self.df["TIME"] <= t_end)][["TIME", "FPOGX", "FPOGY", "FPOGS"]].reset_index(drop=True)
    
    def get_fixation_sequence(self, pres):
        """Return one row per fixation (deduplicated, time-ordered) during the sentence reading window."""
        segment = self.gaze_during_pres(pres)
        fix = (segment[segment["FPOGV"] == 1]
               .drop_duplicates("FPOGID")
               .sort_values("TIME")
               .reset_index(drop=True))
        return fix[["TIME", "FPOGX", "FPOGY", "FPOGID", "FPOGD"]]

    def _build_line_rank(self, df_position, pres):
        """Build a mapping from (part_nr, line_idx) -> global vertical rank.
        Ordered by each line's actual top position on screen (rank 0 = topmost line).
        This makes line comparisons meaningful across parts, since part_nr/line_idx
        numbering resets per part and isn't otherwise comparable."""
        df_pres = df_position[df_position["presentation_index"] == pres]
        line_tops = (df_pres.groupby(["part_nr", "line_idx"])["top"]
                     .mean()
                     .reset_index()
                     .sort_values("top")
                     .reset_index(drop=True))
        line_tops["rank"] = line_tops.index
        return {(row["part_nr"], row["line_idx"]): row["rank"]
                for _, row in line_tops.iterrows()}

    def assign_line_to_fixations(self, df_position, pres, fixations=None, drop_unmatched=True):
        """Tag each fixation with the text line (line_idx, part_nr, and a global
        vertical rank) whose box contains it. By default, fixations that don't
        fall inside any box are dropped (drop_unmatched=True), since they aren't
        meaningful reading fixations and would distort saccade classification."""
        if fixations is None:
            fixations = self.get_fixation_sequence(pres)

        df_pres = df_position[df_position["presentation_index"] == pres].copy()
        df_pres["top_norm"]    = df_pres["top"]    / SCREEN_H
        df_pres["bottom_norm"] = df_pres["bottom"] / SCREEN_H
        df_pres["left_norm"]   = df_pres["left"]   / SCREEN_W
        df_pres["right_norm"]  = df_pres["right"]   / SCREEN_W

        rank_map = self._build_line_rank(df_position, pres)

        line_idx, part_nr, line_rank = [], [], []
        for _, fx in fixations.iterrows():
            match = df_pres[
                (df_pres["top_norm"]  <= fx["FPOGY"]) & (df_pres["bottom_norm"] >= fx["FPOGY"]) &
                (df_pres["left_norm"] <= fx["FPOGX"]) & (df_pres["right_norm"] >= fx["FPOGX"])
            ]
            if not match.empty:
                li = match.iloc[0]["line_idx"]
                pn = match.iloc[0]["part_nr"]
                line_idx.append(li)
                part_nr.append(pn)
                line_rank.append(rank_map.get((pn, li), np.nan))
            else:
                line_idx.append(np.nan)
                part_nr.append(np.nan)
                line_rank.append(np.nan)

        fixations = fixations.copy()
        fixations["line_idx"]  = line_idx
        fixations["part_nr"]   = part_nr
        fixations["line_rank"] = line_rank

        if drop_unmatched:
            fixations = fixations[fixations["line_idx"].notna()].reset_index(drop=True)

        return fixations
    
    def gaze_on_screen(self, df_position):
        """Return gaze samples per presentation and line for all presentations.
        Returns dict: {(pres, part_nr): {line_idx: df_of_gaze_samples}}"""

        self.all_results = {}
        self.gaze_reg = {}

        for pres in df_position["presentation_index"].unique():
            df_pres = df_position[df_position["presentation_index"] == pres].copy()

            try:
                df_gaze = self.gaze_during_pres(pres)
            except ValueError:
                print(f"Warning: no gaze data found for presentation {pres}, skipping.")
                continue

            df_pres["top_norm"]    = df_pres["top"]    / SCREEN_H
            df_pres["bottom_norm"] = df_pres["bottom"] / SCREEN_H
            df_pres["left_norm"]   = df_pres["left"]   / SCREEN_W
            df_pres["right_norm"]  = df_pres["right"]  / SCREEN_W

            for part_nr, df_part in df_pres.groupby("part_nr"):
                line_results = {}
                for _, line in df_part.iterrows():
                    mask = (
                        (df_gaze["BPOGX"] >= line["left_norm"])  &
                        (df_gaze["BPOGX"] <= line["right_norm"]) &
                        (df_gaze["BPOGY"] >= line["top_norm"])   &
                        (df_gaze["BPOGY"] <= line["bottom_norm"]) &
                        (df_gaze["BPOGV"] == 1)
                    )
                    line_results[line["line_idx"]] = df_gaze[mask].reset_index(drop=True)

                self.all_results[(pres, part_nr)] = line_results

            part2_results = self.all_results.get((pres, 2), {})
            part2_timestamps = [line_df["TIME"].min() for line_df in part2_results.values() if not line_df.empty]
            t_part2_start = min(part2_timestamps) if part2_timestamps else None

            part1_results = self.all_results.get((pres, 1), {})
            reg_results = {}
            for line_idx, line_df in part1_results.items():
                if t_part2_start is None:
                    reg_results[line_idx] = line_df
                else:
                    reg_results[line_idx] = line_df[line_df["TIME"] >= t_part2_start].reset_index(drop=True)
            self.gaze_reg[(pres, 1)] = reg_results

        return self.all_results

    def get_reg_gaze(self):
        """Return regression gaze samples (gaze on part 1 after part 2 started).
        Must call gaze_on_screen() first."""
        if self.gaze_reg is None:
            raise ValueError("gaze_reg is None — call gaze_on_screen() first.")
        return self.gaze_reg

    def gaze_fix_stats(self, df_position, n_pres=97) -> pd.DataFrame:
        """Return a DataFrame with regression gaze sample count
        and fixation count per presentation."""
        data = self.get_reg_gaze()

        rows = []
        for i in range(n_pres):
            res = data.get((i, 1), {})
            total_gaze = sum(len(df) for df in res.values())

            try:
                fixations = self.filter_part2_to_part1_saccades(
                    pres=i, df_position=df_position
                )
                total_fixations = len(fixations)
            except ValueError:
                total_fixations = 0

            rows.append({
                "presentation": i,
                "gaze_samples": total_gaze,
                "fixations": total_fixations,
            })

        return pd.DataFrame(rows)
    
    def saccades_movement(self, pres, df_position=None, y_line_threshold=0.03):
        """Compute saccade vectors between consecutive IN-BOX fixations
        (fixations outside any text box are excluded before computing saccades).
        Classifies each saccade as:
        - 'progressive'             : forward reading movement (including skipping down a line)
        - 'line_regression'         : moved UP to a physically earlier (higher) line on screen
        - 'within_line_regression'  : moved leftward within the same physical line

        is_regression is 1 for either regression type, 0 for progressive.
        """
        fixations = self.get_fixation_sequence(pres)

        if df_position is not None:
            # drop_unmatched=True by default -> only fixations inside a box are kept
            fixations = self.assign_line_to_fixations(df_position, pres, fixations)

        columns = [
            "pres", "from_time", "to_time", "from_fpogid", "to_fpogid",
            "from_part", "to_part",
            "dx", "dy", "amplitude", "saccade_type", "is_regression",
        ]

        rows = []
        for i in range(1, len(fixations)):
            prev, curr = fixations.iloc[i - 1], fixations.iloc[i]
            dx = curr["FPOGX"] - prev["FPOGX"]
            dy = curr["FPOGY"] - prev["FPOGY"]
            amplitude = (dx**2 + dy**2) ** 0.5

            if df_position is not None:
                rank_delta = curr["line_rank"] - prev["line_rank"]
                if rank_delta < 0:
                    sac_type = "line_regression"
                elif rank_delta == 0 and dx < 0:
                    sac_type = "within_line_regression"
                else:
                    sac_type = "progressive"
            else:
                if dy < -y_line_threshold:
                    sac_type = "line_regression"
                elif dx < 0 and abs(dy) < y_line_threshold:
                    sac_type = "within_line_regression"
                else:
                    sac_type = "progressive"

            values = [
                pres, prev["TIME"], curr["TIME"], prev["FPOGID"], curr["FPOGID"],
                prev.get("part_nr", np.nan), curr.get("part_nr", np.nan),
                round(dx, 4), round(dy, 4), round(amplitude, 4),
                sac_type, int(sac_type != "progressive"),
            ]
            rows.append(dict(zip(columns, values)))

        return pd.DataFrame(rows, columns=columns)


    def filter_part2_to_part1_saccades(self, pres, df_position=None, saccades=None):
        """Keep only saccades where the gaze moves from a part-2 fixation
        directly to a part-1 fixation (true regressions back to part 1)."""
        if saccades is None:
            saccades = self.saccades_movement(pres, df_position=df_position)

        mask = (saccades["from_part"] == 2) & (saccades["to_part"] == 1)
        return saccades[mask].reset_index(drop=True)

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self):
        """Print a quick overview of the gaze recording."""
        print(f"Total samples    : {len(self.df)}")
        print(f"Duration         : {round(self.df['TIME'].max(), 2)} s")
        print(f"Valid fixations  : {self.df['FPOGV'].sum()}")
        print(f"User events      : {self.df['USER'].notna().sum()}")
        print(f"\nFixation stats:\n{self.fixation_stats()}")
        print(f"\nPupil stats:\n{self.pupil_stats()}")

    def sentence_durations(self):
        """Return the duration (s) spent on each sentence presentation."""
        onsets  = self.sentence_onsets()
        offsets = self.sentence_offsets()

        onsets["pres"]  = onsets["USER"].str.extract(r"PRES(\d+)").astype(int)
        offsets["pres"] = offsets["USER"].str.extract(r"PRES(\d+)").astype(int)

        merged = pd.merge(
            onsets[["pres", "TIME"]].rename(columns={"TIME": "onset"}),
            offsets[["pres", "TIME"]].rename(columns={"TIME": "offset"}),
            on="pres"
        )
        merged["duration"] = (merged["offset"] - merged["onset"]).round(2)
        return merged.sort_values("pres").reset_index(drop=True)
    
    def sentence_duration_stats(self):
        """Summary stats on how long participants read each sentence."""
        durations = self.sentence_durations()["duration"]
        return pd.Series({
            "mean"  : durations.mean().round(2),
            "median": durations.median().round(2),
            "min"   : durations.min().round(2),
            "max"   : durations.max().round(2),
            "std"   : durations.std().round(2),
        })
    
    def pupil_per_presentation(self):
        """Return pupil diameter stats per presentation (averaged across both valid eyes)."""
        rows = []
        for pres in range(self.n_presentations()):
            segment = self.gaze_during_pres(pres)
            left  = segment[segment["LPV"] == 1]["LPD"]
            right = segment[segment["RPV"] == 1]["RPD"]
            both  = pd.concat([left, right])
            rows.append({
                "pres"        : pres,
                "mean_pupil"  : round(both.mean(), 2),
                "median_pupil": round(both.median(), 2),
                "std_pupil"   : round(both.std(), 2),
                "q1"          : round(both.quantile(0.25), 2),
                "q3"          : round(both.quantile(0.75), 2),
                "iqr"         : round(both.quantile(0.75) - both.quantile(0.25), 2),
                "n_valid"     : len(both),
            })
        return pd.DataFrame(rows)
    


    # ── Visualisation ────────────────────────────────────────────────────────

    def plot_gaze_path(self):
        """Scatter plot of best POG gaze path over time."""
        pog = self.best_pog()
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(pog["BPOGX"], pog["BPOGY"],
                              c=pog["TIME"], cmap="viridis", s=5, alpha=0.6)
        plt.colorbar(scatter, label="Time (s)")
        plt.gca().invert_yaxis()          # screen coords: 0,0 is top-left
        plt.title("Gaze path (best POG)")
        plt.xlabel("X (normalised)")
        plt.ylabel("Y (normalised)")
        plt.tight_layout()
        plt.show()

    def plot_fixations(self):
        """Bubble plot of fixations — bubble size encodes duration."""
        fix = self.df[self.df["FPOGV"] == 1].drop_duplicates("FPOGID")
        plt.figure(figsize=(8, 6))
        plt.scatter(fix["FPOGX"], fix["FPOGY"],
                    s=fix["FPOGD"] * 500, alpha=0.5, edgecolors="steelblue", facecolors="none")
        plt.gca().invert_yaxis()
        plt.title("Fixations (size = duration)")
        plt.xlabel("X (normalised)")
        plt.ylabel("Y (normalised)")
        plt.tight_layout()
        plt.show()

    def plot_pupil_diameter(self):
        """Line plot of pupil diameter over time for both eyes."""
        df = self.df.copy()
        left  = df[df["LPV"] == 1][["TIME", "LPD"]]
        right = df[df["RPV"] == 1][["TIME", "RPD"]]
        plt.figure(figsize=(10, 4))
        plt.plot(left["TIME"],  left["LPD"],  label="Left",  alpha=0.7)
        plt.plot(right["TIME"], right["RPD"], label="Right", alpha=0.7)
        plt.title("Pupil diameter over time")
        plt.xlabel("Time (s)")
        plt.ylabel("Diameter (px)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_pupil_per_presentation(self):
        """Plot mean pupil diameter ± std across presentations."""
        df = self.pupil_per_presentation()
        plt.figure(figsize=(10, 4))
        plt.errorbar(df["pres"], df["mean_pupil"], yerr=df["std_pupil"],
                     fmt="o-", capsize=4, alpha=0.8, label="mean ± std")
        plt.fill_between(df["pres"], df["q1"], df["q3"], alpha=0.2, label="IQR")
        plt.title("Pupil dilation variability across presentations")
        plt.xlabel("Presentation")
        plt.ylabel("Pupil diameter (mm)")
        plt.xticks(df["pres"])
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _fontsize_for_box(self, fig, ax, text, box_width_px, box_height_px, font_px=40):
        """Return a font size in points that stays within a box."""
        if not text:
            return self._px_to_fontsize(fig, ax, font_px)
        width_based_px = self._derive_font_px_from_box(text, box_width_px)
        height_based_px = max(box_height_px * 0.75, 1)
        fit_px = min(width_based_px, height_based_px, font_px)
        return self._px_to_fontsize(fig, ax, fit_px)

    def plot_saccades_on_text(self, df_position, pres, saccades=None, show_gaze=False, results=None, font_px=40):
        SACCADE_COLORS = {
        "progressive"            : "green",
        "line_regression"        : "red",
        "within_line_regression" : "orange",
        }
        df_pres = df_position[df_position["presentation_index"] == pres].copy()
        if saccades is None:
            saccades = self.saccades_movement(pres, df_position=df_position)

        # Keep only saccades that jump from a part-2 fixation to a part-1 fixation
        saccades = self.filter_part2_to_part1_saccades(pres, df_position=df_position, saccades=saccades)

        fig_width = 12
        fig_height = fig_width * (SCREEN_H / SCREEN_W)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, SCREEN_W)
        ax.set_ylim(0, SCREEN_H)
        ax.invert_yaxis()
        ax.set_title(f"Saccades — presentation {pres} (part 2 → part 1 regressions)")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")
        plt.tight_layout()

        for _, line in df_pres.iterrows():
            rect = plt.Rectangle(
                (line["left"], line["top"]),
                line["right"] - line["left"],
                line["bottom"] - line["top"],
                linewidth=1, edgecolor="steelblue", facecolor="aliceblue", alpha=0.3
            )
            ax.add_patch(rect)
            box_width_px = max(1, line["right"] - line["left"])
            box_height_px = max(1, line["bottom"] - line["top"])
            fontsize_pt = self._fontsize_for_box(
                fig,
                ax,
                line["text"],
                box_width_px,
                box_height_px,
                font_px=font_px,
            )
            ax.text(
                (line["left"] + line["right"]) / 2,
                (line["top"] + line["bottom"]) / 2,
                line["text"], fontsize=fontsize_pt, family="monospace", color="#111111",
                ha="center", va="center", zorder=2
            )

        if show_gaze:
            if results is None:
                results = self.gaze_on_screen(df_position)
            for (p, part_nr), line_dict in results.items():
                if p != pres:
                    continue
                for gaze in line_dict.values():
                    if not gaze.empty:
                        ax.scatter(gaze["BPOGX"] * SCREEN_W, gaze["BPOGY"] * SCREEN_H,
                                s=8, alpha=0.15, color="gray", zorder=2)

        fixations = self.assign_line_to_fixations(df_position, pres)
        for _, sac in saccades.iterrows():
            prev = fixations[fixations["FPOGID"] == sac["from_fpogid"]].iloc[0]
            curr = fixations[fixations["FPOGID"] == sac["to_fpogid"]].iloc[0]
            x0, y0 = prev["FPOGX"] * SCREEN_W, prev["FPOGY"] * SCREEN_H
            x1, y1 = curr["FPOGX"] * SCREEN_W, curr["FPOGY"] * SCREEN_H
            color = SACCADE_COLORS[sac["saccade_type"]]
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color, alpha=0.8, lw=1.3), zorder=4)

        from matplotlib.lines import Line2D
        legend_elems = [Line2D([0], [0], color=c, lw=2, label=t.replace("_", " "))
                        for t, c in SACCADE_COLORS.items()]
        ax.legend(handles=legend_elems, loc="upper right", fontsize=8)

        plt.show()

    @staticmethod
    def _px_to_fontsize(fig, ax, px_size):
        """Convert a CSS/screen pixel size into a matplotlib fontsize in points,
        based on the AXES' actual rendered size (not the full figure), so it
        accounts for margins taken up by titles, labels, and layout adjustments."""
        fig.canvas.draw()  # finalize layout so axes position/size is accurate
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)  # axes size in display pixels
        ax_width_in  = bbox.width  / fig.dpi
        ax_height_in = bbox.height / fig.dpi

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        data_width_px  = abs(xlim[1] - xlim[0])
        data_height_px = abs(ylim[1] - ylim[0])

        scale_x = ax_width_in  / data_width_px
        scale_y = ax_height_in / data_height_px
        scale = min(scale_x, scale_y)
        return px_size * scale * 72

    def _derive_font_px_from_box(self, text, box_width_px, char_width_ratio=0.6, letter_spacing_px=2):
        """Estimate the true rendered font-size (px) from a line's box width and
        text content, given a monospace font with known letter-spacing.

        For 'Courier New'/monospace, each character's advance width is
        approximately 0.6x the font-size (design metric for Courier-style fonts).
        Total width ≈ n_chars * (char_width_ratio * font_size) + (n_chars - 1) * letter_spacing
        Solve for font_size.
        """
        n_chars = len(text)
        if n_chars <= 1:
            return 40  # fallback, can't solve meaningfully for a 0/1-char line
        spacing_total = (n_chars - 1) * letter_spacing_px
        font_size = (box_width_px - spacing_total) / (n_chars * char_width_ratio)
        return max(font_size, 1)  # guard against negative/zero from bad data

    def plot_gaze_on_text(self, df_position, pres, results=None, time_col="TIME", font_px=40):
        df_pres = df_position[df_position["presentation_index"] == pres].copy()

        if results is None:
            results = self.gaze_on_screen(df_position)

        # ── All gaze samples (both parts) ──────────────────────────────
        all_gaze = []
        for part_nr, df_part in df_pres.groupby("part_nr"):
            line_results = results.get((pres, part_nr), {})
            for _, line in df_part.iterrows():
                gaze = line_results.get(line["line_idx"], pd.DataFrame())
                if not gaze.empty:
                    all_gaze.append(gaze)

        if not all_gaze:
            print(f"No gaze data for presentation {pres}")
            return

        all_gaze = pd.concat(all_gaze).drop_duplicates().reset_index(drop=True)

        # ── Determine when part 2 starts (earliest gaze landing on any part-2 box) ──
        part2_line_results = results.get((pres, 2), {})
        part2_timestamps = [df_line[time_col].min() for df_line in part2_line_results.values() if not df_line.empty]

        if not part2_timestamps:
            print(f"[pres {pres}] WARNING: no gaze landed on part 2 boxes — can't determine part 2 start, nothing highlighted.")
            t_part2_start = None
        else:
            t_part2_start = min(part2_timestamps)

        # ── Split gaze into before / after part 2 start ─────────────────
        if t_part2_start is not None:
            before = all_gaze[all_gaze[time_col] < t_part2_start]
            after  = all_gaze[all_gaze[time_col] >= t_part2_start]
        else:
            before = all_gaze
            after = pd.DataFrame(columns=all_gaze.columns)

        print(f"[pres {pres}] total gaze: {len(all_gaze)} | before part 2: {len(before)} | from part 2 start onward: {len(after)}")

        # ── Plot ─────────────────────────────────────────────────────
        fig_width = 12
        fig_height = fig_width * (SCREEN_H / SCREEN_W)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, SCREEN_W)
        ax.set_ylim(0, SCREEN_H)
        ax.invert_yaxis()
        ax.set_title(f"Gaze on text — presentation {pres} (both parts)")
        ax.set_xlabel("X (px)")
        ax.set_ylabel("Y (px)")

        # Draw text boxes
        for part_nr, df_part in df_pres.groupby("part_nr"):
            for _, line in df_part.iterrows():
                rect = plt.Rectangle(
                    (line["left"], line["top"]),
                    line["right"] - line["left"],
                    line["bottom"] - line["top"],
                    linewidth=1, edgecolor="steelblue", facecolor="aliceblue", alpha=0.4,
                    zorder=1
                )
                ax.add_patch(rect)

                box_width_px = max(1, line["right"] - line["left"])
                box_height_px = max(1, line["bottom"] - line["top"])
                fontsize_pt = self._fontsize_for_box(
                    fig, ax, line["text"], box_width_px, box_height_px, font_px=font_px,
                )

                x_center = (line["left"] + line["right"]) / 2
                y_center = (line["top"] + line["bottom"]) / 2
                ax.text(
                    x_center, y_center, line["text"],
                    fontsize=fontsize_pt, family="monospace", color="#111111",
                    ha="center", va="center", zorder=2
                )

        # Gaze before part 2 starts — grey
        ax.scatter(
            before["BPOGX"] * SCREEN_W,
            before["BPOGY"] * SCREEN_H,
            s=15, alpha=0.4, color="grey", zorder=3, label="before part 2"
        )

        # Gaze from part 2 start onward — soft yellow
        if not after.empty:
            ax.scatter(
                after["BPOGX"] * SCREEN_W,
                after["BPOGY"] * SCREEN_H,
                s=15, alpha=0.35, color="red", zorder=4, label="from part 2 start"
            )

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()