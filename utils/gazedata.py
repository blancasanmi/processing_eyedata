import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from constants import SCREEN_H, SCREEN_W, FONT_SIZE, CHAR_W, LINE_H, CALIBRATION_TOLERANCE_PX

class GazeData:
    def __init__(self, path):
        self.path = path
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
        df_pres["top_norm"]    = (df_pres["top"]    - CALIBRATION_TOLERANCE_PX) / SCREEN_H
        df_pres["bottom_norm"] = (df_pres["bottom"] + CALIBRATION_TOLERANCE_PX) / SCREEN_H
        df_pres["left_norm"]   = (df_pres["left"]   - CALIBRATION_TOLERANCE_PX) / SCREEN_W
        df_pres["right_norm"]  = (df_pres["right"]  + CALIBRATION_TOLERANCE_PX) / SCREEN_W

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

            df_pres["top_norm"]    = (df_pres["top"]    - CALIBRATION_TOLERANCE_PX) / SCREEN_H
            df_pres["bottom_norm"] = (df_pres["bottom"] + CALIBRATION_TOLERANCE_PX) / SCREEN_H
            df_pres["left_norm"]   = (df_pres["left"]   - CALIBRATION_TOLERANCE_PX) / SCREEN_W
            df_pres["right_norm"]  = (df_pres["right"]  + CALIBRATION_TOLERANCE_PX) / SCREEN_W

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

    def get_part2_start_time(self, pres, df_position, results=None, time_col="TIME"):
        """Earliest gaze timestamp landing on any part-2 box for this presentation.
        Returns None if no gaze landed on part 2 (cutoff undefined)."""
        if results is None:
            results = self.gaze_on_screen(df_position)

        part2_line_results = results.get((pres, 2), {})
        part2_timestamps = [
            df_line[time_col].min()
            for df_line in part2_line_results.values()
            if not df_line.empty
        ]

        if not part2_timestamps:
            return None
        return min(part2_timestamps)

    def get_total_horizontal_fixations(self, pres, df_position=None, y_line_threshold=0.03, part_nr=1, results=None, time_col="TIME"):
        """Count within_line_regression saccades inside the given part_nr,
        restricted to saccades occurring at/after part 2's first gaze landing.

        Kept for backward compatibility — equivalent to calling
        `_saccade_type_counts(..., part_nr=part_nr, min_time=t_part2_start)[0]`.
        """
        saccades = self.saccades_movement(pres, df_position=df_position, y_line_threshold=y_line_threshold)

        t_part2_start = self.get_part2_start_time(pres, df_position, results=results, time_col=time_col)
        if t_part2_start is None:
            return 0

        horizontal, _ = self._saccade_type_counts(saccades, part_nr=part_nr, min_time=t_part2_start)
        return horizontal

    def _saccade_type_counts(self, saccades, part_nr=None, min_time=None):
        """Count horizontal (within-line) and vertical (line-to-line) regression
        saccades in a saccades DataFrame (as returned by saccades_movement).

        part_nr: if given, restrict to saccades that stay within that part
                 (from_part == to_part == part_nr).
        min_time: if given, restrict to saccades whose origin fixation (from_time)
                  occurs at/after this timestamp — used to isolate saccades that
                  happen only after the second sentence has already been seen.

        Returns (n_horizontal, n_vertical).
        """
        if saccades.empty:
            return 0, 0

        mask = pd.Series(True, index=saccades.index)
        if part_nr is not None:
            mask &= (saccades["from_part"] == part_nr) & (saccades["to_part"] == part_nr)
        if min_time is not None:
            mask &= (saccades["from_time"] >= min_time)

        subset = saccades[mask]
        n_horizontal = int((subset["saccade_type"] == "within_line_regression").sum())
        n_vertical   = int((subset["saccade_type"] == "line_regression").sum())
        return n_horizontal, n_vertical

    def _fixation_counts_per_sentence(self, df_position, pres):
        """Return (n_fixations_sentence1, n_fixations_sentence2) — the number of
        valid, line-matched fixations landing in each part of the presentation."""
        try:
            fixations = self.assign_line_to_fixations(df_position, pres)
        except (ValueError, KeyError):
            return 0, 0

        if fixations.empty:
            return 0, 0

        counts = fixations["part_nr"].value_counts()
        return int(counts.get(1, 0)), int(counts.get(2, 0))

    def gaze_fix_stats(self, df_position, n_pres=97) -> pd.DataFrame:
        """Return a DataFrame with detailed, per-sentence fixation and
        regression statistics for every presentation.

        For each presentation this reports, per sentence (part 1 / part 2):
        - total fixation count
        - total horizontal regressions (within-line, leftward re-fixations)
        - total vertical regressions (upward, line-to-line re-fixations)

        It additionally reports, for sentence 1 only, the subset of horizontal
        and vertical regressions that occur strictly *after* sentence 2 has
        already been fixated at least once — i.e. re-reading behavior that
        happens once both sentences have been seen. This used to be the only
        thing this method measured; it's now one column group among several,
        rather than something baked into every count.

        Also reports the number of saccades that jump directly from sentence 2
        back into sentence 1 (cross-sentence regressions), and the raw count of
        part-1 gaze samples collected after sentence 2 first appeared.
        """
        results = self.gaze_on_screen(df_position)
        reg_gaze = self.get_reg_gaze()

        rows = []
        for pres in range(n_pres):
            n_fix_s1, n_fix_s2 = self._fixation_counts_per_sentence(df_position, pres)

            try:
                saccades = self.saccades_movement(pres, df_position=df_position)
            except (ValueError, KeyError):
                saccades = pd.DataFrame(columns=["from_part", "to_part", "from_time", "saccade_type"])

            horiz_s1, vert_s1 = self._saccade_type_counts(saccades, part_nr=1)
            horiz_s2, vert_s2 = self._saccade_type_counts(saccades, part_nr=2)

            t_part2_start = self.get_part2_start_time(pres, df_position, results=results)
            if t_part2_start is not None:
                horiz_s1_after, vert_s1_after = self._saccade_type_counts(
                    saccades, part_nr=1, min_time=t_part2_start
                )
            else:
                horiz_s1_after, vert_s1_after = 0, 0

            try:
                cross_sentence_regressions = len(
                    self.filter_part2_to_part1_saccades(pres, df_position=df_position, saccades=saccades)
                )
            except (ValueError, KeyError):
                cross_sentence_regressions = 0

            reg_gaze_pres = reg_gaze.get((pres, 1), {})
            gaze_samples_sentence1_after_sentence2 = sum(len(df) for df in reg_gaze_pres.values())

            rows.append({
                "presentation": pres,

                "n_fixations_sentence1": n_fix_s1,
                "n_fixations_sentence2": n_fix_s2,

                "horizontal_regressions_sentence1": horiz_s1,
                "vertical_regressions_sentence1": vert_s1,
                "horizontal_regressions_sentence2": horiz_s2,
                "vertical_regressions_sentence2": vert_s2,

                "horizontal_regressions_sentence1_after_sentence2": horiz_s1_after,
                "vertical_regressions_sentence1_after_sentence2": vert_s1_after,

                "cross_sentence_regressions": cross_sentence_regressions,
                "gaze_samples_sentence1_after_sentence2": gaze_samples_sentence1_after_sentence2,
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

    # ── Pupil preprocessing ──────────────────────────────────────────────────

    @staticmethod
    def _mean_pupil_per_row(df):
        """Per-row mean pupil diameter across both eyes, using only valid samples.

        If both eyes are valid for a row, averages them. If only one eye is valid,
        uses that one. If neither is valid, returns NaN for that row.
        """
        left  = df["LPD"].where(df["LPV"] == 1)
        right = df["RPD"].where(df["RPV"] == 1)
        return pd.concat([left, right], axis=1).mean(axis=1, skipna=True)

    def get_pupil_baseline(self, pres, window_size=0.5, step=0.05, min_start=0.75):
        """
        ----------
        window_size : float
            Duration (s) of each candidate plateau window.
        step : float
            Step size (s) for sliding the candidate window.
        min_start : float
            Earliest start time (s, relative to fixation onset) a candidate
            window is allowed to begin at, to exclude the initial adaptation
            period.

        Returns
        -------
        float
            Mean pupil diameter (both eyes, valid samples only) over the
            detected plateau window. np.nan if no valid samples are found.
        """
        baseline_segment = self.gaze_during_fixation(pres).sort_values("TIME")
        pupil = self._mean_pupil_per_row(baseline_segment)

        valid = pd.DataFrame({
            "TIME": baseline_segment["TIME"].values,
            "pupil": pupil.values,
        }).dropna()

        if valid.empty:
            return np.nan

        t0 = valid["TIME"].iloc[0]
        valid["t_rel"] = valid["TIME"] - t0
        t_end = valid["t_rel"].iloc[-1]

        # Not enough data to slide a window — fall back to the plain mean.
        if t_end < window_size:
            return valid["pupil"].mean()

        latest_start = t_end - window_size
        starts = np.arange(min(min_start, latest_start), latest_start, step)
        if len(starts) == 0:
            starts = [max(0, latest_start)]

        candidates = []
        for start in starts:
            end = start + window_size
            w = valid[(valid["t_rel"] >= start) & (valid["t_rel"] < end)]
            if len(w) < 3:
                continue
            slope, _ = np.polyfit(w["t_rel"], w["pupil"], 1)
            candidates.append((start, abs(slope), w["pupil"].mean()))

        if not candidates:
            # Fallback: just use the last window_size seconds.
            w = valid[valid["t_rel"] >= latest_start]
            return w["pupil"].mean() if not w.empty else valid["pupil"].mean()

        cand_df = pd.DataFrame(candidates, columns=["start", "abs_slope", "mean_pupil"])
        tolerance = cand_df["abs_slope"].min() * 1.1 + 1e-9  # 10% tolerance band above the flattest slope
        flattest = cand_df[cand_df["abs_slope"] <= tolerance]
        best = flattest.sort_values("start", ascending=False).iloc[0]  # prefer the latest (closest to sentence onset)

        return best["mean_pupil"]

    def get_normalized_pupil_timecourse(self, pres):
        """Per-sample pupil dilation during the sentence-reading window of `pres`,
        normalized so trials/sentences become comparable:

        - norm_pupil_raw : absolute change from this trial's baseline
                           (raw - baseline), same units as the original PD.
        - norm_pupil_pct : percent change from this trial's baseline
                           (raw - baseline) / baseline * 100
        - time_pct       : position in the reading window as a percentage (0-100),
                           so sentences of different durations can be aligned/averaged
                           on a shared x-axis.
        """
        baseline = self.get_pupil_baseline(pres)
        segment = self.gaze_during_pres(pres)

        if segment.empty:
            return pd.DataFrame(columns=[
                "pres", "TIME", "time_pct", "raw_pupil", "baseline",
                "norm_pupil_raw", "norm_pupil_pct"
            ])

        raw_pupil = self._mean_pupil_per_row(segment)

        t_start = segment["TIME"].iloc[0]
        t_end   = segment["TIME"].iloc[-1]
        duration = t_end - t_start

        time_pct = ((segment["TIME"] - t_start) / duration * 100) if duration else np.nan
        norm_pupil_raw = (raw_pupil - baseline) if pd.notna(baseline) else np.nan
        norm_pupil_pct = ((raw_pupil - baseline) / baseline * 100) if baseline else np.nan

        out = pd.DataFrame({
            "pres"          : pres,
            "TIME"          : segment["TIME"].values,
            "time_pct"      : time_pct.values if hasattr(time_pct, "values") else time_pct,
            "raw_pupil"     : raw_pupil.values,
            "baseline"      : baseline,
            "norm_pupil_raw": norm_pupil_raw.values if hasattr(norm_pupil_raw, "values") else norm_pupil_raw,
            "norm_pupil_pct": norm_pupil_pct.values if hasattr(norm_pupil_pct, "values") else norm_pupil_pct,
        })

        return out.dropna(subset=["raw_pupil"]).reset_index(drop=True)

    def resample_pupil_to_pct_grid(self, pres, n_points=101):
        """Resample this presentation's normalized pupil timecourse onto a fixed
        percentage grid (0, 1, ..., 100 by default), via linear interpolation.

        This is what makes trials comparable point-by-point: instead of each
        presentation having pupil values at whatever irregular time_pct its own
        samples happened to land on, every presentation now has exactly one
        pupil value at each percentage point on a shared grid. That's what lets
        you average/compare across sentences and presentations at matching
        percentages later.

        Returns a dataframe with columns: pres, time_pct, norm_pupil_raw, norm_pupil_pct
        (one row per grid point). Rows are NaN where interpolation isn't
        possible (e.g. presentation had no valid pupil samples).
        """
        df_pres = self.get_normalized_pupil_timecourse(pres)
        grid = np.linspace(0, 100, n_points)

        if df_pres.empty or df_pres["norm_pupil_pct"].dropna().empty:
            return pd.DataFrame({
                "pres": pres,
                "time_pct": grid,
                "norm_pupil_raw": np.nan,
                "norm_pupil_pct": np.nan,
            })

        valid = df_pres.dropna(subset=["norm_pupil_pct"]).sort_values("time_pct")

        interp_raw = np.interp(
            grid,
            valid["time_pct"].values,
            valid["norm_pupil_raw"].values,
            left=np.nan, right=np.nan,   # don't extrapolate beyond observed range
        )
        interp_pct = np.interp(
            grid,
            valid["time_pct"].values,
            valid["norm_pupil_pct"].values,
            left=np.nan, right=np.nan,
        )

        return pd.DataFrame({
            "pres": pres,
            "time_pct": grid,
            "norm_pupil_raw": interp_raw,
            "norm_pupil_pct": interp_pct,
        })

    def pupil_timecourse_all_presentations(self, n_pres=None):
        """Baseline- and time-normalized pupil timecourse for every presentation
        of this participant, stacked into one tidy dataframe (irregular time_pct,
        one row per raw gaze sample — not grid-aligned).
        """
        if n_pres is None:
            n_pres = self.n_presentations()

        all_rows = []
        for pres in range(n_pres):
            try:
                df_pres = self.get_normalized_pupil_timecourse(pres)
            except (ValueError, KeyError) as e:
                print(f"Warning: skipping presentation {pres} ({e})")
                continue
            if not df_pres.empty:
                all_rows.append(df_pres)

        if not all_rows:
            return pd.DataFrame(columns=[
                "pres", "TIME", "time_pct", "raw_pupil", "baseline",
                "norm_pupil_raw", "norm_pupil_pct"
            ])

        return pd.concat(all_rows, ignore_index=True)

    def pupil_timecourse_grid_all_presentations(self, n_pres=None, n_points=101):
        """Grid-resampled pupil timecourse for every presentation, stacked.

        Every presentation has one row per shared percentage point (0-100),
        so you can group by time_pct and get a mean/CI pupil value at each
        percentage across all trials — this is the dataframe you want for
        cross-sentence / cross-presentation comparison.
        """
        if n_pres is None:
            n_pres = self.n_presentations()

        all_rows = []
        for pres in range(n_pres):
            try:
                df_grid = self.resample_pupil_to_pct_grid(pres, n_points=n_points)
            except (ValueError, KeyError) as e:
                print(f"Warning: skipping presentation {pres} ({e})")
                continue
            all_rows.append(df_grid)

        return pd.concat(all_rows, ignore_index=True)

    def plot_pupil_during_fixation(self, pres, show_mean=True, window_size=0.5, step=0.05, min_start=0.75):
        """Plot raw pupil diameter (both eyes) during the fixation-cross period
        preceding a presentation, with the detected plateau baseline window
        shaded, so you can visually confirm the plateau detection is sensible.
        """
        segment = self.gaze_during_fixation(pres)

        if segment.empty:
            print(f"No fixation-period gaze data for presentation {pres}")
            return

        t0 = segment["TIME"].iloc[0]

        left  = segment[segment["LPV"] == 1]
        right = segment[segment["RPV"] == 1]

        plt.figure(figsize=(9, 4))
        plt.plot(left["TIME"] - t0, left["LPD"], marker="o", markersize=3, alpha=0.7, label="Left")
        plt.plot(right["TIME"] - t0, right["RPD"], marker="o", markersize=3, alpha=0.7, label="Right")

        if show_mean:
            baseline = self.get_pupil_baseline(pres, window_size=window_size, step=step, min_start=min_start)
            if pd.notna(baseline):
                plt.axhline(baseline, color="grey", linestyle="--", linewidth=1,
                            label=f"plateau baseline ({baseline:.3f})")

        plt.title(f"Pupil diameter during fixation period — presentation {pres}")
        plt.xlabel("Time since fixation onset (s)")
        plt.ylabel("Pupil diameter (mm)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def summary(self):
        """Print a quick overview of the gaze recording."""
        print(f"Total samples    : {len(self.df)}")
        print(f"Duration         : {round(self.df['TIME'].max(), 2)} s")
        print(f"Valid fixations  : {self.df['FPOGV'].sum()}")
        print(f"User events      : {self.df['USER'].notna().sum()}")
        print(f"\nFixation stats:\n{self.fixation_stats()}")
        print(f"\nPupil stats:\n{self.pupil_stats()}")

    def sentence_durations(self):
        """Return the duration (s) spent on each sentence presentation.

        Onset/offset triggers can fire multiple times per presentation (e.g. logged
        throughout the reading window rather than once), so we take the FIRST onset
        and the FIRST offset per presentation before computing duration.
        """
        onsets  = self.sentence_onsets()
        offsets = self.sentence_offsets()

        onsets["pres"]  = onsets["USER"].str.extract(r"PRES(\d+)").astype(int)
        offsets["pres"] = offsets["USER"].str.extract(r"PRES(\d+)").astype(int)

        # collapse to one row per pres: the earliest onset, the earliest offset
        onsets_first  = (onsets.sort_values("TIME")
                                .drop_duplicates("pres", keep="first")
                                [["pres", "TIME"]]
                                .rename(columns={"TIME": "onset"}))
        offsets_first = (offsets.sort_values("TIME")
                                .drop_duplicates("pres", keep="first")
                                [["pres", "TIME"]]
                                .rename(columns={"TIME": "offset"}))

        merged = pd.merge(onsets_first, offsets_first, on="pres")
        merged["duration"] = (merged["offset"] - merged["onset"]).round(2)
        return merged.sort_values("pres").reset_index(drop=True)

    def sentence_durations_real_index(self, pres_to_real_map):
        """Same as sentence_durations(), but with each row's canonical real_index
        attached via the participant-specific presentation->real mapping.

        pres_to_real_map: dict-like (e.g. pandas Series or dict) mapping
                        presentation_index -> real_index, as produced by
                        PresentationMapper.get_pres_to_real_map().
        """
        durations = self.sentence_durations()
        durations["real_index"] = durations["pres"].map(pres_to_real_map)

        unmapped = durations["real_index"].isna().sum()
        if unmapped:
            print(f"Warning: {unmapped} presentation(s) had no real_index mapping.")

        return durations
    
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

    def _fontsize_for_box(self, fig, ax, text, box_width_px, box_height_px, font_px=None, tolerance=0.15):
        """Return the font size (in points) to render `text` at, using the known
        experiment font metrics (FONT_SIZE) rather than estimating from the box.

        The box was measured directly in-browser at FONT_SIZE/CHAR_W, so we don't
        derive a size from box_width_px — we just convert FONT_SIZE to points.
        We do, however, sanity-check that the box's actual width/height are
        consistent with what FONT_SIZE/CHAR_W/LINE_H predict, and warn loudly if
        not (this would indicate a constants.py / CSS mismatch, not a plotting bug).
        """
        if not text:
            return self._px_to_fontsize(fig, ax, FONT_SIZE)

        n_chars = len(text)
        expected_width_px = n_chars * CHAR_W
        expected_height_px = LINE_H

        width_ratio = box_width_px / expected_width_px if expected_width_px else float("nan")
        height_ratio = box_height_px / expected_height_px if expected_height_px else float("nan")

        if not (1 - tolerance <= width_ratio <= 1 + tolerance):
            print(
                f"[_fontsize_for_box] WARNING: box width {box_width_px:.1f}px vs "
                f"expected {expected_width_px:.1f}px ({n_chars} chars x CHAR_W={CHAR_W}) "
                f"— ratio {width_ratio:.2f}. Check constants.py against the CSS used "
                f"in measure_sentence_positions.html. Text: {text!r}"
            )
        if not (1 - tolerance <= height_ratio <= 1 + tolerance):
            print(
                f"[_fontsize_for_box] WARNING: box height {box_height_px:.1f}px vs "
                f"expected LINE_H={expected_height_px}px — ratio {height_ratio:.2f}."
            )

        return self._px_to_fontsize(fig, ax, FONT_SIZE)

    def _draw_spaced_text(self, ax, text, left, right, y_center, fontsize_pt, **text_kwargs):
        """Draw text with characters evenly spaced to exactly fill [left, right],
        matching the browser's letter-spacing rather than matplotlib's default
        (zero) character spacing."""
        n_chars = len(text)
        if n_chars == 0:
            return
        box_width = right - left
        advance = box_width / n_chars  # width allotted to each character, incl. spacing
        for i, ch in enumerate(text):
            x = left + (i + 0.5) * advance  # center of this character's slot
            ax.text(
                x, y_center, ch,
                fontsize=fontsize_pt, family="Courier New",
                ha="center", va="center", zorder=2,
                **text_kwargs
            )

    def plot_saccades_on_text(
        self,
        df_position,
        pres,
        saccades=None,
        show_gaze=False,
        results=None,
        font_px=40,
        show_cross_sentence=True,
        show_within_sentence=True,
        within_sentence_parts=(1, 2),
        after_sentence2_only=False,
    ):
        """Plot saccades over the presented text.

        show_cross_sentence   : draw saccades that jump directly from a sentence-2
                                 fixation back into sentence 1 (cross_sentence_regressions).
        show_within_sentence  : draw regressions (horizontal within-line + vertical
                                 line-to-line) that stay inside a single sentence.
        within_sentence_parts : which sentence(s) — 1, 2, or (1, 2) — to draw
                                 within-sentence regressions for.
        after_sentence2_only  : if True, restrict sentence-1 within-sentence
                                 regressions to those occurring after sentence 2
                                 has already been fixated at least once (i.e. the
                                 old hardcoded "re-reading" behaviour). If False
                                 (default), sentence-1 within-sentence regressions
                                 are shown regardless of timing. Has no effect on
                                 sentence 2, which can only be reached after
                                 sentence 2 has already appeared.
        """
        SACCADE_COLORS = {
            "cross_sentence"            : "crimson",
            "within_sentence_horizontal": "orange",
            "within_sentence_vertical"  : "purple",
        }

        df_pres = df_position[df_position["presentation_index"] == pres].copy()
        if saccades is None:
            saccades = self.saccades_movement(pres, df_position=df_position)

        if results is None:
            results = self.gaze_on_screen(df_position)

        # Saccades that jump straight from a part-2 fixation to a part-1 fixation
        cross_sentence = (
            self.filter_part2_to_part1_saccades(pres, df_position=df_position, saccades=saccades)
            if show_cross_sentence else saccades.iloc[0:0]
        )

        # Within-sentence regressions (horizontal + vertical), for the requested part(s)
        within_sentence = saccades.iloc[0:0].copy()
        if show_within_sentence:
            t_part2_start = self.get_part2_start_time(pres, df_position, results=results)
            parts = (
                within_sentence_parts
                if isinstance(within_sentence_parts, (tuple, list))
                else (within_sentence_parts,)
            )
            for part_nr in parts:
                mask = (
                    (saccades["from_part"] == part_nr) & (saccades["to_part"] == part_nr)
                    & (saccades["saccade_type"].isin(["within_line_regression", "line_regression"]))
                )
                if part_nr == 1 and after_sentence2_only:
                    if t_part2_start is None:
                        mask &= False
                    else:
                        mask &= (saccades["from_time"] >= t_part2_start)
                within_sentence = pd.concat([within_sentence, saccades[mask]])
            within_sentence = within_sentence.drop_duplicates()

        fig_width = 12
        fig_height = fig_width * (SCREEN_H / SCREEN_W)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_xlim(0, SCREEN_W)
        ax.set_ylim(0, SCREEN_H)
        ax.invert_yaxis()

        title_bits = []
        if show_cross_sentence:
            title_bits.append("cross-sentence regressions")
        if show_within_sentence:
            scope = "after sentence 2 only" if after_sentence2_only else "all"
            title_bits.append(f"within-sentence regressions ({scope})")
        ax.set_title(f"Saccades — presentation {pres} ({', '.join(title_bits) or 'none selected'})")
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
            self._draw_spaced_text(
                ax, line["text"], line["left"], line["right"],
                (line["top"] + line["bottom"]) / 2,
                fontsize_pt, color="#111111"
            )

        if show_gaze:
            for (p, part_nr), line_dict in results.items():
                if p != pres:
                    continue
                for gaze in line_dict.values():
                    if not gaze.empty:
                        ax.scatter(gaze["BPOGX"] * SCREEN_W, gaze["BPOGY"] * SCREEN_H,
                                s=8, alpha=0.15, color="gray", zorder=2)

        fixations = self.assign_line_to_fixations(df_position, pres)

        def _draw_saccades(sac_df, color_key):
            color = SACCADE_COLORS[color_key]
            for _, sac in sac_df.iterrows():
                prev = fixations[fixations["FPOGID"] == sac["from_fpogid"]].iloc[0]
                curr = fixations[fixations["FPOGID"] == sac["to_fpogid"]].iloc[0]
                x0, y0 = prev["FPOGX"] * SCREEN_W, prev["FPOGY"] * SCREEN_H
                x1, y1 = curr["FPOGX"] * SCREEN_W, curr["FPOGY"] * SCREEN_H
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=color, alpha=0.8, lw=1.3), zorder=4)

        _draw_saccades(cross_sentence, "cross_sentence")
        if show_within_sentence:
            _draw_saccades(
                within_sentence[within_sentence["saccade_type"] == "within_line_regression"],
                "within_sentence_horizontal",
            )
            _draw_saccades(
                within_sentence[within_sentence["saccade_type"] == "line_regression"],
                "within_sentence_vertical",
            )

        from matplotlib.lines import Line2D
        active_colors = {}
        if show_cross_sentence:
            active_colors["cross_sentence"] = SACCADE_COLORS["cross_sentence"]
        if show_within_sentence:
            active_colors["within_sentence_horizontal"] = SACCADE_COLORS["within_sentence_horizontal"]
            active_colors["within_sentence_vertical"] = SACCADE_COLORS["within_sentence_vertical"]
        legend_elems = [Line2D([0], [0], color=c, lw=2, label=t.replace("_", " "))
                        for t, c in active_colors.items()]
        if legend_elems:
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

    def plot_gaze_on_text(
        self,
        df_position,
        pres,
        results=None,
        time_col="TIME",
        font_px=40,
        split_by_sentence2_onset=True,
    ):
        """Plot gaze samples over the presented text.

        split_by_sentence2_onset : if True (default), gaze samples are split into
            'before sentence 2 appeared' (grey) and 'from sentence 2 onset onward'
            (red), using the earliest gaze landing on any sentence-2 box as the
            cutoff — matching the "after sentence 2" scope used elsewhere. If
            False, all gaze samples are plotted in a single color with no
            before/after distinction.
        """
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
        t_part2_start = None
        if split_by_sentence2_onset:
            part2_line_results = results.get((pres, 2), {})
            part2_timestamps = [df_line[time_col].min() for df_line in part2_line_results.values() if not df_line.empty]

            if not part2_timestamps:
                print(f"[pres {pres}] WARNING: no gaze landed on part 2 boxes — can't determine part 2 start, nothing highlighted.")
            else:
                t_part2_start = min(part2_timestamps)

        # ── Split gaze into before / after part 2 start (if requested) ──
        if t_part2_start is not None:
            before = all_gaze[all_gaze[time_col] < t_part2_start]
            after  = all_gaze[all_gaze[time_col] >= t_part2_start]
            print(f"[pres {pres}] total gaze: {len(all_gaze)} | before part 2: {len(before)} | from part 2 start onward: {len(after)}")
        else:
            before = all_gaze
            after = pd.DataFrame(columns=all_gaze.columns)
            print(f"[pres {pres}] total gaze: {len(all_gaze)}")

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
                self._draw_spaced_text(
                    ax, line["text"], line["left"], line["right"],
                    (line["top"] + line["bottom"]) / 2,
                    fontsize_pt, color="#111111"
                )

        if split_by_sentence2_onset:
            # Gaze before part 2 starts — grey
            ax.scatter(
                before["BPOGX"] * SCREEN_W,
                before["BPOGY"] * SCREEN_H,
                s=15, alpha=0.4, color="grey", zorder=3, label="before part 2"
            )
            # Gaze from part 2 start onward — red
            if not after.empty:
                ax.scatter(
                    after["BPOGX"] * SCREEN_W,
                    after["BPOGY"] * SCREEN_H,
                    s=15, alpha=0.35, color="red", zorder=4, label="from part 2 start"
                )
        else:
            # No before/after distinction — all gaze in one color
            ax.scatter(
                all_gaze["BPOGX"] * SCREEN_W,
                all_gaze["BPOGY"] * SCREEN_H,
                s=15, alpha=0.4, color="steelblue", zorder=3, label="gaze"
            )

        ax.legend(loc="upper right")
        plt.tight_layout()
        plt.show()