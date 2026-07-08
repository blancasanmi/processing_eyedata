from typing import List
from gaze import GazeData
import re
import pandas as pd
import warnings


class PresentationMapper:
    FIRST_COL = "first"
    REAL_IDX_COL = "real_index"
    PRES_IDX_COL = "presentation_index"

    def __init__(self, real_order_path: str, exp_order_path: str):
        print(f"\n[PresentationMapper] Initialising...")
        print(f"  real_order_path : {real_order_path}")
        print(f"  exp_order_path  : {exp_order_path}")

        self.sentence_order = self._load_sentence_order(real_order_path)
        print(f"  Loaded {len(self.sentence_order)} sentences from real_order")
        print(f"  First sentence : {self.sentence_order[0]!r}")
        print(f"  Last sentence  : {self.sentence_order[-1]!r}")

        self.sentence_to_real_idx = self._build_sentence_lookup(self.sentence_order)

        self.exp_order = pd.read_csv(exp_order_path)
        print(f"  exp_order shape : {self.exp_order.shape}")
        print(f"  exp_order columns : {self.exp_order.columns.tolist()}")

        self._add_real_and_pres_idx()
        print(f"  Mapping complete. Sample rows:")
        print(self.exp_order[[self.PRES_IDX_COL, self.FIRST_COL, self.REAL_IDX_COL]].head(5).to_string(index=False))

    def _load_sentence_order(self, path: str) -> list:
        """Parse SENTENCE_FIRST array from JS file."""
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        match = re.search(r'const SENTENCE_FIRST\s*=\s*\[(.*?)\]', content, re.DOTALL)
        if not match:
            raise ValueError("Could not find SENTENCE_FIRST in JS file")

        array_content = match.group(1)
        sentences = re.findall(r'"([^"]*?)"|\'([^\']*?)\'', array_content)
        return [d or s for d, s in sentences]

    def _build_sentence_lookup(self, sentence_order: list) -> dict:
        """Map sentence -> real_index, warning on duplicate sentences."""
        lookup = {}
        for i, sentence in enumerate(sentence_order):
            if sentence in lookup:
                warnings.warn(
                    f"Duplicate sentence at real_index {i} "
                    f"(first seen at {lookup[sentence]}): {sentence!r}"
                )
                continue
            lookup[sentence] = i
        return lookup

    def _add_real_and_pres_idx(self):
        """Add real_index and presentation_index columns to exp_order, in place."""
        df = self.exp_order.copy()
        df[self.PRES_IDX_COL] = df.index
        df[self.REAL_IDX_COL] = df[self.FIRST_COL].map(self.sentence_to_real_idx)

        unmapped = df[df[self.REAL_IDX_COL].isna()]
        if not unmapped.empty:
            raise ValueError(
                f"{len(unmapped)} row(s) in exp_order['{self.FIRST_COL}'] did not match "
                f"any sentence in real_order, e.g.: {unmapped[self.FIRST_COL].iloc[0]!r}"
            )
        df[self.REAL_IDX_COL] = df[self.REAL_IDX_COL].astype(int)
        self.exp_order = df

    def get_exp_order(self) -> pd.DataFrame:
        return self.exp_order

    def get_real_to_pres_map(self) -> dict:
        """real_index -> presentation_index."""
        return dict(zip(self.exp_order[self.REAL_IDX_COL], self.exp_order[self.PRES_IDX_COL]))

    def get_pres_to_real_map(self) -> dict:
        """presentation_index -> real_index."""
        real_to_pres = self.get_real_to_pres_map()
        if len(set(real_to_pres.values())) != len(real_to_pres):
            raise ValueError("presentation_index is not unique — cannot invert mapping")
        return {pres: real for real, pres in real_to_pres.items()}

    def adapt_df_pos(self, df_pos: pd.DataFrame) -> pd.DataFrame:
        """Attach real_index to an external df that already has presentation_index."""
        print(f"\n[PresentationMapper.adapt_df_pos]")
        print(f"  df_pos shape: {df_pos.shape}")
        print(f"  presentation_index range: {df_pos['presentation_index'].min()} → {df_pos['presentation_index'].max()}")

        pres_to_real = self.get_pres_to_real_map()
        df = df_pos.copy()
        df[self.REAL_IDX_COL] = df["presentation_index"].map(pres_to_real)

        unmapped = df[df[self.REAL_IDX_COL].isna()]
        if not unmapped.empty:
            raise ValueError(
                f"{len(unmapped)} row(s) had a presentation_index with no matching real_index, "
                f"e.g.: {unmapped['presentation_index'].iloc[0]!r}"
            )
        df[self.REAL_IDX_COL] = df[self.REAL_IDX_COL].astype(int)

        print(f"  real_index range after mapping: {df[self.REAL_IDX_COL].min()} → {df[self.REAL_IDX_COL].max()}")
        print(f"  Sample mapping (pres → real):")
        print(df[["presentation_index", self.REAL_IDX_COL]].drop_duplicates().head(5).to_string(index=False))
        return df


class DataAggregator:
    def __init__(
        self,
        gazedata: List[GazeData],
        real_order_path: str,
        exp_order_paths: List[str],
        df_pos: pd.DataFrame,
        df_pos_participant_idx: int = 1
    ):
        print(f"\n[DataAggregator] Initialising...")
        print(f"  Number of participants : {len(gazedata)}")
        print(f"  df_pos shape           : {df_pos.shape}")
        print(f"  df_pos built from participant (0-indexed): {df_pos_participant_idx}")

        self.gazedata = gazedata
        self.df_pos = df_pos
        self.df_pos_participant_idx = df_pos_participant_idx

        print(f"\n[DataAggregator] Building mappers...")
        self.mappers: List[PresentationMapper] = [
            PresentationMapper(real_order_path, path)
            for path in exp_order_paths
        ]
        print(f"[DataAggregator] Built {len(self.mappers)} mappers")

        print(f"\n[DataAggregator] Translating df_pos for each participant...")
        self.df_pos_per_participant = self._build_df_pos_per_participant()
        print(f"[DataAggregator] df_pos translated for all participants")

    def get_mapper(self, participant_idx: int) -> PresentationMapper:
        if participant_idx < 0 or participant_idx >= len(self.mappers):
            raise IndexError(f"Participant index {participant_idx} out of range.")
        return self.mappers[participant_idx]

    def get_df_pos(self, participant_idx: int) -> pd.DataFrame:
        return self.df_pos_per_participant[participant_idx]

    def _build_df_pos_per_participant(self) -> List[pd.DataFrame]:
        # Step 1: add real_index to df_pos using reference participant's mapper
        mapper_ref = self.mappers[self.df_pos_participant_idx]
        print(f"\n  [Step 1] Adding real_index to df_pos using participant {self.df_pos_participant_idx}'s mapper")
        df_pos_with_real = mapper_ref.adapt_df_pos(self.df_pos)

        translated = []
        for i, mapper in enumerate(self.mappers):
            print(f"\n  [Step 2] Translating df_pos for participant {i}...")
            df = df_pos_with_real.copy()

            real_to_pres = mapper.get_real_to_pres_map()
            df["presentation_index"] = df[PresentationMapper.REAL_IDX_COL].map(real_to_pres)

            unmapped = df[df["presentation_index"].isna()]
            if not unmapped.empty:
                print(f"  WARNING: participant {i} has {len(unmapped)} unmapped rows in df_pos")
                print(f"  Unmapped real_indices: {df[df['presentation_index'].isna()][PresentationMapper.REAL_IDX_COL].unique()}")
            else:
                print(f"  All rows mapped successfully")

            df["presentation_index"] = df["presentation_index"].astype(int)

            # Sanity check: show a few example mappings
            print(f"  Sample (real_index → presentation_index for participant {i}):")
            print(df[[PresentationMapper.REAL_IDX_COL, "presentation_index"]]
                  .drop_duplicates()
                  .head(5)
                  .to_string(index=False))

            translated.append(df)

        return translated

    def compute_gaze_fix_stats(self) -> pd.DataFrame:
        """Run gaze_fix_stats for all participants."""
        print(f"\n[DataAggregator] Computing gaze_fix_stats for all participants...")
        rows_all = []

        for i, gaze in enumerate(self.gazedata):
            print(f"\n  Participant {i}:")
            df_pos_translated = self.get_df_pos(i)
            mapper = self.get_mapper(i)

            print(f"    Running gaze_on_screen...")
            gaze.gaze_on_screen(df_pos_translated)

            print(f"    Running gaze_fix_stats...")
            stats = gaze.gaze_fix_stats(df_pos_translated)

            pres_to_real = mapper.get_pres_to_real_map()
            stats["real_index"] = stats["presentation"].map(pres_to_real)
            stats["participant"] = i

            total_gaze     = stats["gaze_samples"].sum()
            total_fix      = stats["fixations"].sum()
            n_with_reg     = (stats["gaze_samples"] > 0).sum()
            print(f"    Total regression gaze samples : {total_gaze}")
            print(f"    Total regression fixations    : {total_fix}")
            print(f"    Presentations with regressions: {n_with_reg} / {len(stats)}")
            print(f"    Sample rows:")
            print(stats[stats["gaze_samples"] > 0].head(3).to_string(index=False))

            rows_all.append(stats)

        df_all = pd.concat(rows_all).reset_index(drop=True)
        print(f"\n[DataAggregator] Done. Combined df shape: {df_all.shape}")
        return df_all

    def aggregate_by_sentence(self) -> pd.DataFrame:
        """Aggregate gaze_fix_stats across participants by real_index."""
        print(f"\n[DataAggregator] Aggregating by sentence (real_index)...")
        df_all = self.compute_gaze_fix_stats()

        result = (
            df_all.groupby("real_index")
            .agg(
                mean_gaze_samples=("gaze_samples", "mean"),
                std_gaze_samples=("gaze_samples", "std"),
                mean_fixations=("fixations", "mean"),
                std_fixations=("fixations", "std"),
                n_participants=("participant", "count")
            )
            .reset_index()
        )

        print(f"  Aggregated {len(result)} sentences")
        print(f"  Sentences with mean_gaze_samples > 0: {(result['mean_gaze_samples'] > 0).sum()}")
        print(f"  Top 5 sentences by mean gaze samples:")
        print(result.nlargest(5, "mean_gaze_samples")[
            ["real_index", "mean_gaze_samples", "mean_fixations", "n_participants"]
        ].to_string(index=False))

        return result