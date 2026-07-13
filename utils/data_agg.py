from typing import List
from utils.gazedata import GazeData
from utils.pres_mapper import PresentationMapper
import pandas as pd


class DataAggregator:
    def __init__(
        self,
        gazedata: List[GazeData],
        real_order_path: str,
        exp_order_paths: List[str],
        df_pos: pd.DataFrame,
        sentence_metadata_path: str = None,  # optional
    ):
        self.gazedata = gazedata
        self.df_pos = df_pos  # already has a "real_index" column

        self.sentence_metadata = None
        if sentence_metadata_path is not None:
            self.sentence_metadata = self.load_sentence_metadata(sentence_metadata_path)

        self.mappers: List[PresentationMapper] = [
            PresentationMapper(real_order_path, path)
            for path in exp_order_paths
        ]

        self.df_pos_per_participant = self._build_df_pos_per_participant()

    def get_mapper(self, participant_idx: int) -> PresentationMapper:
        if participant_idx < 0 or participant_idx >= len(self.mappers):
            raise IndexError(f"Participant index {participant_idx} out of range.")
        return self.mappers[participant_idx]

    def get_df_pos(self, participant_idx: int) -> pd.DataFrame:
        return self.df_pos_per_participant[participant_idx]

    def _build_df_pos_per_participant(self) -> List[pd.DataFrame]:
        """Map real_index -> presentation_index for each participant's own order."""
        translated = []
        for i, mapper in enumerate(self.mappers):
            df = self.df_pos.copy()

            real_to_pres = mapper.get_real_to_pres_map()
            df["presentation_index"] = df[PresentationMapper.REAL_IDX_COL].map(real_to_pres)

            unmapped = df[df["presentation_index"].isna()]
            if not unmapped.empty:
                print(f"  WARNING: participant {i} has {len(unmapped)} unmapped rows in df_pos")

            df["presentation_index"] = df["presentation_index"].astype(int)
            translated.append(df)

        return translated

    def load_sentence_metadata(self, path: str) -> pd.DataFrame:
        """
        Load sentence metadata from combined_valid_sentences.xlsx.
        Renames unnamed index column to real_index.
        """
        df = pd.read_excel(path, index_col=0)
        df.index.name = "real_index"
        df = df.reset_index()
        print(f"[load_sentence_metadata] Loaded {len(df)} sentences")
        return df

    def compute_gaze_fix_stats(self) -> pd.DataFrame:
        """Run gaze_fix_stats for all participants."""
        rows_all = []

        for i, gaze in enumerate(self.gazedata):
            df_pos_translated = self.get_df_pos(i)
            mapper = self.get_mapper(i)

            gaze.gaze_on_screen(df_pos_translated)
            stats = gaze.gaze_fix_stats(df_pos_translated)

            pres_to_real = mapper.get_pres_to_real_map()
            stats["real_index"] = stats["presentation"].map(pres_to_real)
            stats["participant"] = i

            rows_all.append(stats)

        df_all = pd.concat(rows_all).reset_index(drop=True)

        if self.sentence_metadata is not None:
            meta_cols = ["real_index", "groupUnc", "groupRes", "groupVal"]
            df_all = df_all.merge(self.sentence_metadata[meta_cols], on="real_index", how="left")

            missing = df_all[df_all["groupUnc"].isna()]
            if not missing.empty:
                print(f"  WARNING: {len(missing)} rows in df_all have no metadata")

        print(f"[DataAggregator] Combined df shape: {df_all.shape}")
        return df_all

    def aggregate_by_sentence(self) -> pd.DataFrame:
        df_all = self.compute_gaze_fix_stats()

        result = (
            df_all.groupby("real_index")
            .agg(
                mean_gaze_samples=("gaze_samples", "mean"),
                std_gaze_samples=("gaze_samples", "std"),
                mean_fixations=("fixations", "mean"),
                std_fixations=("fixations", "std"),
                mean_horizontal_fixations=("horizontal_fixations", "mean"),
                std_horizontal_fixations=("horizontal_fixations", "std"),
                n_participants=("participant", "count"),
            )
            .reset_index()
        )

        if self.sentence_metadata is not None:
            meta_cols = ["real_index", "groupUnc", "groupRes", "groupVal"]
            result = result.merge(self.sentence_metadata[meta_cols], on="real_index", how="left")

            missing = result[result["groupUnc"].isna()]
            if not missing.empty:
                print(f"  WARNING: {len(missing)} sentences have no metadata")

        return result