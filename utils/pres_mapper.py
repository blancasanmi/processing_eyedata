import re
import warnings

import pandas as pd


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