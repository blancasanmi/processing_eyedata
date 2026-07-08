from typing import List
from utils import Experiment
from gaze import GazeData
import re
import pandas as pd

class PresentationMapper:
    FIRST_COL = "first"
    REAL_IDX_COL = "real_index"
    PRES_IDX_COL = "presentation_index"

    def __init__(self, real_order_path: str, exp_order_path: str):
        self.sentence_order = self._load_sentence_order(real_order_path)
        self.sentence_to_real_idx = self._build_sentence_lookup(self.sentence_order)

        self.exp_order = pd.read_csv(exp_order_path)
        self._add_real_and_pres_idx()  # populates real_index + presentation_index on exp_order

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
                continue  # keep the first occurrence
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
        """real_index -> presentation_index, derived straight from exp_order."""
        return dict(zip(self.exp_order[self.REAL_IDX_COL], self.exp_order[self.PRES_IDX_COL]))

    def get_pres_to_real_map(self) -> dict:
        """presentation_index -> real_index, derived straight from exp_order."""
        real_to_pres = self.get_real_to_pres_map()
        if len(set(real_to_pres.values())) != len(real_to_pres):
            raise ValueError("presentation_index is not unique in exp_order — cannot invert mapping")
        return {pres: real for real, pres in real_to_pres.items()}

    def adapt_df_pos(self, df_pos: pd.DataFrame) -> pd.DataFrame:
        """Attach real_index to an external df that already has presentation_index."""
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
        return df


class DataAggregator:
    def __init__(self, gazedata: List[GazeData], experimentdata: List[Experiment], df_pos_real: pd.DataFrame):
        self.gazedata = gazedata
        self.experimentdata = experimentdata
        self.sentence_position = df_pos_real
        
        self.df_pos = self.position_translator() # I hve to use df_pos for sure 
        self.mapper = self.mapper_maker()

    ## getters and setters
    def get_sentence_pos(self):
        return self.sentence_position

    def get_df_pos(self):
        return self.df_pos

    def get_map(self, pilot_nr: int):
        if pilot_nr < 0 or pilot_nr >= len(self.mapper):
            raise IndexError(f"Pilot number {pilot_nr} is out of range. Must be between 0 and {len(self.mapper)-1}.")
        return self.mapper[pilot_nr - 1]
    
    def position_translator(self):
        df_pos = self.get_sentence_pos()
        map = self.get_map(2) # the file is made from this 
        df_pos["real_index"] = map[df_pos["presentation_index"]]
        return df_pos

    def mapper_maker(self) -> List[dict]:
        """
        Returns a list of mappers, one per participant.
        Each mapper: {presentation_index -> real_index}
        """
        sentence_to_idx = {sentence: i for i, sentence in enumerate(self.sentence_order)}
        
        mappers = []
        for exp in self.experimentdata:
            if type(exp) is not Experiment:
                raise TypeError(f"Expected Experiment, got {type(exp)}")
            
            df_exp = exp.get_sentencefirst_and_idx()
            
            mapper = {}
            for _, row in df_exp.iterrows():
                real_idx = sentence_to_idx.get(row["sentence_first"], None)
                if real_idx is None:
                    print(f"Warning: sentence not found in SENTENCE_FIRST: {row['sentence_first']}")
                mapper[row["presentation_index"]] = real_idx
            
            mappers.append(mapper)
        
        return mappers  # mappers[0][presentation_index] -> real_index for participant 0
    






