import pandas as pd

from constants import GAP, SCREEN_H
from utils.part import Part


class Experiment:
    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.df = self.read_exp_data()
        self.df_reading = self.filter_df_reading()

    def get_df(self):
        return self.df

    def get_reading_df(self):
        return self.df_reading

    def read_exp_data(self):
        return pd.read_csv(self.experiment_path)

    def filter_df_reading(self):
        df = self.get_df()
        if df is not None:
            return df[df["task"] == "reading"]
        return None
    
    def get_sentencefirst_and_idx(self):
        df = self.get_reading_df()[["sentence_first", "presentation_index"]]
        return df

    def get_presentation_row(self, pres_nr: int):
        df = self.get_reading_df()
        return df.loc[df["presentation_index"] == pres_nr]

    def get_part(self, part_nr: int, pres_nr: int) -> str:
        if part_nr != 1 and part_nr != 2:
            print(f"Error in part number: {part_nr} must be 1 or 2")
            return None
        row = self.get_presentation_row(pres_nr)
        col = "sentence_first" if part_nr == 1 else "sentence_second"
        return row[col].values[0]
    
    def boxes_on_screen(self, pres_nr: int) -> dict:
        part1 = Part(pres_nr, 1, self.get_part(1, pres_nr))
        part2 = Part(pres_nr, 2, self.get_part(2, pres_nr))

        total_block_h = part1.height() + GAP + part2.height()
        block_top     = (SCREEN_H - total_block_h) / 2

        part1.sentence_top = block_top
        part2.sentence_top = block_top + part1.height() + GAP

        return {
            "part1": part1.boxes(part1.sentence_top),
            "part2": part2.boxes(part2.sentence_top),
        }
    
    def boxes_to_csv(self, output_path: str):
        rows = []
        
        for pres_nr in self.get_reading_df()["presentation_index"].unique():
            result = self.boxes_on_screen(pres_nr)
            
            for part_key, box_list in result.items():
                part_nr = 1 if part_key == "part1" else 2
                lines = Part(pres_nr, part_nr, self.get_part(part_nr, pres_nr)).get_lines()
                
                for box, line_text in zip(box_list, lines):
                    rows.append({
                        "presentation_index": pres_nr,
                        "part_nr":            part_nr,
                        "line_idx":           box.line,
                        "text":               line_text,
                        "top":                box.top,
                        "bottom":             box.bottom,
                        "left":               box.left,
                        "right":              box.right,
                    })
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved {len(rows)} boxes to {output_path}")
        

class Box:
    def __init__(self, pres_nr: int, part_nr: int, line: int):
        self.pres_nr = pres_nr
        self.part_nr = part_nr
        self.line = line

        self.top    = None
        self.bottom = None
        self.right  = None
        self.left   = None
        self.values = {}

    def get_pres_nr(self):
        return self.pres_nr

    def get_part_nr(self):
        return self.part_nr

    def fill_box(self, top: float, bottom: float, left: float, right: float):
        self.top    = top;    self.values["top"]    = top
        self.bottom = bottom; self.values["bottom"] = bottom
        self.left   = left;   self.values["left"]   = left
        self.right  = right;  self.values["right"]  = right

    def get_box(self):
        return self.values
