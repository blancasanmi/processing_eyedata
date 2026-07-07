import pandas as pd
import math
from constants import (
    FONT_SIZE,
    SCREEN_H,
    SCREEN_W,
    CHAR_W,
    LINE_H,
    GAP,
    TEXT_MAX_W,
)


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


class Part:
    def __init__(self, pres_nr: int, part_nr: int, part_str: str):
        self.part_str = part_str
        self.pres_nr  = pres_nr
        self.part_nr  = part_nr

        self.lines = None

    def get_part(self) -> str:
        return self.part_str

    # change this - it has to be able to recognize how the strings are divided on the screen 
    def get_lines(self) -> list[str]:
        chars_per_line = int(TEXT_MAX_W / CHAR_W)
        words = self.part_str.split()

        lines = []
        current_line = ""

        for word in words:
            # +1 for the space before the word (except at line start)
            candidate = word if current_line == "" else current_line + " " + word

            if len(candidate) <= chars_per_line:
                current_line = candidate
            else:
                # current line is full, save it and start a new one
                if current_line:
                    lines.append(current_line)
                current_line = word

        # don't forget the last line
        if current_line:
            lines.append(current_line)

        self.lines = lines

        return lines
        
    def lines_count(self):
        if self.lines != None:
            return len(self.lines)
        return len(self.get_lines())

    def height(self) -> float:
        return self.lines_count() * (LINE_H - 1) + self.lines_count() * FONT_SIZE # this height refers to the lines in between I think
    
    def boxes(self, sentence_top) -> list[Box]:
        container_left = (SCREEN_W - TEXT_MAX_W) / 2
        container_right = container_left + TEXT_MAX_W

        boxes = []
        for line_idx, line_text in enumerate(self.get_lines()):
            top    = sentence_top + line_idx * (LINE_H + FONT_SIZE)
            bottom = top + FONT_SIZE

            line_width_px = len(line_text) * CHAR_W
            left  = container_left + (TEXT_MAX_W - line_width_px) / 2
            right = left + line_width_px

            box = Box(self.pres_nr, self.part_nr, line_idx)
            box.fill_box(top=top, bottom=bottom, left=left, right=right)
            boxes.append(box)

        return boxes