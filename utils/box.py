class Box:
    def __init__(self, pres_nr: int, part_nr: int, line: int):
        self.pres_nr = pres_nr
        self.part_nr = part_nr
        self.line = line

        self.top = None
        self.bottom = None
        self.right = None
        self.left = None
        self.values = {}

    def get_pres_nr(self):
        return self.pres_nr

    def get_part_nr(self):
        return self.part_nr

    def fill_box(self, top: float, bottom: float, left: float, right: float):
        self.top = top
        self.values["top"] = top
        self.bottom = bottom
        self.values["bottom"] = bottom
        self.left = left
        self.values["left"] = left
        self.right = right
        self.values["right"] = right

    def get_box(self):
        return self.values
