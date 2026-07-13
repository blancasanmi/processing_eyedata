import glob
import json
import os
import re
import pandas as pd

DATA_ROOT = "data"
PARTICIPANT_FOLDER_PATTERN = "participant_*"  # data/participant_<participant_nr>/

# One entry per file type living inside each participant folder.
# 'pattern' is a glob fragment (participant_nr gets substituted in),
# 'loader' says how to read the file once found.
FILE_TYPES = {
    "calibration":     {"pattern": "*calibration*{nr}*", "loader": "json"},
    "catch_trials":    {"pattern": "*catch_trials*{nr}*", "loader": "csv"},
    "experiment":      {"pattern": "*experiment*{nr}*",   "loader": "csv"},
    "gaze":            {"pattern": "*gaze*{nr}*",         "loader": "csv"},
    "sentence_order":  {"pattern": "*sentence_order*{nr}*", "loader": "csv"},
}


def discover_participants(data_root=DATA_ROOT):
    """Find all participant folders under data_root and extract participant_nr
    from the folder name (data/participant_<nr>/)."""
    folders = sorted(glob.glob(os.path.join(data_root, PARTICIPANT_FOLDER_PATTERN)))
    participants = {}
    for folder in folders:
        m = re.search(r"participant_(\w+)$", os.path.basename(folder))
        if m:
            participants[m.group(1)] = folder
    return participants


def find_file(folder, pattern_template, participant_nr):
    """Locate a single file inside a participant folder matching pattern_template,
    with {nr} substituted for participant_nr. Returns None if nothing matches."""
    pattern = pattern_template.format(nr=participant_nr)
    candidates = glob.glob(os.path.join(folder, pattern))
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"  WARNING: multiple matches for pattern '{pattern}' in {folder}, "
              f"using {candidates[0]} — {candidates}")
    return candidates[0]


def gather_participant_files(data_root=DATA_ROOT):
    """
    Return {participant_nr: {file_type: path_or_None}} for every discovered
    participant and every entry in FILE_TYPES. Missing files are recorded as
    None (with a warning) rather than raising, so one missing file doesn't
    block the rest of the participant's data.
    """
    participants = discover_participants(data_root)
    all_files = {}
    for participant_nr, folder in participants.items():
        files = {}
        for file_type, spec in FILE_TYPES.items():
            path = find_file(folder, spec["pattern"], participant_nr)
            if path is None:
                print(f"  WARNING: no '{file_type}' file found for participant "
                      f"{participant_nr} in {folder}")
            files[file_type] = path
        all_files[participant_nr] = files
    return all_files


def _load_file(path, loader):
    if path is None:
        return None
    if loader == "json":
        with open(path, "r") as f:
            return json.load(f)
    if loader == "csv":
        return pd.read_csv(path)
    raise ValueError(f"Unknown loader '{loader}'")


class ParticipantData:
    """Lazily-loaded bundle of all files for a single participant.
    Access e.g. pdata.catch_trials, pdata.gaze, pdata.calibration — each is
    loaded from disk the first time it's accessed and cached after that.
    """

    def __init__(self, participant_nr, file_paths):
        self.participant_nr = participant_nr
        self.file_paths = file_paths  # {file_type: path_or_None}
        self._cache = {}

    def _get(self, file_type):
        if file_type not in self._cache:
            spec = FILE_TYPES[file_type]
            self._cache[file_type] = _load_file(self.file_paths.get(file_type), spec["loader"])
        return self._cache[file_type]

    @property
    def calibration(self):
        """dict loaded from the calibration JSON, or None if missing."""
        return self._get("calibration")

    @property
    def catch_trials(self):
        """DataFrame loaded from the catch_trials CSV, or None if missing."""
        return self._get("catch_trials")

    @property
    def experiment(self):
        """DataFrame loaded from the experiment CSV, or None if missing."""
        return self._get("experiment")

    @property
    def gaze(self):
        """DataFrame loaded from the gaze CSV, or None if missing."""
        return self._get("gaze")

    @property
    def sentence_order(self):
        """DataFrame loaded from the sentence_order CSV, or None if missing."""
        return self._get("sentence_order")

    def available_files(self):
        """Which file types were actually found on disk for this participant."""
        return [ft for ft, p in self.file_paths.items() if p is not None]

    def missing_files(self):
        """Which file types are missing for this participant."""
        return [ft for ft, p in self.file_paths.items() if p is None]

    def __repr__(self):
        return f"ParticipantData(participant_nr={self.participant_nr!r}, " \
               f"available={self.available_files()}, missing={self.missing_files()})"


def load_all_participants(data_root=DATA_ROOT):
    """
    Return {participant_nr: ParticipantData} for every discovered participant.
    Nothing is read from disk yet at this point — each file is loaded lazily
    the first time it's accessed via the corresponding property.
    """
    all_files = gather_participant_files(data_root)
    return {
        participant_nr: ParticipantData(participant_nr, files)
        for participant_nr, files in all_files.items()
    }


def load_participant(participant_nr, data_root=DATA_ROOT):
    """
    Return a single ParticipantData for participant_nr — meant for notebook
    use, e.g.:

        p = load_participant("3")
        p.catch_trials.head()
        p.gaze.plot(...)
        p.calibration

    Raises FileNotFoundError if no folder matches data/participant_<nr>.
    Individual files that are missing inside that folder still just come
    back as None (with a warning), same as load_all_participants().
    """
    participants = discover_participants(data_root)
    participant_nr = str(participant_nr)
    if participant_nr not in participants:
        raise FileNotFoundError(
            f"No folder found for participant '{participant_nr}' under {data_root}/"
            f"{PARTICIPANT_FOLDER_PATTERN}. Found participants: {sorted(participants.keys())}"
        )
    folder = participants[participant_nr]
    files = {
        file_type: find_file(folder, spec["pattern"], participant_nr)
        for file_type, spec in FILE_TYPES.items()
    }
    for file_type, path in files.items():
        if path is None:
            print(f"  WARNING: no '{file_type}' file found for participant "
                  f"{participant_nr} in {folder}")
    return ParticipantData(participant_nr, files)


if __name__ == "__main__":
    participants = load_all_participants()
    print(f"Discovered {len(participants)} participants: {sorted(participants.keys())}\n")

    for participant_nr, pdata in sorted(participants.items()):
        print(pdata)

    # Example: force-load everything and report shapes/sizes as a sanity check
    print("\n=== File load sanity check ===")
    for participant_nr, pdata in sorted(participants.items()):
        print(f"\nParticipant {participant_nr}:")
        print(f"  calibration     : {'loaded (dict)' if pdata.calibration is not None else 'MISSING'}")
        print(f"  catch_trials    : {pdata.catch_trials.shape if pdata.catch_trials is not None else 'MISSING'}")
        print(f"  experiment      : {pdata.experiment.shape if pdata.experiment is not None else 'MISSING'}")
        print(f"  gaze            : {pdata.gaze.shape if pdata.gaze is not None else 'MISSING'}")
        print(f"  sentence_order  : {pdata.sentence_order.shape if pdata.sentence_order is not None else 'MISSING'}")