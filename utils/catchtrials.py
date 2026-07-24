import pandas as pd
 
try:
    from .participant_data import load_all_participants
except ImportError:  # pragma: no cover - fallback for direct script execution
    from participant_data import load_all_participants


def add_conditions_by_sentence(catch_trials, conditions_df, catch_col="sentence_first", cond_col="Part 1"):
    """Merge catch_trials with condition labels by matching sentence text."""
    catch_trials = catch_trials.copy()
    conditions_df = conditions_df.copy()

    def normalize(s):
        if pd.isna(s):
            return s
        return " ".join(str(s).strip().split())  # strip + collapse internal whitespace

    catch_trials["_sentence_key"] = catch_trials[catch_col].apply(normalize)
    conditions_df["_sentence_key"] = conditions_df[cond_col].apply(normalize)

    condition_cols = ["_sentence_key", "Uncertainty", "Resolution", "Unc Valence", "Res Valence"]
    condition_lookup = conditions_df[condition_cols].drop_duplicates(subset=["_sentence_key"])

    merged = catch_trials.merge(condition_lookup, on="_sentence_key", how="left")

    n_unmatched = merged["Uncertainty"].isna().sum()
    if n_unmatched > 0:
        print(f"WARNING: {n_unmatched} / {len(merged)} rows did not match any condition.")
        print("Unmatched sentences:")
        print(merged.loc[merged["Uncertainty"].isna(), catch_col].unique())

    return merged.drop(columns=["_sentence_key"])
 
 
class CatchTrials:
    def __init__(self, path=None, df=None):
        """Build from a file path (reads the CSV itself) OR from an
        already-loaded DataFrame (e.g. pdata.catch_trials from the
        participant_files helper) — pass exactly one of the two."""
        if df is not None:
            self.path = path  # kept for reference, may be None
            self.df = df
        elif path is not None:
            self.path = path
            self.df = pd.read_csv(path)
        else:
            raise ValueError("CatchTrials requires either 'path' or 'df'")
 
    @classmethod
    def from_dataframe(cls, df, path=None):
        return cls(path=path, df=df)
 
    def get_catch_idx(self, idx):
        """Return all rows where catch_index == idx."""
        return self.df[self.df["catch_index"] == idx]
 
    def get_by_sentence_position(self, position):
        """Return all rows where sentence_position == position."""
        return self.df[self.df["sentence_position"] == position]
 
    def percent_correct(self):
        """Return proportion of correct responses (handles bool or string 'true'/'false')."""
        correct = self.df["correct"]
        if correct.dtype == object:
            correct = correct.str.strip().str.lower() == "true"
        return (correct.sum() / len(correct))
 
    def mean_rt(self):
        """Return mean reaction time across all trials."""
        return round(self.df["rt"].mean(), 2)
 
    def rt_stats(self):
        """Return median, IQR, and full quartile breakdown across all trials."""
        rt = self.df["rt"]
        return pd.Series({
            "median": rt.median(),
            "q1": rt.quantile(0.25),
            "q3": rt.quantile(0.75),
            "iqr": rt.quantile(0.75) - rt.quantile(0.25),
            "min": rt.min(),
            "max": rt.max(),
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