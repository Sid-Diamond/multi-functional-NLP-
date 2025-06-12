from pathlib import Path
import os
import random
import json
import pandas as pd
from datasets import load_dataset, Dataset

# -----------------------------------------------------------------------------
# Central helper for all data‑folder assets
# -----------------------------------------------------------------------------
class DataFiles:
    """Resolve any file that lives under <project_root>/data."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ → project root

    @staticmethod
    def path(fname: str) -> Path:
        """Return a Path object for <project_root>/data/fname."""
        return DataFiles.PROJECT_ROOT / "data" / fname


class DatasetHandler:
    """Handles CSV I/O and quick dataset sampling."""

    def __init__(self, csv_file: str = "output.csv"):
        self.csv_file = csv_file

    # ---------------------------------------------------------------------
    # Metadata / JSON helpers
    # ---------------------------------------------------------------------
    def load_dataset_config(self, dataset_name: str) -> dict:
        """Load the dataset‑specific block from *data/dataset_configs.json*."""
        cfg_path = DataFiles.path("dataset_configs.json")
        with cfg_path.open("r", encoding="utf-8") as f:
            configs = json.load(f)
        if dataset_name not in configs:
            raise ValueError(f"Dataset configuration for '{dataset_name}' not found.")
        return configs[dataset_name]

    # ---------------------------------------------------------------------
    # CSV initialisation
    # ---------------------------------------------------------------------
    def initialize_csv(
        self,
        your_dataset_toggle: bool = False,
        our_datasets_toggle: bool = False,
        our_datasets: str | None = None,
        sample_size: int = 50,
        shuffle_data: bool = True,
        split: str = "test",
    ) -> None:
        """Create *output.csv* from either your own or our reference datasets."""

        uncleaned_texts: list[str] = []
        time_fields: list[str] = []
        source_fields: list[str] = []

        if not your_dataset_toggle and not our_datasets_toggle:
            print("Either your_dataset_toggle or our_datasets_toggle must be True.")
            return

        # -------------------------------
        # 1) User‑provided dataset
        # -------------------------------
        if your_dataset_toggle:
            cfg = self.load_dataset_config("your_dataset")
            csv_fname = cfg.get("csv_file", "your_dataset.csv")
            csv_path = DataFiles.PROJECT_ROOT / csv_fname  # user CSV sits outside data/
            if not csv_path.exists():
                raise FileNotFoundError(f"Your dataset CSV '{csv_path}' not found.")
            df = pd.read_csv(csv_path)
            text_col = cfg.get("text_field", "Text")
            if text_col not in df.columns:
                raise KeyError(
                    f"Text column '{text_col}' specified in dataset_configs.json not found in your dataset."
                )

            uncleaned_texts.extend(df[text_col].astype(str).tolist())
            # time / source handling
            time_col = cfg.get("time_field")
            source_col = cfg.get("source_field")
            default_source = cfg.get("default_source_value", "your_dataset")

            if time_col and time_col in df.columns:
                time_fields.extend(df[time_col].astype(str).tolist())
            else:
                time_fields.extend([""] * len(df))

            if source_col and source_col in df.columns:
                source_fields.extend(df[source_col].astype(str).tolist())
            else:
                source_fields.extend([default_source] * len(df))

        # -------------------------------
        # 2) Reference (HuggingFace/local) datasets
        # -------------------------------
        if our_datasets_toggle and our_datasets:
            cfg = self.load_dataset_config(our_datasets)
            data_source = cfg.get("data_source", "huggingface")

            if data_source == "local_csv":
                local_csv = DataFiles.PROJECT_ROOT / cfg.get("csv_file", "")
                if not local_csv.exists():
                    raise FileNotFoundError(f"CSV file '{local_csv}' not found.")
                df_local = pd.read_csv(local_csv)
                dataset = Dataset.from_pandas(df_local)
            else:
                dataset = load_dataset(our_datasets, split=split, trust_remote_code=True)

            dataset = self.sample_dataset(dataset, sample_size, shuffle_data)

            text_col = cfg.get("text_field")
            time_col = cfg.get("time_field")
            source_col = cfg.get("source_field")
            default_source = cfg.get("default_source_value", our_datasets)

            for ex in dataset:
                uncleaned_texts.append(ex[text_col])
                time_fields.append(ex.get(time_col, ""))
                if source_col and source_col in ex:
                    source_fields.append(ex[source_col])
                else:
                    source_fields.append(default_source)

        # -------------------------------
        # 3) Dump to CSV
        # -------------------------------
        df_out = pd.DataFrame({
            "Text": uncleaned_texts,
            "Source": source_fields,
            "Time": time_fields,
        })
        df_out.to_csv(self.csv_file, index=False)
        print(f"Initialised '{self.csv_file}' with {len(df_out)} rows.")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    @staticmethod
    def sample_dataset(dataset: Dataset, sample_size: int, shuffle_data: bool) -> Dataset:
        """Optionally shuffle then take a sample of `sample_size` rows."""
        total = len(dataset)
        if sample_size and sample_size > total:
            sample_size = total
        if shuffle_data:
            dataset = dataset.shuffle(seed=random.randint(1, 100_000))
        if sample_size:
            dataset = dataset.select(range(sample_size))
        return dataset

    # ------------------------------------------------------------------
    # CSV conveniences
    # ------------------------------------------------------------------
    def read_csv(self) -> pd.DataFrame:
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file '{self.csv_file}' not found.")
        return pd.read_csv(self.csv_file)

    def write_csv(self, df: pd.DataFrame) -> None:
        df.to_csv(self.csv_file, index=False)

    def get_texts_for_analysis(self) -> list[str]:
        return self.read_csv()["Text"].astype(str).tolist()

    def update_csv_with_results(self, results_dict: dict) -> None:
        df = self.read_csv()
        for col, vals in results_dict.items():
            df[col] = vals
        self.write_csv(df)
