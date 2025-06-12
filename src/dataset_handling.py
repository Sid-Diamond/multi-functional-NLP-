import os
import random
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset, Dataset

class DataFiles:
    """
    Helper to locate data files relative to the project root.
    """
    @staticmethod
    def path(filename: str) -> Path:
        # Assume this file is in src/, so project root is two levels up
        root = Path(__file__).resolve().parent.parent
        return root / filename

class DatasetHandler:
    def __init__(self, csv_file: str = 'output.csv'):
        # CSV file path is relative to project root
        self.csv_file = DataFiles.path(csv_file)

    def initialize_csv(
        self,
        your_dataset_toggle: bool = False,
        our_datasets_toggle: bool = False,
        our_datasets: str = None,
        sample_size: int = 50,
        shuffle_data: bool = True,
        split: str = 'test'
    ):
        """
        Initializes the CSV file with uncleaned texts, time fields, and source fields.
        """
        uncleaned_texts_list = []
        time_fields_list = []
        source_fields_list = []

        if not your_dataset_toggle and not our_datasets_toggle:
            print("One of your_dataset_toggle or our_datasets_toggle must be True.")
            return

        # Load 'your_dataset' from local CSV if toggled
        if your_dataset_toggle:
            config = self.load_dataset_config('your_dataset')
            csv_name = config.get('csv_file', '')
            file_path = DataFiles.path(csv_name)
            if not file_path.exists():
                raise FileNotFoundError(f"Your dataset CSV '{csv_name}' not found at {file_path}.")
            df_local = pd.read_csv(file_path)
            text_field = config.get('text_field', 'Text')
            if text_field not in df_local.columns:
                raise KeyError(f"Text field '{text_field}' missing in your_dataset CSV.")
            uncleaned_texts_list.extend(df_local[text_field].astype(str).tolist())
            # time and source
            tfield = config.get('time_field', None)
            sfield = config.get('source_field', None)
            default_src = config.get('default_source_value', 'your_dataset')
            time_fields_list.extend(df_local[tfield].astype(str).tolist() if tfield in df_local else ['']*len(df_local))
            source_fields_list.extend(df_local[sfield].astype(str).tolist() if sfield in df_local else [default_src]*len(df_local))

        # Load 'our_datasets' from Huggingface or local CSV if toggled
        if our_datasets_toggle and our_datasets:
            config = self.load_dataset_config(our_datasets)
            data_source = config.get('data_source', 'huggingface')

            if data_source == 'local_csv':
                csv_name = config.get('csv_file', '')
                file_path = DataFiles.path(csv_name)
                if not file_path.exists():
                    raise FileNotFoundError(f"CSV '{csv_name}' not found at {file_path}.")
                df_hf = pd.read_csv(file_path)
                dataset = Dataset.from_pandas(df_hf)
            else:
                dataset = load_dataset(our_datasets, split=split, trust_remote_code=True)

            dataset = self.sample_dataset(dataset, sample_size, shuffle_data)
            text_field = config['text_field']
            time_field = config.get('time_field', None)
            source_field = config.get('source_field', None)
            default_src = config.get('default_source_value', our_datasets)

            for ex in dataset:
                uncleaned_texts_list.append(ex[text_field])
                time_fields_list.append(ex.get(time_field, '') if time_field else '')
                source_fields_list.append(ex[source_field] if source_field in ex else default_src)

        # Build DataFrame and save
        df = pd.DataFrame({
            'Text': uncleaned_texts_list,
            'Source': source_fields_list,
            'Time': time_fields_list
        })
        df.to_csv(self.csv_file, index=False)
        print(f"Initialized CSV '{self.csv_file}' with {len(df)} rows.")

    def load_dataset_config(self, dataset_name: str) -> dict:
        """
        Loads dataset-specific configurations from data/dataset_configs.json.
        """
        config_fp = DataFiles.path('dataset_configs.json')
        if not config_fp.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_fp}.")
        with config_fp.open('r', encoding='utf-8') as f:
            configs = json.load(f)
        if dataset_name not in configs:
            raise ValueError(f"Dataset config '{dataset_name}' not in dataset_configs.json.")
        return configs[dataset_name]

    def sample_dataset(self, dataset: Dataset, sample_size: int, shuffle_data: bool) -> Dataset:
        """
        Sample and optionally shuffle a Huggingface Dataset.
        """
        total = len(dataset)
        if sample_size and sample_size < total:
            if shuffle_data:
                dataset = dataset.shuffle(seed=random.randint(1,99999))
            dataset = dataset.select(range(sample_size))
        elif shuffle_data:
            dataset = dataset.shuffle(seed=random.randint(1,99999))
        print(f"Sampled {len(dataset)} examples (shuffle={shuffle_data}).")
        return dataset

    def read_csv(self) -> pd.DataFrame:
        """
        Reads the internal CSV into a DataFrame.
        """
        if not self.csv_file.exists():
            raise FileNotFoundError(f"CSV '{self.csv_file}' not found.")
        return pd.read_csv(self.csv_file)

    def write_csv(self, df: pd.DataFrame):
        """
        Writes a DataFrame back to the internal CSV.
        """
        df.to_csv(self.csv_file, index=False)

    def get_texts_for_analysis(self) -> list:
        """
        Returns list of texts from the CSV.
        """
        df = self.read_csv()
        return df['Text'].astype(str).tolist()

    def update_csv_with_results(self, results: dict):
        """
        Appends columns of results to the CSV.
        """
        df = self.read_csv()
        for col, vals in results.items():
            df[col] = vals
        self.write_csv(df)