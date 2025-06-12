# src/metadata_module.py
from __future__ import annotations
import json
from typing import List, Optional

# Re-use the path helper defined in dataset_handling.py
from src.dataset_handling import DataFiles


class MetadataCSVDataSaver:
    """
    Adds a one-shot “metadata” column to the run’s output CSV.
    """

    def __init__(self, dataset_handler):
        self.dataset_handler = dataset_handler

    # ------------------------------------------------------------------ #
    # 1) Centralised JSON helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _metadata_path():
        """Return Path to data/metadata.json (create file if missing)."""
        meta_p = DataFiles.path("data/metadata.json")

        if not meta_p.exists():
            meta_p.parent.mkdir(parents=True, exist_ok=True)
            meta_p.write_text("[]", encoding="utf-8")
        return meta_p

    def _load_metadata_json(self) -> Optional[List[dict]]:
        """Load the *list* stored in metadata.json, or None on error."""
        try:
            with self._metadata_path().open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as err:
            print(f"[MetadataCSVDataSaver] Failed to load metadata.json → {err}")
            return None

    # ------------------------------------------------------------------ #
    # 2) Public API – append a metadata column
    # ------------------------------------------------------------------ #

    def save_metadata(
        self,
        *,
        your_dataset_toggle: bool,
        our_datasets: str | None,
        sample_size: int,
        shuffle_data: bool,
        sentiment_context: str | None,
        LDA_analysis: bool,
        LDA_num_topics: int | None,
        version_name: str | None,
        model_type: str | None = None,
    ) -> None:
        """
        Assemble a list of human-readable run parameters and append
        them to the next free “Metadata …” column in your run CSV.
        """

        # ------------------------------------------------------------------
        # (A) Build the list of descriptive strings
        # ------------------------------------------------------------------
        entries: list[str] = []

        if sentiment_context:
            entries.append(f"sentiment context: {sentiment_context}")

        if your_dataset_toggle:
            entries.append("your_dataset: your_dataset")

        if our_datasets:
            try:
                ds_cfg = self.dataset_handler.load_dataset_config(our_datasets)
                ds_ref = ds_cfg.get("dataset reference", our_datasets)
                entries.append(f"our_datasets: {ds_ref}")
            except ValueError:
                print(f"[save_metadata] No config block for '{our_datasets}'")

        entries.append(f"background dataset size = {sample_size}")
        entries.append(
            "shuffled data: shuffled background data"
            if shuffle_data
            else "shuffled data: constant background data"
        )

        if LDA_analysis and LDA_num_topics is not None:
            entries.append(f"number of LDA topics: {LDA_num_topics}")

        # Grab extra fields (model_type, societally linear) from metadata.json
        meta_list = self._load_metadata_json()
        if meta_list and version_name:
            match = next((m for m in meta_list if m.get("version_name") == version_name), {})
            model_type_val = model_type or match.get("model_type", "unknown")
            entries.append(f"model_type: {model_type_val}")

            soc_lin_val = match.get("societally linear", "unknown")
            entries.append(f"societally linear: {soc_lin_val}")

        # ------------------------------------------------------------------
        # (B) Pad to CSV length & write column
        # ------------------------------------------------------------------
        df = self.dataset_handler.read_csv()

        col_name = "Metadata"
        idx = 1
        while col_name in df.columns:
            col_name = f"Metadata_{idx}"
            idx += 1

        padded = entries + [""] * (len(df) - len(entries))
        df[col_name] = padded
        self.dataset_handler.write_csv(df)
        print(f"[MetadataCSVDataSaver] Wrote '{col_name}' with {len(entries)} entries.")
