import os
import sys
import types
import contextlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub heavy dependencies not available in the test environment
torch = types.ModuleType("torch")
torch.cuda = types.SimpleNamespace(is_available=lambda: False)
@contextlib.contextmanager
def dummy_no_grad():
    yield
torch.no_grad = dummy_no_grad
sys.modules.setdefault("torch", torch)
sys.modules.setdefault("torch.nn", types.ModuleType("torch.nn"))
sys.modules.setdefault("torch.nn.functional", types.ModuleType("torch.nn.functional"))

sys.modules.setdefault("numpy", types.ModuleType("numpy"))

tqdm_mod = types.ModuleType("tqdm")
def dummy_tqdm(*args, **kwargs):
    class Dummy:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def update(self, n=1):
            pass
    return Dummy()
tqdm_mod.tqdm = dummy_tqdm
sys.modules.setdefault("tqdm", tqdm_mod)

alb_s1_mod = types.ModuleType("alb_s1")
class SentimentCSVDataSaver:
    def __init__(self, dataset_handler, sentiment_context=None):
        self.dataset_handler = dataset_handler
        self.sentiment_context = sentiment_context
        self.output_mode = getattr(dataset_handler, "num_labels", 1)
        self.class_labels = None
        self.continuous_label = None
alb_s1_mod.SentimentCSVDataSaver = SentimentCSVDataSaver
alb_s1_mod.SentimentBaseProcessor = type("SentimentBaseProcessor", (), {})
sys.modules.setdefault("alb_s1", alb_s1_mod)

# Minimal pandas replacement so dataset_handling imports succeed
pandas_mod = types.ModuleType("pandas")
class DataFrame:
    pass
pandas_mod.DataFrame = DataFrame
pandas_mod.read_csv = lambda *a, **k: None
pandas_mod.Series = list
sys.modules.setdefault("pandas", pandas_mod)

# Stub datasets module required by dataset_handling
datasets_mod = types.ModuleType("datasets")
datasets_mod.load_dataset = lambda *a, **k: None
class Dataset:
    pass
datasets_mod.Dataset = Dataset
sys.modules.setdefault("datasets", datasets_mod)


class FakeDataFrame:
    def __init__(self, data):
        self.data = {k: list(v) for k, v in data.items()}

    def __len__(self):
        if not self.data:
            return 0
        first_key = next(iter(self.data))
        return len(self.data[first_key])

    @property
    def columns(self):
        return list(self.data.keys())

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = list(value)

    def copy(self):
        return FakeDataFrame(self.data)


from unittest.mock import MagicMock
from monte_carlo_module import MonteCarloDropoutCSVDataSaver
from dataset_handling import DatasetHandler


def test_save_monte_carlo_results_creates_mean_and_std_columns():
    df = FakeDataFrame({"Text": ["a", "b", "c"]})
    handler = DatasetHandler()
    handler.num_labels = 1
    handler.read_csv = MagicMock(return_value=df)
    handler.write_csv = MagicMock()

    saver = MonteCarloDropoutCSVDataSaver(dataset_handler=handler)

    mc_results = [
        {"mc_mean": 0.1, "mc_std": 0.01},
        {"mc_mean": 0.2, "mc_std": 0.02},
        {"mc_mean": 0.3, "mc_std": 0.03},
    ]

    saver.save_monte_carlo_results(mc_results)

    assert handler.write_csv.called
    written_df = handler.write_csv.call_args[0][0]

    assert "MC Continuous Mean" in written_df.columns
    assert "MC Continuous Std" in written_df.columns
    assert len(written_df["MC Continuous Mean"]) == len(df)
    assert len(written_df["MC Continuous Std"]) == len(df)

