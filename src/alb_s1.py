import concurrent.futures
import matplotlib.pyplot as plt
import re
import emoji
import torch
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from typing import Tuple, Optional, Dict
from transformers import default_data_collator
import os
import datetime
import random
import numpy as np
from tqdm import tqdm
import matplotlib.patches as mpatches
from datasets import load_dataset, Dataset
from sklearn.metrics import (
    f1_score, accuracy_score, mean_squared_error,
    mean_absolute_error, r2_score
)
from transformers import (
    AlbertTokenizer,
    BertTokenizer,
    AlbertForSequenceClassification,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AlbertConfig,
    BertConfig,
    EarlyStoppingCallback
)
import json
from src.dataset_handling import DataFiles


class SentimentBaseProcessor:
    """Minimal base processor for tokenization and data loading."""
    def __init__(self, max_seq_length=128, model_size='albert-base'):
        self.max_seq_length = max_seq_length
        self.model_size = model_size
        if model_size == 'bert':
            model_name = 'bert-base-uncased'
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        elif model_size == 'albert-base':
            model_name = 'albert-base-v2'
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        elif model_size == 'xlarge':
            model_name = 'albert-xlarge-v2'
            self.tokenizer = AlbertTokenizer.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model_size '{model_size}'.")

        self.model = None
        self.output_mode = None
        self.num_labels = None
        self.dataset_config = None

    def clean_text(self, text, remove_mentions=True, remove_urls=True, segment_hashtags=True, replace_emojis=True):
        if remove_mentions:
            text = re.sub(r'@\w+', '', text)
        if remove_urls:
            text = re.sub(r'http\S+|www.\S+', '', text)
        if segment_hashtags:
            text = re.sub(
                r'#(\w+)',
                lambda x: ' '.join(re.findall(r'[A-Z][a-z]+|\w+', x.group(1))),
                text
            )
        if replace_emojis:
            text = emoji.demojize(text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_text(self, text):
        return self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_seq_length,
            return_tensors='pt'
        )

    def parallel_tokenization(self, texts):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            tokenized_texts = list(executor.map(self.tokenize_text, texts))
        return tokenized_texts

    def load_dataset_config(self, dataset_name):
        """
        Loads dataset-specific configurations from data/dataset_configs.json.
        """
        config_path = DataFiles.path("data/dataset_configs.json")
        with config_path.open('r', encoding='utf-8') as f:
            configs = json.load(f)
        if dataset_name not in configs:
            raise ValueError(f"Dataset configuration '{dataset_name}' not found.")
        self.dataset_config = configs[dataset_name]
        self.num_labels = self.dataset_config.get('num_labels')
        return self.dataset_config

    def initialize_model(self, sentiment_context=None):
        if self.model_size == 'bert':
            model_name = 'bert-base-uncased'
            config_class = BertConfig
            model_class = BertForSequenceClassification
        elif self.model_size == 'albert-base':
            model_name = 'albert-base-v2'
            config_class = AlbertConfig
            model_class = AlbertForSequenceClassification
        elif self.model_size == 'xlarge':
            model_name = 'albert-xlarge-v2'
            config_class = AlbertConfig
            model_class = AlbertForSequenceClassification
        else:
            raise ValueError("Unsupported model size.")

        if sentiment_context:
            try:
                meta_path = DataFiles.path("data/metadata.json")
                with meta_path.open('r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                meta = next((m for m in metadata_list if m['version_name'] == sentiment_context), None)
                if meta is None:
                    raise FileNotFoundError(f"No metadata found for '{sentiment_context}'.")

                weights_fp = meta['weights_filepath']
                dir_for_weights = os.path.dirname(weights_fp)
                self.num_labels = meta.get('output_mode')
                if self.num_labels is None:
                    raise ValueError("Metadata must contain 'output_mode'.")
                config = config_class.from_pretrained(model_name, num_labels=self.num_labels)
                self.model = model_class.from_pretrained(dir_for_weights, config=config)
                self.output_mode = meta['output_mode']
                print(f"Initialized model with fine-tuned context '{sentiment_context}'.")
            except Exception as e:
                raise RuntimeError(f"Failed to load fine-tuned weights for '{sentiment_context}': {e}")
        else:
            n_labels = self.num_labels if self.num_labels else 3
            config = config_class.from_pretrained(model_name, num_labels=n_labels)
            self.model = model_class.from_pretrained(model_name, config=config)
            self.output_mode = n_labels
            print(f"Initialized {model_name} with {n_labels} labels.")


class FAdam(Optimizer):
    """
    FAdam (Fisher Adam): a PyTorch implementation based on the paper:
    "FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher information"

    This class uses a diagonal approximation to the empirical Fisher information
    to adaptively scale gradients.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        betas: Tuple[float, float] = (0.9, 0.999),
        clip: float = 1.0,
        p: float = 0.5,
        eps: float = 1e-8,
        momentum_dtype: torch.dtype = torch.float32,
        fim_dtype: torch.dtype = torch.float32,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
            momentum_dtype=momentum_dtype,
            fim_dtype=fim_dtype,
            clip=clip,
            p=p,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            clip_val = group["clip"]
            pval = group["p"]
            momentum_dtype = group["momentum_dtype"]
            fim_dtype = group["fim_dtype"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("FAdam does not support sparse gradients")

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["momentum"] = torch.zeros_like(p, dtype=momentum_dtype)
                    state["fim"] = torch.ones_like(p, dtype=fim_dtype)

                state["step"] += 1
                step = state["step"]
                m = state["momentum"]
                fim = state["fim"]
                g = p.grad

                # (A) Beta2 bias correction
                curr_beta2 = beta2 * (1 - beta2 ** (step - 1)) / (1 - beta2 ** step)

                # (B) Update FIM
                fim.mul_(curr_beta2).add_(g * g, alpha=1 - curr_beta2)

                # (C) Adaptive epsilon
                rms_grad = torch.sqrt(torch.mean(g * g))
                curr_eps = eps * min(1, rms_grad)

                # (D) fim_base
                fim_base = fim.pow(pval) + curr_eps

                # (E) Natural gradient
                grad_nat = g / fim_base

                # (F) Clip
                rms_nat = torch.sqrt(torch.mean(grad_nat ** 2))
                divisor = max(1, rms_nat) / clip_val
                grad_nat = grad_nat / divisor

                # (G) Momentum
                m.mul_(beta1).add_(grad_nat, alpha=1 - beta1)

                # (H) Weight decay portion
                gw = p / fim_base
                rmsw = torch.sqrt(torch.mean(gw ** 2))
                divisor_w = max(1, rmsw) / clip_val
                gw = gw / divisor_w

                # (I) Combine momentum + weight decay => final step
                full_step = m + wd * gw

                # (J) Update
                p.sub_(lr * full_step)

        return loss


class Diamond_uncertainty:
    def __init__(self, model):
        self.model = model
        self.fisher_inverse = None  # Treated as the "variance" (inverse-FIM).
        self.final_classifier_bias_pre_dropout = None
        self.original_classifier_bias_values = None
        self.ordered_param_names = [n for n, _ in model.named_parameters()]

    def load_fisher_from_fadam(self, optimizer, invert=True):
        """
        Grab each param's 'fim' from the FAdam optimizer, optionally invert it,
        and store in self.fisher_inverse.
        """
        fishers = {}
        for group in optimizer.param_groups:
            for p in group["params"]:
                state_dict = optimizer.state.get(p, {})
                if "fim" in state_dict:
                    # find param name
                    name_ = None
                    for (n, param_obj) in self.model.named_parameters():
                        if param_obj is p:
                            name_ = n
                            break
                    if name_ is not None:
                        fishers[name_] = state_dict["fim"].clone().detach()

        if not fishers:
            self.fisher_inverse = None
            return

        finv = {}
        eps_ = 1e-12
        for k, fim_mat in fishers.items():
            if invert:
                inv_ = 1.0 / (fim_mat + eps_)
                inv_ = torch.where(
                    torch.isnan(inv_) | torch.isinf(inv_),
                    torch.tensor(1e8, device=inv_.device),
                    inv_
                )
                finv[k] = inv_
            else:
                finv[k] = fim_mat

        self.fisher_inverse = finv if finv else None

    def all_layers_mean_Diag_FIM(
        self,
        fine_tune_version_name="",
        title_fontsize=14,
        axis_label_fontsize=8,
        axis_number_fontsize=8
    ):
        """
        Visualize average variance (inverse-FIM) per layer, linear & log scale.

        :param fine_tune_version_name: subdirectory in 'variance_plots'.
        :param title_fontsize: Font size for the plot title.
        :param axis_label_fontsize: Font size for the X-axis label text.
        :param axis_number_fontsize: Font size for the numeric axis ticks (the X-axis numbers).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if not self.fisher_inverse:
            print("[diamond_uncertainty] No fisher_inverse found.")
            return

        plt.rcParams["font.family"] = "Times New Roman"
        plot_dir = os.path.join("variance_plots", fine_tune_version_name, "all_layers")
        os.makedirs(plot_dir, exist_ok=True)

        weight_names, weight_means = [], []
        bias_names, bias_means = [], []

        # Collect the average variance for .weight and .bias
        for name in self.ordered_param_names:
            if name in self.fisher_inverse:
                param_var = self.fisher_inverse[name]
                mean_val = param_var.mean().item()
                if ".weight" in name:
                    weight_names.append(name)
                    weight_means.append(mean_val)
                elif ".bias" in name:
                    bias_names.append(name)
                    bias_means.append(mean_val)

        # -------------------------------------------------------
        # (A) Weights (linear scale)
        # -------------------------------------------------------
        plt.figure(figsize=(8, max(6, 0.4 * len(weight_names))))
        y_pos = np.arange(len(weight_names))

        plt.barh(y_pos, weight_means, color='blue', edgecolor='black')
        plt.yticks(y_pos, weight_names, fontsize=axis_number_fontsize)
        plt.xticks(fontsize=axis_number_fontsize)

        plt.xlabel("Average Variance", fontsize=axis_label_fontsize)
        plt.title("Average Weight Variance Per Transformer Layer", fontsize=title_fontsize)

        # Flip so the first layer is at the top
        plt.gca().invert_yaxis()
        plt.tight_layout()

        outpath_w = os.path.join(plot_dir, "all_layers_weights_fisher.png")
        plt.savefig(outpath_w, dpi=300, bbox_inches='tight')
        plt.close()

        # -------------------------------------------------------
        # (B) Weights (log scale)
        # -------------------------------------------------------
        safe_means_w = [max(1e-15, x) for x in weight_means]
        plt.figure(figsize=(8, max(6, 0.4 * len(weight_names))))
        y_pos = np.arange(len(weight_names))

        plt.barh(y_pos, safe_means_w, color='blue', edgecolor='black')
        plt.xscale('log')

        plt.yticks(y_pos, weight_names, fontsize=axis_number_fontsize)
        plt.xticks(fontsize=axis_number_fontsize)

        plt.xlabel("Average Variance (Log Scale)", fontsize=axis_label_fontsize)
        plt.title("Average Weight Variance Per Transformer Layer", fontsize=title_fontsize)

        plt.gca().invert_yaxis()
        plt.tight_layout()

        outpath_w_log = os.path.join(plot_dir, "all_layers_weights_fisher_log.png")
        plt.savefig(outpath_w_log, dpi=300, bbox_inches='tight')
        plt.close()

        # -------------------------------------------------------
        # (C) Bias (linear scale)
        # -------------------------------------------------------
        plt.figure(figsize=(8, max(6, 0.4 * len(bias_names))))
        y_pos = np.arange(len(bias_names))

        plt.barh(y_pos, bias_means, color='orange', edgecolor='black')
        plt.yticks(y_pos, bias_names, fontsize=axis_number_fontsize)
        plt.xticks(fontsize=axis_number_fontsize)

        plt.xlabel("Average Variance", fontsize=axis_label_fontsize)
        plt.title("Average Bias Variance Per Transformer Layer", fontsize=title_fontsize)

        plt.gca().invert_yaxis()
        plt.tight_layout()

        outpath_b = os.path.join(plot_dir, "all_layers_biases_fisher.png")
        plt.savefig(outpath_b, dpi=300, bbox_inches='tight')
        plt.close()

        # -------------------------------------------------------
        # (D) Bias (log scale)
        # -------------------------------------------------------
        safe_means_b = [max(1e-15, x) for x in bias_means]
        plt.figure(figsize=(8, max(6, 0.4 * len(bias_names))))
        y_pos = np.arange(len(bias_names))

        plt.barh(y_pos, safe_means_b, color='orange', edgecolor='black')
        plt.xscale('log')

        plt.yticks(y_pos, bias_names, fontsize=axis_number_fontsize)
        plt.xticks(fontsize=axis_number_fontsize)

        plt.xlabel("Average Variance (Log Scale)", fontsize=axis_label_fontsize)
        plt.title("Average Bias Variance Per Transformer Layer", fontsize=title_fontsize)

        plt.gca().invert_yaxis()
        plt.tight_layout()

        outpath_b_log = os.path.join(plot_dir, "all_layers_biases_fisher_log.png")
        plt.savefig(outpath_b_log, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_histograms_matplotlib(
        self,
        fine_tune_version_name="",
        classif_weight_key="classifier.weight",
        classif_bias_key="classifier.bias",
        num_labels=None,
        hidden_size=None,
        class_labels=None
    ):
        """
        Creates linear/log histograms for bias and weight variance.

        Part C updates:
        - For the per-label classifier weight histograms, each label gets
        a distinct color from the same color cycle used in the variance
        bar chart (matching index).
        - No legend is created for these histograms; the plot title states
        which label it corresponds to.
        - We keep everything else exactly the same (combined hist, bias chart, etc.).
        """
        import os
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        if not self.fisher_inverse:
            print("[diamond_uncertainty] No fisher_inverse to generate histograms.")
            return

        plt.rcParams["font.family"] = "Times New Roman"
        run_dir = os.path.join("variance_plots", fine_tune_version_name)
        non_log_dir = os.path.join(run_dir, "non_log_histograms")
        log_dir = os.path.join(run_dir, "log_histograms")
        os.makedirs(non_log_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        def save_fig(folder, fname):
            outpath = os.path.join(folder, f"{fname}.png")
            plt.savefig(outpath, dpi=300, bbox_inches='tight')
            plt.close()

        # ---- Classifier Bias (Linear) ----
        if classif_bias_key in self.fisher_inverse:
            bias_vals = self.fisher_inverse[classif_bias_key].detach().cpu().numpy()
            C = bias_vals.size

            plt.figure(figsize=(8, 6))
            plt.title("Classifier Bias Variance", fontsize=16)
            plt.xlabel("Bias Index (Per Class)", fontsize=14)
            plt.ylabel("Variance Value", fontsize=14)
            plt.grid(alpha=0.7, linestyle='--')

            # We keep the original color-cycle + legend approach for bias
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            patches_list = []
            for i in range(C):
                bar_color = color_cycle[i % len(color_cycle)]
                plt.bar(i, bias_vals[i], color=bar_color, edgecolor="black")
                if class_labels and i < len(class_labels):
                    label_str = class_labels[i]
                else:
                    label_str = f"Bias_{i}"
                patch = mpatches.Patch(color=bar_color, label=label_str)
                patches_list.append(patch)

            # Filter out duplicates so we only list each color once
            unique_patches = []
            used_colors = set()
            for patch in patches_list:
                c_ = patch.get_facecolor()
                if c_ not in used_colors:
                    unique_patches.append(patch)
                    used_colors.add(c_)
            plt.legend(handles=unique_patches, fontsize=10)
            plt.tight_layout()
            save_fig(non_log_dir, "bias_var_linear_barchart")

        # ---- Classifier Weight (All) ----
        if classif_weight_key in self.fisher_inverse:
            w_np = self.fisher_inverse[classif_weight_key].detach().cpu().numpy()
            if w_np.ndim == 2 and num_labels is not None and hidden_size is not None:
                all_vals = w_np.flatten()
                # Linear
                plt.figure(figsize=(8, 6))
                plt.title("Classifier Weight Variance Combined", fontsize=16)
                plt.xlabel("Variance Value", fontsize=14)
                plt.ylabel("Frequency", fontsize=14)
                plt.grid(alpha=0.7, linestyle='--')
                plt.hist(all_vals, bins=50, color="salmon", edgecolor="black")
                plt.tight_layout()
                save_fig(non_log_dir, "weight_var_all_linear")

                # Log
                max_val = all_vals.max() if all_vals.size > 0 else 1
                if max_val < 1:
                    max_val = 1
                max_power = int(math.ceil(math.log10(max_val))) if max_val > 0 else 0
                log_bins = [0] + [10 ** p for p in range(max_power + 1)]
                plt.figure(figsize=(8, 6))
                plt.title("Classifier Weight Variance Combined", fontsize=16)
                plt.xlabel("Variance Value (Log Scale)", fontsize=14)
                plt.ylabel("Frequency", fontsize=14)
                plt.grid(alpha=0.7, linestyle='--')
                plt.hist(all_vals, bins=log_bins, color="salmon", edgecolor="black")
                plt.xscale('log')
                plt.tight_layout()
                save_fig(log_dir, "weight_var_all_log_axis")

                # ----------------------
                # Now the Per-label histograms
                # We apply color-coding from the color cycle, same as for the bias bar chart.
                # No legend is needed, the title says which label it is.
                # ----------------------
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for i in range(num_labels):
                    row_vals = w_np[i]
                    if class_labels and i < len(class_labels):
                        label_str = class_labels[i]
                    else:
                        label_str = f"Label_{i}"

                    # Linear scale
                    plt.figure(figsize=(8, 6))
                    plt.title(f"Weight Variance for {label_str}", fontsize=16)
                    plt.xlabel("Variance Value", fontsize=14)
                    plt.ylabel("Frequency", fontsize=14)
                    plt.grid(alpha=0.7, linestyle='--')

                    color_i = color_cycle[i % len(color_cycle)]
                    plt.hist(row_vals, bins=50, color=color_i, edgecolor="black")

                    plt.tight_layout()
                    save_fig(non_log_dir, f"weight_var_label_{i}_linear")

                    # Log scale
                    rmax_val = row_vals.max() if row_vals.size > 0 else 1
                    if rmax_val < 1:
                        rmax_val = 1
                    rmax_power = int(math.ceil(math.log10(rmax_val))) if rmax_val > 0 else 0
                    row_log_bins = [0] + [10 ** p for p in range(rmax_power + 1)]

                    plt.figure(figsize=(8, 6))
                    plt.title(f"Weight Variance for {label_str}", fontsize=16)
                    plt.xlabel("Variance Value (Log-Scale)", fontsize=14)
                    plt.ylabel("Frequency", fontsize=14)
                    plt.grid(alpha=0.7, linestyle='--')

                    plt.hist(row_vals, bins=row_log_bins, color=color_i, edgecolor="black")
                    plt.xscale('log')

                    plt.tight_layout()
                    save_fig(log_dir, f"weight_var_label_{i}_log_axis")

    def apply_smallest_n_proportion_filter(
        self,
        variance: Dict[str, torch.Tensor],
        n_proportion=1.0,
        base_value=0.0,
        layer_filter='classifier.weight'
    ):
        """
        Zero out all but the smallest `n_proportion` fraction of values in 'variance'
        for the given layer(s).
        """
        import torch

        if isinstance(n_proportion, str) and n_proportion.lower() == 'auto':
            return variance
        if n_proportion >= 1.0:
            return variance

        layer_keys = [k for k in variance if layer_filter in k]
        if not layer_keys:
            return variance

        all_var = []
        shapes = {}
        for k in layer_keys:
            shapes[k] = variance[k].shape
            all_var.append(variance[k].view(-1))

        all_var_cat = torch.cat(all_var, dim=0)
        total_params = all_var_cat.numel()
        keep_count = int(total_params * n_proportion)
        if keep_count < 1:
            keep_count = 1

        sorted_vals, sorted_idxs = torch.sort(all_var_cat)
        keep_idxs = sorted_idxs[:keep_count]
        mask = torch.zeros_like(all_var_cat, dtype=torch.bool)
        mask[keep_idxs] = True

        filtered_var = torch.where(
            mask,
            all_var_cat,
            torch.tensor(base_value, device=all_var_cat.device)
        )

        offset = 0
        new_variance = {}
        for k in layer_keys:
            param_size = shapes[k].numel()
            new_variance[k] = filtered_var[offset : offset + param_size].view(shapes[k])
            offset += param_size

        for k in layer_keys:
            variance[k] = new_variance[k]
        return variance

    def plot_f1_dropout_4plots(
        self,
        trainer,
        test_dataset,
        # Single float OR tuple => (step_a, step_b, pivot)
        f1_dropout_step=0.05,
        n_proportion=1.0,
        plot_title="F1 vs. dropout: Classifier Weights & Word Embeddings",
        save_folder="variance_plots",
        fine_tune_version_name="",
        class_labels=None,

        # --- NEW TOGGLES ---
        title_fontsize=22,          # Title ("F1 Dropout for ...")
        axis_label_fontsize=22,     # Axis labels ("F1 Score", "n_proportion")
        axis_number_fontsize=16,    # Numeric labels on major ticks (0, 0.2, 0.4, etc.)
        minor_tick_size=4,          # Length of the unlabeled minor tick marks
        marker_size=1,              # Dot/marker size for points on the plot
        legend_fontsize=16,         # Legend text size
        y_axis_upper=1.2            # Extend Y-axis to 1.2 (no label at 1.2)
    ):
        """
        Creates dropout plots for:
        1) Classifier Weights
        2) Word Embeddings
        3) Classifier Bias Removal
        4) Attention Bias Removal (if any)

        Changes:
        - 'f1_dropout_step' can be float or (step_a, step_b, pivot).
        - Replaces "(n_prop=..., f1=...)" with "(dropout proportion=..., f1=...)" in legend.
        - Places a y-axis tick at the chosen F1.
        - Stores that chosen F1 in self.dropout_chosen_f1, so we can reference it later (e.g. in plot_starting_weight_f1_gaussian).
        """

        import os
        import numpy as np
        import torch
        import matplotlib.pyplot as plt
        from sklearn.metrics import f1_score

        device = next(self.model.parameters()).device

        # Create a subfolder for these plots
        f1_dir = os.path.join(save_folder, fine_tune_version_name, "f1_output_plots")
        os.makedirs(f1_dir, exist_ok=True)

        if not self.fisher_inverse:
            print("[plot_f1_dropout_4plots] No stored variance found.")
            return

        original_var = {k: v.clone() for k, v in self.fisher_inverse.items()}

        # ---------------------------------------------------------------
        # 1) Build n_values from f1_dropout_step (single float or tuple)
        # ---------------------------------------------------------------
        def build_n_values(step_param):
            """
            If 'step_param' is float => old single-step logic.
            If 'step_param' is (step_a, step_b, pivot) => multi-step logic.
            E.g. (0.1, 0.01, 0.3): decrement by 0.1 until ~0.3, then by 0.01 down to 0.
            """
            if isinstance(step_param, (tuple, list)) and len(step_param) == 3:
                step_a, step_b, pivot = step_param
                vals = []
                curr = 1.0
                while curr >= pivot:
                    vals.append(round(curr, 6))
                    curr -= step_a
                    if curr < pivot:
                        curr = pivot
                        vals.append(round(curr, 6))
                        break
                curr = pivot - step_b
                while curr >= 0.0:
                    vals.append(round(curr, 6))
                    curr -= step_b
                if vals[-1] != 0.0:
                    vals.append(0.0)
                return vals
            else:
                # Single-float approach
                step_size = float(step_param)
                nvals = []
                curr_n = 1.0
                while curr_n >= 0.0:
                    nvals.append(round(curr_n, 6))
                    curr_n -= step_size
                    if curr_n < 0.0:
                        nvals.append(0.0)
                        break
                return nvals

        # ---------------------------------------------------------------
        # 2) Helper: run dropout for each n, measure F1, plot
        # ---------------------------------------------------------------
        def run_dropout_experiment(layer_filter, out_png_name, custom_title, show_red_lines=True):
            """
            - We build n_values,
            - For each n => zero out largest fraction in variance, measure F1,
            - Plot (n vs. F1) with the new styling + legend changes.
            """
            # Backup original params
            param_backup = {}
            for name, param in self.model.named_parameters():
                if layer_filter in name:
                    param_backup[name] = param.detach().cpu().clone()

            if not param_backup:
                print(f"[plot_f1_dropout_4plots] No params for '{layer_filter}'. Skipping.")
                return None, None

            n_values = build_n_values(f1_dropout_step)
            f1_scores = []

            # Evaluate with different n_proportions
            for current_n in n_values:
                var_copy = {k: v.clone() for k, v in original_var.items()}
                self.apply_smallest_n_proportion_filter(
                    variance=var_copy,
                    n_proportion=current_n,
                    base_value=0.0,
                    layer_filter=layer_filter
                )

                # Mask the model parameters
                for pname, pval in self.model.named_parameters():
                    if layer_filter in pname:
                        mask = (var_copy[pname].to(device) != 0.0)
                        pval.data = pval.data * mask

                # Measure F1
                test_preds = trainer.predict(test_dataset)
                preds = test_preds.predictions
                labels = np.array(test_dataset['labels'])
                predicted_classes = np.argmax(preds, axis=-1)
                current_f1 = f1_score(labels, predicted_classes, average='weighted')
                f1_scores.append(current_f1)

                # Restore
                for pname, pval in self.model.named_parameters():
                    if pname in param_backup:
                        pval.data = param_backup[pname].to(device)

            # Plot if at least 2 points
            if len(n_values) > 1:
                plt.figure(figsize=(8, 6))
                plt.plot(
                    n_values, f1_scores,
                    marker='o', markersize=marker_size,
                    linestyle='-', color='b'
                )
                plt.title(custom_title, fontsize=title_fontsize)
                plt.xlabel("Parameter Fraction Retained", fontsize=axis_label_fontsize)
                plt.ylabel("F1 Score", fontsize=axis_label_fontsize)
                plt.grid(True, linestyle='--', alpha=0.7)

                ax = plt.gca()
                ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                fontsize=axis_number_fontsize)
                ax.set_xticks(n_values, minor=True)
                ax.set_xticklabels([], minor=True)

                ax.set_ylim(0.0, y_axis_upper)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"],
                                fontsize=axis_number_fontsize)
                ax.tick_params(axis='x', which='minor', length=minor_tick_size)

                # If user wants the red lines at 'n_proportion'
                if show_red_lines and 0.0 <= n_proportion < 1.0:
                    idx_closest = min(range(len(n_values)), key=lambda i: abs(n_values[i] - n_proportion))
                    chosen_f1 = f1_scores[idx_closest]

                    # (A) Vertical/horizontal dashed lines
                    plt.plot([n_proportion, n_proportion], [0.0, chosen_f1],
                            color='red', linestyle='--',
                            label=f"(dropout proportion={n_proportion:.2f}, f1={chosen_f1:.3f})")
                    plt.plot([n_proportion, 0.0], [chosen_f1, chosen_f1],
                            color='red', linestyle='--')

                    # (B) Add a y-axis tick for chosen_f1
                    existing_yticks = list(ax.get_yticks())
                    if not any(abs(t - chosen_f1) < 1e-9 for t in existing_yticks):
                        new_yticks = sorted(existing_yticks + [chosen_f1])
                        ax.set_yticks(new_yticks)
                        ax.set_yticklabels([f"{v:.2f}" for v in new_yticks],
                                        fontsize=axis_number_fontsize)

                    # (C) Store in self.dropout_chosen_f1 so we can reuse it
                    self.dropout_chosen_f1 = chosen_f1

                    plt.legend(fontsize=legend_fontsize, loc='upper right')

                plt.tight_layout()
                outpath = os.path.join(f1_dir, out_png_name)
                plt.savefig(outpath, dpi=300, bbox_inches='tight')
                plt.close()

            return n_values, f1_scores

        # -------------------------------
        # (1) F1 Dropout: Classifier Weights
        # -------------------------------
        run_dropout_experiment(
            layer_filter="classifier.weight",
            out_png_name="f1_dropout_classifier_weights.png",
            custom_title="F1 Dropout for Classifier Weights",
            show_red_lines=True
        )

        # -------------------------------
        # (2) F1 Dropout: Word Embeddings
        # -------------------------------
        run_dropout_experiment(
            layer_filter="embeddings.word_embeddings.weight",
            out_png_name="f1_dropout_word_embedding_weights.png",
            custom_title="F1 Dropout for Word Embedding Weights",
            show_red_lines=False
        )

        # -------------------------------
        # (3) Classifier Bias Removal
        # -------------------------------
        bias_param_name = None
        for name, param in self.model.named_parameters():
            if "classifier.bias" in name:
                bias_param_name = name
                break

        if bias_param_name is not None:
            bias_param = dict(self.model.named_parameters())[bias_param_name]
            if bias_param.dim() == 1:
                num_biases = bias_param.shape[0]
                original_bias = bias_param.detach().cpu().clone()
                self.original_classifier_bias_values = original_bias.tolist()
                f1_per_bias = []

                print(f"[Bias removal] Removing each of {num_biases} classifier biases one-by-one...")

                for i in range(num_biases):
                    tmp = bias_param.data.cpu().clone()
                    tmp[i] = 0.0
                    bias_param.data = tmp.to(device)

                    test_preds = trainer.predict(test_dataset)
                    preds = test_preds.predictions
                    labels = np.array(test_dataset['labels'])
                    predicted_classes = np.argmax(preds, axis=-1)
                    current_f1 = f1_score(labels, predicted_classes, average='weighted')
                    f1_per_bias.append(current_f1)

                    # restore
                    bias_param.data = original_bias.to(device)

                x_vals = list(range(num_biases))
                plt.figure(figsize=(8, 6))
                plt.plot(x_vals, f1_per_bias, color='gray', linestyle='-', zorder=1)
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for i in range(num_biases):
                    color_i = color_cycle[i % len(color_cycle)]
                    lbl_str = (
                        f"{class_labels[i]} bias" if (class_labels and i < len(class_labels))
                        else f"Bias_{i}"
                    )
                    plt.scatter(
                        x_vals[i], f1_per_bias[i],
                        color=color_i, label=lbl_str, s=80, zorder=2
                    )

                plt.title("Classifier Bias Removal", fontsize=14)
                plt.xlabel("Index of Bias Removed", fontsize=12)
                plt.ylabel("F1 Score", fontsize=12)  # removing "(weighted)"
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim([0, 1])
                plt.xlim([0, max(num_biases - 1, 1)])
                plt.xticks(x_vals, [str(x) for x in x_vals])
                plt.legend(fontsize=10)
                plt.tight_layout()
                outpath_bias = os.path.join(f1_dir, "f1_bias_removal_plot.png")
                plt.savefig(outpath_bias, dpi=300, bbox_inches='tight')
                plt.close()

        # -------------------------------
        # (4) Attention Bias Removal
        # -------------------------------
        att_bias_param_name = None
        for name, param in self.model.named_parameters():
            if "groups0.albert_layers0.attention.bias" in name:
                att_bias_param_name = name
                break

        if att_bias_param_name is not None:
            att_bias_param = dict(self.model.named_parameters())[att_bias_param_name]
            if att_bias_param.dim() == 1:
                num_att_biases = att_bias_param.shape[0]
                original_att_bias = att_bias_param.detach().cpu().clone()
                f1_per_att_bias = []

                print(f"[Bias removal] Removing each of {num_att_biases} attention biases one-by-one...")

                for i in range(num_att_biases):
                    tmp = att_bias_param.data.cpu().clone()
                    tmp[i] = 0.0
                    att_bias_param.data = tmp.to(device)

                    test_preds = trainer.predict(test_dataset)
                    preds = test_preds.predictions
                    labels = np.array(test_dataset['labels'])
                    predicted_classes = np.argmax(preds, axis=-1)
                    current_f1 = f1_score(labels, predicted_classes, average='weighted')
                    f1_per_att_bias.append(current_f1)

                    # restore
                    att_bias_param.data = original_att_bias.to(device)

                x_vals_att = list(range(num_att_biases))
                plt.figure(figsize=(8, 6))
                plt.plot(x_vals_att, f1_per_att_bias, color='gray', linestyle='-', zorder=1)
                color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

                for i in range(num_att_biases):
                    color_i = color_cycle[i % len(color_cycle)]
                    lbl_str = f"AttnBias_{i}"
                    plt.scatter(
                        x_vals_att[i], f1_per_att_bias[i],
                        color=color_i, label=lbl_str, s=80, zorder=2
                    )

                plt.title("Attention Bias Removal", fontsize=14)
                plt.xlabel("Index of Bias Removed", fontsize=12)
                plt.ylabel("F1 Score", fontsize=12)  # removing "(weighted)"
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim([0, 1])
                plt.xlim([0, max(num_att_biases - 1, 1)])
                plt.xticks(x_vals_att, [str(x) for x in x_vals_att])
                plt.legend(fontsize=10)
                plt.tight_layout()
                outpath_att_bias = os.path.join(f1_dir, "f1_attention_bias_removal_plot.png")
                plt.savefig(outpath_att_bias, dpi=300, bbox_inches='tight')
                plt.close()

        print("Done: generated 4 total plots for dropout + bias removal.")

    def plot_starting_weight_f1_gaussian(
        self,
        f1_mean,
        f1_std,
        chosen_f1,
        n_proportion,
        fine_tune_version_name="",
        sentiment_context="base-weights",
        marker_size=12
    ):
        """
        Plots a Gaussian curve for the 'starting weights' F1 distribution.
        """
        import math
        import numpy as np

        if f1_std < 1e-12:
            f1_std = 1e-12

        left_limit = f1_mean - 4.0 * f1_std
        right_limit = f1_mean + 4.0 * f1_std
        min_val = min(0.0, left_limit)
        max_val = max(1.0, right_limit)
        x_vals = np.linspace(min_val, max_val, 400)

        gauss = (1.0 / (np.sqrt(2.0 * math.pi) * f1_std)) * \
                np.exp(-0.5 * ((x_vals - f1_mean) / f1_std) ** 2)

        calc_val = (chosen_f1 - f1_mean) / f1_std

        plt.rcParams["font.family"] = "Times New Roman"
        plt.figure(figsize=(8, 6))
        plt.plot(x_vals, gauss, color='blue', label="(solid) F1 dist start weights")
        plt.title(f" {sentiment_context} F1 Distribution Pre-Fine-Tuning", fontsize=14)
        plt.xlabel("F1 Score", fontsize=12)
        plt.ylabel("Normalisation Density", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        mu_peak_val = (1.0 / (np.sqrt(2.0 * math.pi) * f1_std)) * \
                      np.exp(-0.5 * ((f1_mean - f1_mean) / f1_std)**2)
        plt.plot([f1_mean, f1_mean], [mu_peak_val, 0.0],
                 color='black', linestyle='--',
                 label=f"(dashed) μ={f1_mean:.3f}, σ={f1_std:.3f}")

        label_str = f"(n_prop={n_proportion:.2f}, f1={chosen_f1:.3f})\nf1= μ + {calc_val:.2f}σ"
        f1_peak_val = (1.0 / (np.sqrt(2.0 * math.pi) * f1_std)) * \
                      np.exp(-0.5 * ((chosen_f1 - f1_mean) / f1_std)**2)
        plt.plot([chosen_f1, chosen_f1], [f1_peak_val, 0.0],
                 color='red', linestyle='--',
                 label=label_str)
        plt.plot([chosen_f1], [f1_peak_val],
                 marker='x', color='red',
                 markersize=marker_size, linestyle='None')

        current_x_ticks = list(plt.xticks()[0])
        if not any(abs(t - f1_mean) < 1e-7 for t in current_x_ticks):
            current_x_ticks.append(f1_mean)
        if not any(abs(t - chosen_f1) < 1e-7 for t in current_x_ticks):
            current_x_ticks.append(chosen_f1)
        current_x_ticks = sorted(current_x_ticks)
        plt.xticks(current_x_ticks, [f"{tick:.2f}" for tick in current_x_ticks])
        plt.legend()
        plt.tight_layout()

        out_dir = os.path.join("variance_plots", fine_tune_version_name, "f1_output_plots")
        os.makedirs(out_dir, exist_ok=True)
        outpath = os.path.join(out_dir, "f1_starting_weight_gaussian.png")
        plt.savefig(outpath, dpi=300, bbox_inches='tight')
        plt.close()

    def compute_softmax_uncertainty_for_samples(
        self,
        logits_probs: torch.Tensor,
        hidden_mean: torch.Tensor,
        variance: dict,
        consider_bias=True
    ) -> torch.Tensor:
        """
        Example formula for each class i:
          var(p_i) ~ p_i^2 * [ ( Σ_k x_k^2 * var_{w_{i,k}} + var_{b_i} )*(1 - p_i)^2
                               + Σ_{j != i} p_j^2 ( Σ_k x_k^2*var_{w_{j,k}} + var_{b_j} ) ]
        """
        device = logits_probs.device
        w_var = variance.get("classifier.weight", None)
        b_var = variance.get("classifier.bias", None)

        if w_var is None or b_var is None:
            return torch.zeros_like(logits_probs)

        w_var = w_var.to(device)
        b_var = b_var.to(device)
        hidden_mean = hidden_mean.to(device)

        if not consider_bias:
            b_var = torch.zeros_like(b_var)

        batch_size, num_classes = logits_probs.shape
        out_unc = torch.zeros(batch_size, num_classes, device=device)

        for i in range(batch_size):
            p_ = logits_probs[i]
            x_sq = hidden_mean[i].pow(2)

            z_vars = []
            for j in range(num_classes):
                z_j = (x_sq * w_var[j]).sum() + b_var[j]
                z_vars.append(z_j)

            for i_cls in range(num_classes):
                pi_ = p_[i_cls]
                pi_sq_ = pi_ * pi_
                zi_ = z_vars[i_cls]
                part_self = zi_ * (1.0 - pi_).pow(2)
                part_cross = 0.0
                for j_cls in range(num_classes):
                    if j_cls == i_cls:
                        continue
                    pj_sq_ = p_[j_cls] * p_[j_cls]
                    part_cross += pj_sq_ * z_vars[j_cls]
                var_pi = pi_sq_ * (part_self + part_cross)
                out_unc[i, i_cls] = var_pi

        return out_unc

    def plot_uncertainty_for_first_sample(
        self,
        prob_vec,
        var_vec,
        class_labels=None,
        fine_tune_version_name="",
        fisher_uncertainty_plot_width="auto"
    ):
        """
        Creates a Gaussian-like plot for the first sample's predicted probabilities
        (with variance).

        Now saves fisher_uncertainty.png to 'example_case' subfolder.
        The legend is positioned in the top right.
        """
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        import os

        # 1) Instead of "f1_output_plots", use "example_case"
        out_dir = os.path.join("variance_plots", fine_tune_version_name, "example_case")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "fisher_uncertainty.png")

        if class_labels is None:
            class_labels = [f"Class_{i}" for i in range(len(prob_vec))]

        means = np.array(prob_vec, dtype=float)
        vars_ = np.array(var_vec, dtype=float)
        stds = np.sqrt(np.maximum(vars_, 1e-15))

        if isinstance(fisher_uncertainty_plot_width, (int, float)):
            w = float(fisher_uncertainty_plot_width)
            x_min, x_max = -w, w
        else:
            lower = np.min(means - 2.0 * stds)
            upper = np.max(means + 2.0 * stds)
            if lower >= upper:
                lower, upper = -0.5, 1.5
            margin = 0.1 * (upper - lower)
            x_min = lower - margin
            x_max = upper + margin

        x = np.linspace(x_min, x_max, 1000)

        plt.figure(figsize=(12, 8))
        plt.title("Sample Case Fisher-Derived Uncertainty", fontsize=20)
        plt.xlabel("Predicted Probability", fontsize=18)
        plt.ylabel(" Probability Density", fontsize=18)
        plt.grid(True, linestyle='--', alpha=0.7)

        for idx, (mean, var) in enumerate(zip(prob_vec, var_vec)):
            var = max(var, 1e-15)
            std = np.sqrt(var)
            y = (1.0 / (np.sqrt(2 * math.pi) * std)) * \
                np.exp(-0.5 * ((x - mean) / std) ** 2)
            mean_str = f"{mean:.3f}"
            var_str = f"{var:.6f}"
            label_name = f"{class_labels[idx]}: Mean={mean_str}, Var={var_str}"
            line = plt.plot(x, y, label=label_name)
            color = line[0].get_color()
            peak_y = np.max(y)
            plt.plot([mean, mean], [0, peak_y], linestyle='--', color=color)

        # Place legend at the top right
        plt.legend(fontsize=14, loc="upper right")

        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_finetuning_info_table(self, metadata, fine_tune_version_name=""):
        """
        Generates two tables:
        1) "Fine Tuning Information" (Table 1)
        2) "sampled textual data / bias info" (Table 2)

        B) Adjustments:
        - Removed the row for 'final_classifier_bias_pre_dropout' from Table 1.
        - Ensure 'lr', 'weight_decay', 'betas', 'clip', 'eps' appear in Table 1
            with renamed keys: "learning rate", "weight decay", "betas", "clip", "epsilon".
        """
        import textwrap
        import matplotlib.pyplot as plt
        from matplotlib.table import Table
        import os

        out_dir = os.path.join("variance_plots", fine_tune_version_name, "example_case")
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, "fine_tuning_info.png")

        plt.rcParams["font.family"] = "Times New Roman"

        # ------------------------------------------------------------------
        # 1) Renaming certain hyperparam keys for Table 1 display
        # ------------------------------------------------------------------
        rename_map = {
            "lr": "learning rate",
            "weight_decay": "weight decay",
            "betas": "betas",
            "clip": "clip",
            "eps": "epsilon"
        }
        for old_k, new_k in rename_map.items():
            if old_k in metadata:
                metadata[new_k] = metadata[old_k]
                del metadata[old_k]

        # ------------------------------------------------------------------
        # 2) Exclude keys we do NOT want in Table 1
        # ------------------------------------------------------------------
        excluded_keys = {
            "version_name",
            "weights_filepath",
            "output_mode",
            "class_labels",
            "first_sample_class_names",
            "first_sample_probs",
            "first_sample_vars",
            "sampled_text_data",
            "original_classifier_bias_values",
            # The following line specifically removes final_classifier_bias_pre_dropout from Table 1
            "final_classifier_bias_pre_dropout"
        }

        # Build Table 1 data
        table_items = []
        for k, v in metadata.items():
            if k not in excluded_keys:
                table_items.append((str(k), str(v)))

        fig_height = max(2, len(table_items) * 0.8)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.set_title("Fine Tuning Information", fontsize=20)
        ax.axis('off')

        tbl = Table(ax, bbox=[0, 0, 1, 1])
        row_height = 1.0 / (len(table_items) + 1)
        col_widths = [0.4, 0.6]
        font_size = 16

        for i, (key, val) in enumerate(table_items):
            cell_left = tbl.add_cell(
                row=i, col=0,
                width=col_widths[0],
                height=row_height,
                text=key,
                loc='left'
            )
            cell_left.get_text().set_fontsize(font_size)

            cell_right = tbl.add_cell(
                row=i, col=1,
                width=col_widths[1],
                height=row_height,
                text=val,
                loc='left'
            )
            cell_right.get_text().set_fontsize(font_size)

        ax.add_table(tbl)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        # ------------------------------------------------------------------
        # 3) Build Table 2 for sample data, etc.
        # ------------------------------------------------------------------
        text_data = metadata.get("sampled_text_data")
        class_names = metadata.get("first_sample_class_names")
        probs = metadata.get("first_sample_probs")
        vars_ = metadata.get("first_sample_vars")
        final_bias = metadata.get("final_classifier_bias_pre_dropout")

        second_table_rows = []
        if text_data is not None:
            wrapped_text = textwrap.fill(str(text_data), width=70)
            second_table_rows.append(("Text", wrapped_text))
        if class_names is not None:
            second_table_rows.append(("Class Names", str(class_names)))
        if probs is not None:
            second_table_rows.append(("Probability Dist.", str(probs)))
        if vars_ is not None:
            second_table_rows.append(("Variance", str(vars_)))
        # The user did NOT request removing final_classifier_bias_pre_dropout from Table 2.
        # So we keep it if present:
        if final_bias is not None:
            second_table_rows.append(("Original Classifier Bias", str(final_bias)))

        if not second_table_rows:
            return  # no second table needed

        fig_height2 = 3 + max(len(second_table_rows) * 0.7, 3)
        fig2, ax2 = plt.subplots(figsize=(10, fig_height2))
        ax2.set_title(" Sample Case Output Data", fontsize=20)
        ax2.axis('off')

        tbl2 = Table(ax2, bbox=[0, 0, 1, 1])
        row_height2 = 1.0 / (len(second_table_rows) + 1)
        col_widths2 = [0.3, 0.7]
        font_size2 = 16

        for i, (k, v) in enumerate(second_table_rows):
            left_cell = tbl2.add_cell(
                row=i, col=0,
                width=col_widths2[0],
                height=row_height2,
                text=k,
                loc='left'
            )
            left_cell.get_text().set_fontsize(font_size2)

            right_cell = tbl2.add_cell(
                row=i, col=1,
                width=col_widths2[1],
                height=row_height2,
                text=v,
                loc='left'
            )
            right_cell.get_text().set_fontsize(font_size2)

        ax2.add_table(tbl2)
        plt.tight_layout()
        second_table_path = save_path.replace(".png", "_2.png")
        plt.savefig(second_table_path, dpi=300, bbox_inches='tight')
        plt.close()


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_function = kwargs.pop('loss_function', None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.loss_function:
            loss, outputs = self.loss_function(model, inputs, return_outputs=True)
        else:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        return (loss, outputs) if return_outputs else loss


class SentimentFineTuner(SentimentBaseProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_function = None
        self.dataset_name = None
        self.csv_saver = None
        self.diamond_uncertainty = None

    def measure_starting_weights_f1(self, dataset, num_points=1):
        """
        Evaluates the UNTRAINED (i.e. 'starting weights') model on the test split
        multiple times. Instead of running the entire test set N times,
        we partition the test set into N subsets and measure F1 on each subset.

        :param dataset: A dict containing e.g. dataset['train'] and dataset['test'].
        :param num_points: number of subsets (and thus number of distinct F1 measurements).
        :return: (mean_f1, std_f1)
        """
        import numpy as np
        from transformers import Trainer
        from sklearn.metrics import f1_score

        # We require a 'test' split in dataset
        if 'test' not in dataset:
            raise ValueError("[measure_starting_weights_f1] 'test' split not found.")

        test_ds = dataset['test']
        total_len = len(test_ds)
        if num_points < 1:
            num_points = 1

        # Partition test dataset into `num_points` slices
        # e.g. if total_len=40 and num_points=5, each slice is size=8
        subset_size = max(1, total_len // num_points)

        f1_list = []
        start_idx = 0
        for i in range(num_points):
            end_idx = start_idx + subset_size
            # handle last slice
            if i == num_points - 1:
                end_idx = total_len

            subset = test_ds.select(range(start_idx, end_idx))
            start_idx = end_idx

            # Minimal trainer just to run predict
            trainer = Trainer(model=self.model)
            preds = trainer.predict(subset)
            logits = preds.predictions
            labels = np.array(subset['labels'])
            pred_classes = np.argmax(logits, axis=-1)
            f1_val = f1_score(labels, pred_classes, average='weighted')
            f1_list.append(f1_val)

        f1_array = np.array(f1_list)
        mean_f1 = float(f1_array.mean())
        std_f1 = float(f1_array.std())

        return mean_f1, std_f1
 
    def prepare_finetuning_dataset(self, dataset_name, split='train', sample_size=None, shuffle_data=True):
        self.dataset_name = dataset_name
        self.current_sample_size = sample_size if sample_size is not None else "all"
        config = self.load_dataset_config(dataset_name)
        self.num_labels = config.get('num_labels')
        if self.num_labels is None:
            raise ValueError("num_labels not set in dataset config.")

        dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        if sample_size is not None:
            total = len(dataset)
            sample_size = min(sample_size, total)
            indices = random.sample(range(total), sample_size) if shuffle_data else list(range(sample_size))
            dataset = dataset.select(indices)
        else:
            if shuffle_data:
                ds_seed = random.randint(1, 99999)
                dataset = dataset.shuffle(seed=ds_seed)

        text_field = config.get('text_field')
        label_field = config.get('label_field')
        label_mapping = config.get('label_mapping')

        texts = [self.clean_text(ex[text_field]) for ex in dataset]
        if label_mapping:
            labels = []
            for ex in dataset:
                lbl = ex[label_field]
                if str(lbl) in label_mapping:
                    labels.append(label_mapping[str(lbl)])
                elif int(lbl) in label_mapping:
                    labels.append(label_mapping[int(lbl)])
                else:
                    raise ValueError(f"Label {lbl} not found in label_mapping.")
        else:
            labels = [ex[label_field] for ex in dataset]

        tokenized = self.parallel_tokenization(texts)
        final_ds = Dataset.from_dict({
            'input_ids': [t['input_ids'].squeeze(0) for t in tokenized],
            'attention_mask': [t['attention_mask'].squeeze(0) for t in tokenized],
            'labels': labels,
            'text': texts
        })
        return final_ds

    def compute_cross_entropy(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels').long()
        outputs = model(**inputs)
        logits = outputs.logits
        loss = F.cross_entropy(logits, labels, reduction='mean')
        return (loss, outputs) if return_outputs else loss

    def compute_huber_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop('labels').float()
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        loss = F.smooth_l1_loss(logits, labels, reduction='mean')
        return (loss, outputs) if return_outputs else loss

    def compute_classification_metrics(self, model, eval_ds):
        trainer = Trainer(model=model)
        preds = trainer.predict(eval_ds)
        logits = preds.predictions
        labels = np.array(eval_ds['labels'])
        predicted_classes = np.argmax(logits, axis=-1)
        f1v = f1_score(labels, predicted_classes, average='weighted')
        acc = accuracy_score(labels, predicted_classes)
        return {'f1_score': f1v, 'accuracy': acc}

    def compute_regression_metrics(self, model, eval_ds):
        trainer = Trainer(model=model)
        preds = trainer.predict(eval_ds)
        logits = preds.predictions.squeeze(-1)
        labels = np.array(eval_ds['labels'])
        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        r2 = r2_score(labels, logits)
        return {
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'r2_score': r2
        }

    def get_loss_and_metric(self):
        if self.dataset_config is None:
            raise ValueError("Dataset config not loaded yet.")
        loss_name = self.dataset_config.get('loss_function')
        metric_name = self.dataset_config.get('evaluation_metric')
        if not loss_name or not hasattr(self, loss_name):
            raise ValueError("Missing/invalid 'loss_function'.")
        if not metric_name or not hasattr(self, metric_name):
            raise ValueError("Missing/invalid 'evaluation_metric'.")
        return getattr(self, loss_name), getattr(self, metric_name)

    def fine_tune(
            self,
            dataset,
            sentiment_context=None,
            start_from_context=True,
            save_fine_tune='yes',
            fine_tune_version_name='',
            fine_tune_quality=True,
            hyperparams=None,
            n_proportion=1.0,
            class_labels=None,
            societally_linear=None,
            starting_weight_f1_points=None,
            starting_weight_f1_plot='false',
            diamond_uncertainty=False,
            f1_dropout_plot_step=0.5
        ):
        """
        Fully updated fine_tune method that:
        - Initializes the model (base or from context).
        - Optionally measures base F1 distribution (for the 'starting_weights' Gaussian).
        - Trains via FAdam.
        - Loads fisher, calls diamond_uncertainty plots (including the new "dropout proportion" legend).
        - If user wants the Gaussian distribution plot, uses EXACT f1 from the dropout line (self.diamond_uncertainty.dropout_chosen_f1).
        """
        import torch
        import os
        import numpy as np
        import torch.nn.functional as F
        from transformers import Trainer
        from sklearn.metrics import f1_score

        if not hyperparams:
            raise ValueError("You must provide hyperparams.")

        # 1) Initialize model from scratch or from a fine-tuned context
        if start_from_context and sentiment_context:
            self.initialize_model(sentiment_context=sentiment_context)
        else:
            self.initialize_model()

        # 2) If we want the base-weight distribution, measure it BEFORE training:
        base_f1_mean, base_f1_std = None, None
        if (starting_weight_f1_plot.lower() == 'true'
            and starting_weight_f1_points is not None):
            base_f1_mean, base_f1_std = self.measure_starting_weights_f1(
                dataset=dataset,
                num_points=starting_weight_f1_points
            )

        # 3) Build trainer with FAdam, then train
        trainer = self._build_trainer(dataset, hyperparams)
        trainer.train()

        # 4) Initialize Diamond_uncertainty & load fisher
        self.diamond_uncertainty = Diamond_uncertainty(self.model)
        fadam_optimizer = trainer.optimizer
        self.diamond_uncertainty.load_fisher_from_fadam(fadam_optimizer, invert=True)

        # 5) Save final bias
        final_bias = None
        for name, param in self.model.named_parameters():
            if "classifier.bias" in name:
                final_bias = param.detach().cpu().tolist()
                break
        self.diamond_uncertainty.final_classifier_bias_pre_dropout = final_bias

        # 6) If fisher_inverse is available, produce fisher-based plots
        if self.diamond_uncertainty.fisher_inverse is not None:
            # (A) Average FIM bar charts & histograms
            # Now you can pass additional font-size toggles if you like
            self.diamond_uncertainty.all_layers_mean_Diag_FIM(
                fine_tune_version_name=fine_tune_version_name,
                title_fontsize=16,
                axis_label_fontsize=14,
            )
            self.diamond_uncertainty.generate_histograms_matplotlib(
                fine_tune_version_name=fine_tune_version_name,
                num_labels=self.num_labels,
                hidden_size=self.model.config.hidden_size,
                class_labels=class_labels
            )

            # (B) Additional Diamond Uncertainty features if diamond_uncertainty=True
            if diamond_uncertainty:
                self.model.eval()
                # Single-sample probability/uncertainty figure
                first_test_item = dataset['test'][0]
                input_ids = torch.tensor(first_test_item['input_ids']).unsqueeze(0).to(self.model.device)
                attention_mask = torch.tensor(first_test_item['attention_mask']).unsqueeze(0).to(self.model.device)
                with torch.no_grad():
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                logits = out.logits
                hidden_cls = out.hidden_states[-1][:, 0, :]
                probs = F.softmax(logits, dim=-1).squeeze(0).cpu()

                single_unc = self.diamond_uncertainty.compute_softmax_uncertainty_for_samples(
                    logits_probs=probs.unsqueeze(0),
                    hidden_mean=hidden_cls,
                    variance=self.diamond_uncertainty.fisher_inverse,
                    consider_bias=True
                ).squeeze(0).cpu()

                local_class_labels = []
                for i in range(self.num_labels):
                    if class_labels and i < len(class_labels):
                        local_class_labels.append(class_labels[i])
                    else:
                        local_class_labels.append(f"Class_{i}")

                # Plot the single-sample fisher uncertainty curve
                self.diamond_uncertainty.plot_uncertainty_for_first_sample(
                    prob_vec=probs.numpy(),
                    var_vec=single_unc.numpy(),
                    class_labels=local_class_labels,
                    fine_tune_version_name=fine_tune_version_name
                )

                # F1 vs dropout lines
                # (the code within 'plot_f1_dropout_4plots' is updated to store 
                #  self.dropout_chosen_f1 and rename legend to 'dropout proportion=...')
                self.diamond_uncertainty.plot_f1_dropout_4plots(
                    trainer=trainer,
                    test_dataset=dataset['test'],
                    f1_dropout_step=f1_dropout_plot_step,
                    plot_title="F1 vs. dropout for Diamond Uncertainty",
                    class_labels=class_labels,
                    n_proportion=n_proportion,
                    fine_tune_version_name=fine_tune_version_name
                )

                # (C) If user wants the starting-weight F1 Gaussian, use 
                #     the EXACT dropout-chosen F1 if it exists:
                if (starting_weight_f1_plot.lower() == 'true'
                    and starting_weight_f1_points is not None
                    and base_f1_mean is not None
                    and base_f1_std is not None):
                    
                    # By default, we compare final model's F1:
                    final_f1 = 0.0
                    eval_metrics = {}
                    if fine_tune_quality:
                        _, metric_func = self.get_loss_and_metric()
                        eval_metrics = metric_func(self.model, dataset['test'])
                    if 'f1_score' in eval_metrics:
                        final_f1 = eval_metrics['f1_score']

                    # If we have a chosen_f1 from the dropout line, override:
                    if hasattr(self.diamond_uncertainty, "dropout_chosen_f1"):
                        final_f1 = self.diamond_uncertainty.dropout_chosen_f1

                    # Now plot the Gaussian with final_f1 as the red X
                    self.diamond_uncertainty.plot_starting_weight_f1_gaussian(
                        f1_mean=base_f1_mean,
                        f1_std=base_f1_std,
                        chosen_f1=final_f1,
                        n_proportion=n_proportion,
                        fine_tune_version_name=fine_tune_version_name,
                        sentiment_context=(sentiment_context if start_from_context else "base-weights"),
                        marker_size=12
                    )

        # 7) Optionally evaluate & save the model
        eval_metrics = {}
        if save_fine_tune == 'yes':
            if fine_tune_quality:
                _, metric_func = self.get_loss_and_metric()
                eval_metrics = metric_func(self.model, dataset['test'])

            save_info = self._save_model_and_metadata(
                fine_tune_version_name,
                hyperparams,
                self.dataset_name,
                eval_metrics,
                class_labels=class_labels,
                societally_linear=societally_linear,
                start_from_context=start_from_context,
                sentiment_context=sentiment_context
            )

            # Save the covariance matrix
            if self.diamond_uncertainty.fisher_inverse is not None and save_info is not None:
                save_dir = save_info['save_directory']
                cov_path = os.path.join(save_dir, "cov_matrix.pt")
                torch.save(self.diamond_uncertainty.fisher_inverse, cov_path)
        else:
            save_info = None

        # 8) Optionally store results in CSV
        if self.csv_saver and self.diamond_uncertainty.fisher_inverse is not None:
            from transformers import DataCollatorWithPadding
            from torch.utils.data import DataLoader
            self.model.eval()

            test_set_no_text = dataset['test'].remove_columns(['text'])
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            test_dl = DataLoader(test_set_no_text, batch_size=16, collate_fn=data_collator)
            device = next(self.model.parameters()).device

            all_probs = []
            all_uncs = []
            all_labels = dataset['test']['labels']
            if not isinstance(all_labels, list):
                all_labels = list(all_labels)

            with torch.no_grad():
                for batch in test_dl:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    logits = out.logits
                    hidden_cls = out.hidden_states[-1][:, 0, :]
                    probs = F.softmax(logits, dim=-1)
                    unc_vals = self.diamond_uncertainty.compute_softmax_uncertainty_for_samples(
                        logits_probs=probs,
                        hidden_mean=hidden_cls,
                        variance=self.diamond_uncertainty.fisher_inverse,
                        consider_bias=True
                    )
                    all_probs.append(probs.cpu())
                    all_uncs.append(unc_vals.cpu())

            import torch
            all_probs = torch.cat(all_probs, dim=0).numpy()
            all_uncs = torch.cat(all_uncs, dim=0).numpy()

            self.csv_saver.save_finetune_results(
                label_array=all_labels,
                probs_array=all_probs,
                unc_array=all_uncs,
                label_column_name="gold_label"
            )

        return save_info

    def _build_trainer(self, dataset, hyperparams):
        training_args = TrainingArguments(
            output_dir='./results',
            learning_rate=hyperparams['lr'],
            per_device_train_batch_size=hyperparams['batch_size'],
            num_train_epochs=hyperparams['epochs'],
            weight_decay=hyperparams.get('weight_decay', 0.0),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        class FadamTrainer(CustomTrainer):
            def create_optimizer(inner_self):
                if inner_self.optimizer is None:
                    decay_params = []
                    no_decay_params = []
                    for n, p in inner_self.model.named_parameters():
                        if p.requires_grad:
                            if any(nd in n for nd in ["bias", "LayerNorm.weight"]):
                                no_decay_params.append(p)
                            else:
                                decay_params.append(p)
                    optimizer_grouped_params = [
                        {"params": decay_params, "weight_decay": hyperparams.get('weight_decay', 0.0)},
                        {"params": no_decay_params, "weight_decay": 0.0},
                    ]
                    inner_self.optimizer = FAdam(
                        optimizer_grouped_params,
                        lr=hyperparams['lr'],
                        betas=hyperparams.get('betas', (0.9, 0.999)),
                        weight_decay=hyperparams.get('weight_decay', 0.0),
                        clip=hyperparams.get('clip', 1.0),
                        p=hyperparams.get('p', 0.5),
                        eps=hyperparams.get('eps', 1e-8),
                    )
                return inner_self.optimizer

        self.loss_function, _ = self.get_loss_and_metric()
        trainer = FadamTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['test'],
            data_collator=data_collator,
            loss_function=self.loss_function
        )
        return trainer
    def _save_model_and_metadata(
        self,
        version_name: str,
        hyperparams: dict,
        dataset_name: str,
        eval_metrics: dict,
        class_labels=None,
        societally_linear=None,
        start_from_context=False,
        sentiment_context=None
    ):
        import os
        import json
        from src.metadata_module import MetadataCSVDataSaver

        # 1) Build save directory under local_experiments
        save_dir = os.path.join('model_versions', 'local_experiments', version_name)
        os.makedirs(save_dir, exist_ok=True)

        # 2) Save the model weights/config
        self.model.save_pretrained(save_dir)

        # 3) Prepare metadata entry
        metadata = {
            'version_name': version_name,
            'dataset': dataset_name,
            # point to the directory only; HF loader will auto-detect .bin or .safetensors
            'weights_filepath': save_dir,
            'output_mode': self.num_labels,
            'model_type': self.model_size,
            'hyperparameters': hyperparams,
            'eval_metrics': eval_metrics,
            'sample_size': getattr(self, 'current_sample_size', 'unknown'),
            'generated_at': datetime.utcnow().isoformat() + 'Z'
        }

        # 4) Append to data/metadata.json via the existing helper
        meta_path = MetadataCSVDataSaver._metadata_path()
        try:
            existing = json.loads(meta_path.read_text(encoding='utf-8'))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
        existing.append(metadata)
        meta_path.write_text(json.dumps(existing, indent=4), encoding='utf-8')

        return {'save_directory': save_dir, 'version_name': version_name}


class SentimentInferencer(SentimentBaseProcessor):
    """For inference. Unchanged outside scope."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loaded_covariance = None

    def load_covariance_for_context(self, sentiment_context):
            if sentiment_context is None:
                return
            meta_path = DataFiles.path("metadata.json")
            with meta_path.open('r', encoding='utf-8') as f:
                meta_list = json.load(f)
            entry = next((m for m in meta_list if m['version_name'] == sentiment_context), None)
            if not entry:
                print(f"No metadata found for context '{sentiment_context}'.")
                return
            cov_path = os.path.join(os.path.dirname(entry['weights_filepath']), "cov_matrix.pt")
            if not os.path.isfile(cov_path):
                print(f"No covariance file found at {cov_path}.")
                return
            self.loaded_covariance = torch.load(cov_path)
            print(f"Loaded covariance from {cov_path} for context '{sentiment_context}'.")

    def run_inference(self, texts, sentiment_context=None, dataset_name=None, sentiment_mode='classic'):
        if dataset_name:
            self.load_dataset_config(dataset_name)
        if self.model is None:
            if sentiment_context:
                self.initialize_model(sentiment_context=sentiment_context)
            else:
                raise ValueError("Model not initialized.")
        self.load_covariance_for_context(sentiment_context)
        cleaned_texts = [self.clean_text(tx) for tx in texts]
        total_chunks = self._count_chunks(cleaned_texts, sentiment_mode)
        predictions_out = []
        self.model.eval()
        with torch.no_grad(), tqdm(total=total_chunks, desc="Running Inference") as pbar:
            for text in cleaned_texts:
                chunk_preds, chunk_uncs = [], []
                chunked_tokenized = self._split_and_tokenize_text(text, sentiment_mode)
                for token_dict in chunked_tokenized:
                    input_ids = token_dict['input_ids'].to(self.model.device)
                    attention_mask = token_dict['attention_mask'].to(self.model.device)
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    logits = out.logits.cpu()
                    hidden = out.hidden_states[-1][:, 0, :].cpu()
                    chunk_pred, chunk_unc = self._compute_chunk_prediction_and_uncertainty(logits=logits, hidden=hidden)
                    chunk_preds.append(chunk_pred)
                    chunk_uncs.append(chunk_unc)
                    pbar.update(1)
                final_pred, final_unc = self._aggregate_chunk_preds(chunk_preds, chunk_uncs)
                predictions_out.append((final_pred, final_unc))
        return predictions_out

    def _split_and_tokenize_text(self, text, sentiment_mode):
        tokens = self.tokenizer.tokenize(text)
        tokens_len = len(tokens)
        if sentiment_mode == 'longform' and tokens_len > self.max_seq_length:
            chunked = []
            for i in range(0, tokens_len, self.max_seq_length):
                chunk_sub = tokens[i:i+self.max_seq_length]
                chunk_sub_str = self.tokenizer.convert_tokens_to_string(chunk_sub)
                chunked.append(chunk_sub_str)
        elif sentiment_mode == 'snapshot' and tokens_len > self.max_seq_length:
            half_seq = self.max_seq_length // 2
            first_tokens = tokens[:half_seq]
            last_tokens = tokens[-half_seq:]
            chunked = [
                self.tokenizer.convert_tokens_to_string(first_tokens),
                self.tokenizer.convert_tokens_to_string(last_tokens)
            ]
        else:
            chunked = [text]
        token_dicts = []
        for c in chunked:
            tok = self.tokenize_text(c)
            token_dicts.append(tok)
        return token_dicts

    def _count_chunks(self, cleaned_texts, sentiment_mode):
        total = 0
        for text in cleaned_texts:
            t = self.tokenizer.tokenize(text)
            if sentiment_mode == 'longform' and len(t) > self.max_seq_length:
                num = (len(t) + self.max_seq_length - 1) // self.max_seq_length
            elif sentiment_mode == 'snapshot' and len(t) > self.max_seq_length:
                num = 2
            else:
                num = 1
            total += num
        return total

    def _compute_chunk_prediction_and_uncertainty(self, logits, hidden):
        if self.output_mode == 1:
            val = logits.mean().item()
            uncertainty = None
            return val, uncertainty
        else:
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            return probs.tolist(), None

    def _aggregate_chunk_preds(self, chunk_preds, chunk_uncs):
        if not chunk_preds:
            return None, None
        if self.output_mode == 1:
            pred_val = float(np.mean(chunk_preds))
            valid_uncs = [u for u in chunk_uncs if u is not None]
            unc_val = float(np.mean(valid_uncs)) if valid_uncs else None
            return pred_val, unc_val
        else:
            arr = np.array(chunk_preds)
            mean_probs = arr.mean(axis=0)
            valid_uncs = [np.array(u) for u in chunk_uncs if u is not None]
            mean_uncs = np.mean(valid_uncs, axis=0) if valid_uncs else None
            return mean_probs.tolist(), mean_uncs.tolist() if mean_uncs is not None else None


class SentimentCSVDataSaver:
    def __init__(self, dataset_handler, sentiment_context=None):
        self.dataset_handler = dataset_handler
        self.sentiment_context = sentiment_context
        self.output_mode, self.class_labels, self.continuous_label = self.get_metadata_info()

    def get_metadata_info(self):
        if self.sentiment_context is None:
            return self.dataset_handler.num_labels, None, None
        meta_path = DataFiles.path("metadata.json")
        with meta_path.open('r', encoding='utf-8') as f:
            metadata_list = json.load(f)
        metadata = next((item for item in metadata_list if item['version_name'] == self.sentiment_context), None)
        if metadata is None:
            raise FileNotFoundError(f"No metadata found for '{self.sentiment_context}'.")
        return (
            metadata.get('output_mode'),
            metadata.get('class_labels', None),
            metadata.get('continuous_label', 'Sentiment')
        )
    
    def save_inference_results(self, predictions):
        df = self.dataset_handler.read_csv()
        if self.output_mode == 1:
            col_name = f"continuous: {self.continuous_label}"
            df[col_name] = predictions
        else:
            num_labels = self.output_mode
            labels = self.class_labels if self.class_labels else [f'Class_{i}' for i in range(num_labels)]
            for i, label in enumerate(labels):
                column_header = f"Probability Distribution {i+1}: {label}"
                df[column_header] = [p[i] for p in predictions]
        self.dataset_handler.write_csv(df)

    def save_finetune_results(self, label_array, probs_array, unc_array=None, label_column_name="gold_label"):
        df = self.dataset_handler.read_csv()

        # IMPORTANT: reset the dataframe index so we can safely assign to df[...] = ...
        df.reset_index(drop=True, inplace=True)

        num_rows_csv = len(df)
        N = len(label_array)

        # Match lengths
        if num_rows_csv > N:
            df = df.head(N)
            num_rows_csv = N
        elif num_rows_csv < N:
            print(f"[Warning] CSV has {num_rows_csv} rows, but we have {N} samples. "
                f"Will only store first {num_rows_csv} from label/probs.")
            N = num_rows_csv

        df[label_column_name] = label_array[:N]

        prob_strs = []
        for i in range(N):
            row_probs = probs_array[i].tolist()
            prob_strs.append(str(row_probs))
        df["prob_distribution"] = prob_strs[:N]

        import numpy as np
        if unc_array is not None:
            unc_strs = []
            for i in range(N):
                row_unc = unc_array[i].tolist()
                unc_strs.append(str(row_unc))
            df["diamond_uncertainty"] = unc_strs[:N]
        else:
            df["diamond_uncertainty"] = [np.nan]*N

        # Reorder columns so "prob_distribution" and "diamond_uncertainty" are at the end
        desired_order = [c for c in df.columns if c not in ("prob_distribution","diamond_uncertainty")]
        desired_order += ["prob_distribution","diamond_uncertainty"]
        df = df[desired_order]

        self.dataset_handler.write_csv(df)
