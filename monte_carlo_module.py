import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from alb_s1 import SentimentBaseProcessor
from alb_s1 import SentimentCSVDataSaver  # We'll reuse your CSV saver style


class MonteCarloDropoutInferencer(SentimentBaseProcessor):

    def __init__(self, max_seq_length=128, model_size='albert-base'):
        super().__init__(max_seq_length=max_seq_length, model_size=model_size)

    def run_monte_carlo_dropout_inference(
        self, 
        texts, 
        n_samples=10, 
        sentiment_context=None, 
        dataset_name=None
    ):

        if dataset_name:
            self.load_dataset_config(dataset_name)
        if self.model is None:
            self.initialize_model(sentiment_context=sentiment_context)


        self.model.train()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)


        cleaned_texts = [self.clean_text(tx) for tx in texts]

        results = []
        total_iterations = len(cleaned_texts) * n_samples

        with tqdm(total=total_iterations, desc="MC Dropout Inference", unit="sample") as pbar:
            for text in cleaned_texts:
                tokenized = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_seq_length,
                    return_tensors='pt'
                ).to(device)

                sample_outputs = []
                for _ in range(n_samples):
                    with torch.no_grad():
                        outputs = self.model(**tokenized)
                        logits = outputs.logits

                        if self.output_mode == 1:

                            val = logits.squeeze(-1).mean().item()
                            sample_outputs.append(val)
                        else:

                            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                            sample_outputs.append(probs)

                    pbar.update(1)


                sample_outputs = np.array(sample_outputs)

                if self.output_mode == 1:

                    mean_val = float(sample_outputs.mean())
                    std_val = float(sample_outputs.std())
                    results.append({
                        "mc_mean": mean_val,
                        "mc_std": std_val
                    })
                else:

                    mean_probs = sample_outputs.mean(axis=0).tolist()
                    std_probs = sample_outputs.std(axis=0).tolist()
                    results.append({
                        "mc_mean": mean_probs,
                        "mc_std": std_probs
                    })

        # Switch back to eval mode
        self.model.eval()
        return results


class MonteCarloDropoutCSVDataSaver(SentimentCSVDataSaver):

    def save_monte_carlo_results(self, mc_results):

        df = self.dataset_handler.read_csv()
        num_rows_csv = len(df)


        if len(mc_results) > num_rows_csv:
            print(f"[Warning] We have {len(mc_results)} MC results but only {num_rows_csv} rows in CSV.")
            mc_results = mc_results[:num_rows_csv]

        # If single-value output
        if self.output_mode == 1:
            # We'll create two new columns: "MC Continuous Mean", "MC Continuous Std"
            mean_vals = []
            std_vals = []
            for r in mc_results:
                std_vals.append(r["mc_std"])

            df["MC Continuous Std"] = std_vals

        else:
            # classification => each row has "mc_mean" as an array of length = num_labels,
            #                   "mc_std"  as an array of length = num_labels.
            # We'll do one column per class for means, and one column per class for std dev.
            # E.g. "MC Mean Probability Dist 1: Class_0", "MC Std Probability Dist 1: Class_0"
            num_labels = self.output_mode

            # If we do have class labels from metadata, use them. Otherwise "Class_0", etc.
            if not self.class_labels:
                class_labels = [f"Class_{i}" for i in range(num_labels)]
            else:
                class_labels = self.class_labels

            # We'll build separate arrays for each column
            stds_per_label  = [[] for _ in range(num_labels)]

            for r in mc_results:
                # r["mc_mean"] => list of shape [num_labels], r["mc_std"] => same
                std_arr  = r["mc_std"]
                for i in range(num_labels):
                    stds_per_label[i].append(std_arr[i])

            # Now we assign them to new columns
            for i in range(num_labels):
                col_std  = f"MC Std Probability Dist {i+1}: {class_labels[i]}"
                df[col_std]  = stds_per_label[i]

        # Write back to CSV
        self.dataset_handler.write_csv(df)
        print("[MC Dropout] CSV updated with MC mean and std columns.")
