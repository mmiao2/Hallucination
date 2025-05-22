from transformers import TrainerCallback, TrainerControl, TrainerState
import os
import csv
import torch
from utils import (
    miscalibration_analysis,
    inaccuracy_analysis,
    hallucination_analysis
)

class CallBackTrainer(TrainerCallback):
    """
    Runs accuracy, hallucination, and miscalibration analysis every 5 steps,
    toggling the model into eval mode and back to train mode, and appends results to CSV.
    """
    def __init__(
        self,
        train_dataset_texts,    # HF Dataset or dict-like with ["x","y","names","gold"]
        train_datasets,         # dict of alpha→training DataFrame
        p_datasets,             # dict of alpha→powerlaw DataFrame
        tokenizer,
        alpha,
        device,
        output_csv_path="metrics_log.csv",
        epsilon=0.1,
        batch_size=32,
    ):
        self.train_dataset_texts = train_dataset_texts
        self.train_datasets      = train_datasets
        self.p_datasets          = p_datasets
        self.tokenizer           = tokenizer
        self.alpha               = alpha
        self.device              = device
        self.epsilon             = epsilon
        self.batch_size          = batch_size
        self.output_csv_path     = output_csv_path

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        ## only main GPU will write to CSV
        if state.global_step % 10!= 0 or args.local_rank != 0:
            return

        step = state.global_step
        model = kwargs["model"]
        model.eval()
        with torch.no_grad():
            # attribute loss and accuracy
            loss_rate, inacc_rate = inaccuracy_analysis(
                model,
                self.train_dataset_texts,
                self.tokenizer,
                self.batch_size,
                self.alpha,
                self.device
            )
            # hallucination rate
            hall_rate = hallucination_analysis(
                model,
                self.train_dataset_texts,
                self.tokenizer,
                self.batch_size,
                self.alpha,
                self.device
            )
            # total variation & regret
            miscal_rate, regret, _ = miscalibration_analysis(
                self.p_datasets,
                self.train_datasets,
                model,
                self.tokenizer,
                self.alpha,
                self.epsilon,
                self.device
            )
        model.train()

        row = {
            "step": step,
            "inacc_rate": inacc_rate,
            "loss_rate": loss_rate,
            "hallucination_rate": hall_rate,
            "miscalibration_rate": miscal_rate,
            "total_regret": regret
        }

        # append to CSV
        exists = os.path.exists(self.output_csv_path)
        with open(self.output_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "inacc_rate",
                    "loss_rate", 
                    "hallucination_rate",
                    "miscalibration_rate",
                    "total_regret"
                ]
            )
            if not exists or os.stat(self.output_csv_path).st_size == 0:
                writer.writeheader()
            writer.writerow(row)

        print(
            f"[Step {step}] "
            f"Inaccuracy={inacc_rate:.4f} "
            f"Loss={loss_rate:.4f} "
            f"Hallucination={hall_rate:.4f} "
            f"Miscalibration={miscal_rate:.4f}"
            f"Regret={regret:.4f}"
        )
