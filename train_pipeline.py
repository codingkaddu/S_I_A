"""
train_pipeline.py -- Support Integrity Auditor (SIA) . Stage 2
==============================================================
Production-grade Hugging Face fine-tuning of microsoft/deberta-v3-small
for binary mismatch classification on pseudo-labelled support tickets.

Architecture
------------
1. Metadata-text fusion (Channel | Category | Subject | Description)
2. Stratified 80/20 train/val split preserving label balance
3. DeBERTa-v3-small tokenisation with max_length=128 (CPU-optimised)
4. Class-weight-adjusted CrossEntropyLoss for imbalance mitigation
5. HuggingFace Trainer with per-epoch accuracy, macro-F1, per-class recall
6. Model + tokenizer checkpoint saved to --model_dir

CPU-Optimisation Notes
----------------------
- max_length=128 (aggressive truncation)
- fp16/bf16 disabled
- gradient_accumulation_steps to simulate larger effective batch
- dataloader_num_workers=0 (Windows-safe)
- no_cuda=True forced

Usage
-----
    python train_pipeline.py
    python train_pipeline.py --data_path ./pseudo_labelled_tickets.csv --epochs 2 --batch_size 8
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# 0 . ARGUMENT PARSING
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SIA Stage 2 -- DeBERTa-v3-small Mismatch Classifier Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(r"C:\Users\Rudra\Desktop\Mars\pseudo_labelled_tickets.csv"),
        help="Path to the pseudo-labelled CSV from Stage 1.",
    )
    parser.add_argument(
        "--raw_data_path",
        type=Path,
        default=Path(r"C:\Users\Rudra\Desktop\Mars\archive\customer_support_tickets.csv"),
        help="Path to the original CSV (used for Ticket_Channel metadata).",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(r"C:\Users\Rudra\Desktop\Mars\models"),
        help="Directory to save the fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-small",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device training batch size.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate for AdamW optimiser.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum token length (aggressive truncation for CPU).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * this).",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# 1 . CUSTOM DATASET
# -----------------------------------------------------------------------------

class TicketDataset(torch.utils.data.Dataset):
    """
    Custom PyTorch Dataset that holds pre-tokenised encodings and labels.
    """

    def __init__(self, encodings: dict, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# -----------------------------------------------------------------------------
# 2 . WEIGHTED TRAINER (class-imbalance aware)
# -----------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    """
    Subclass of HuggingFace Trainer that injects class-weight-adjusted
    CrossEntropyLoss to combat label imbalance.
    """

    def __init__(self, class_weights: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights.to(device=logits.device, dtype=logits.dtype)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# 3 . METRIC COMPUTATION
# -----------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """
    Compute binary accuracy, macro-F1, and per-class recall.
    Called by the Trainer at the end of each evaluation epoch.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)

    # Per-class recall: class 0 = consistent, class 1 = mismatch
    recall_per_class = recall_score(
        labels, preds, average=None, zero_division=0
    )
    recall_consistent = float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0
    recall_mismatch = float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(f1_macro, 4),
        "recall_consistent": round(recall_consistent, 4),
        "recall_mismatch": round(recall_mismatch, 4),
    }


# -----------------------------------------------------------------------------
# 4 . METADATA-TEXT FUSION
# -----------------------------------------------------------------------------

def fuse_metadata_text(df: pd.DataFrame) -> pd.Series:
    """
    Build a single formatted text sequence per ticket by combining
    structured metadata fields with free-text content.

    Format: "Channel: {channel} | Category: {category} | Subject: {subject} | Description: {desc}"

    This gives the model access to both categorical signals and natural
    language content in a single input sequence.
    """

    def _fuse_row(row):
        channel = str(row.get("Ticket_Channel", "Unknown")).strip()
        category = str(row.get("ticket_type", "Unknown")).strip()
        subject = str(row.get("ticket_subject", "")).strip()
        description = str(row.get("ticket_description", "")).strip()

        return (
            f"Channel: {channel} | "
            f"Category: {category} | "
            f"Subject: {subject} | "
            f"Description: {description}"
        )

    return df.apply(_fuse_row, axis=1)


# -----------------------------------------------------------------------------
# 5 . MAIN PIPELINE
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    t0 = time.time()

    print("=" * 70)
    print("  SIA Stage 2 -- DeBERTa-v3-small Mismatch Classifier Training")
    print("=" * 70)
    print(f"  Model        : {args.model_name}")
    print(f"  Data         : {args.data_path}")
    print(f"  Output       : {args.model_dir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  Grad. accum  : {args.gradient_accumulation_steps}")
    print(f"  Effective BS : {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Max length   : {args.max_length}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device       : CPU (forced)")
    print(f"  Seed         : {args.seed}")
    print("=" * 70)

    # -- Reproducibility ------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ==========================================================================
    # STEP 1 . LOAD DATA & FUSE METADATA
    # ==========================================================================
    print("\n[Step 1/7] Loading pseudo-labelled data ...")

    if not args.data_path.exists():
        print(f"  ERROR: Pseudo-labelled CSV not found at {args.data_path}")
        print("  -> Run notebook.ipynb (Stage 1) first to generate it.")
        sys.exit(1)

    df = pd.read_csv(args.data_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Merge Ticket_Channel from the raw CSV if available
    if args.raw_data_path.exists():
        print(f"  Merging Ticket_Channel from: {args.raw_data_path.name}")
        df_raw = pd.read_csv(args.raw_data_path, usecols=["Ticket_Channel"])
        if len(df_raw) == len(df):
            df["Ticket_Channel"] = df_raw["Ticket_Channel"].values
        else:
            print(f"  WARNING: Row count mismatch ({len(df_raw)} vs {len(df)}), skipping merge")
            df["Ticket_Channel"] = "Unknown"
    else:
        print(f"  WARNING: Raw CSV not found, using 'Unknown' for Ticket_Channel")
        df["Ticket_Channel"] = "Unknown"

    # Validate target column
    if "is_mismatch" not in df.columns:
        print("  ERROR: Column 'is_mismatch' not found in the dataset.")
        sys.exit(1)

    # Fuse metadata + text
    print("  Fusing metadata and text fields ...")
    df["fused_text"] = fuse_metadata_text(df)
    print(f"  Sample fused text:\n    {df['fused_text'].iloc[0][:120]}...")

    texts = df["fused_text"].tolist()
    labels = df["is_mismatch"].values.astype(int)

    print(f"  Label distribution: {dict(pd.Series(labels).value_counts().sort_index())}")

    # ==========================================================================
    # STEP 2 . STRATIFIED TRAIN/VALIDATION SPLIT
    # ==========================================================================
    print("\n[Step 2/7] Stratified 80/20 train/validation split ...")

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=args.seed,
    )
    print(f"  Train : {len(train_texts):,} samples")
    print(f"  Val   : {len(val_texts):,} samples")
    print(f"  Train label dist : {dict(pd.Series(train_labels).value_counts().sort_index())}")
    print(f"  Val   label dist : {dict(pd.Series(val_labels).value_counts().sort_index())}")

    # ==========================================================================
    # STEP 3 . TOKENISATION
    # ==========================================================================
    print(f"\n[Step 3/7] Loading tokenizer: {args.model_name} ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"  Tokenizing train set ({len(train_texts):,} samples, max_length={args.max_length}) ...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    print(f"  Tokenizing val set ({len(val_texts):,} samples, max_length={args.max_length}) ...")
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding=True,
        max_length=args.max_length,
        return_tensors="pt",
    )

    train_dataset = TicketDataset(train_encodings, train_labels)
    val_dataset = TicketDataset(val_encodings, val_labels)

    print(f"  Train dataset : {len(train_dataset)} examples")
    print(f"  Val dataset   : {len(val_dataset)} examples")

    # ==========================================================================
    # STEP 4 . CLASS WEIGHTS FOR IMBALANCE MITIGATION
    # ==========================================================================
    print("\n[Step 4/7] Computing class weights for imbalance mitigation ...")

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=train_labels,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"  Class 0 (consistent) weight : {class_weights[0]:.4f}")
    print(f"  Class 1 (mismatch)   weight : {class_weights[1]:.4f}")

    # ==========================================================================
    # STEP 5 . MODEL INITIALISATION
    # ==========================================================================
    print(f"\n[Step 5/7] Loading model: {args.model_name} ...")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "consistent", 1: "mismatch"},
        label2id={"consistent": 0, "mismatch": 1},
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable_params:,}")

    # ==========================================================================
    # STEP 6 . TRAINING
    # ==========================================================================
    print(f"\n[Step 6/7] Configuring Trainer (CPU-optimised) ...")

    output_dir = args.model_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        # -- Epochs & batching ---------------------------------------------
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,  # eval can use bigger batches
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        # -- Optimiser -----------------------------------------------------
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=100,
        # -- CPU safety ----------------------------------------------------
        fp16=False,
        bf16=False,
        use_cpu=True,
        dataloader_num_workers=0,
        # -- Evaluation strategy -------------------------------------------
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        # -- Logging -------------------------------------------------------
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        # -- Misc ----------------------------------------------------------
        seed=args.seed,
        save_total_limit=2,
        disable_tqdm=False,
    )

    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    total_steps = (len(train_dataset) // args.batch_size // args.gradient_accumulation_steps) * args.epochs
    print(f"  Total training steps : ~{total_steps:,}")
    print(f"  Effective batch size : {args.batch_size * args.gradient_accumulation_steps}")
    print(f"\n{'-' * 70}")
    print("  TRAINING STARTED")
    print(f"{'-' * 70}\n")

    train_result = trainer.train()

    print(f"\n{'-' * 70}")
    print("  TRAINING COMPLETE")
    print(f"{'-' * 70}")
    print(f"  Total training time : {train_result.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Samples/second      : {train_result.metrics.get('train_samples_per_second', 0):.2f}")

    # -- Final evaluation with full report ---------------------------------
    print(f"\n{'-' * 70}")
    print("  FINAL VALIDATION EVALUATION")
    print(f"{'-' * 70}\n")

    eval_results = trainer.evaluate()
    for key, val in sorted(eval_results.items()):
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "")
            if isinstance(val, float):
                print(f"  {metric_name:25s} : {val:.4f}")
            else:
                print(f"  {metric_name:25s} : {val}")

    # Full classification report
    print(f"\n  {'=' * 50}")
    print("  DETAILED CLASSIFICATION REPORT")
    print(f"  {'=' * 50}\n")

    val_preds = trainer.predict(val_dataset)
    pred_labels = np.argmax(val_preds.predictions, axis=-1)
    report = classification_report(
        val_labels,
        pred_labels,
        target_names=["Consistent (0)", "Mismatch (1)"],
        digits=4,
    )
    print(report)

    # ==========================================================================
    # STEP 7 . SAVE MODEL ARTEFACTS
    # ==========================================================================
    print(f"\n[Step 7/7] Saving model artefacts to {args.model_dir} ...")

    final_model_dir = args.model_dir / "final"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    # Save model + tokenizer
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    # Save training config for reproducibility
    config = {
        "model_name": args.model_name,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "seed": args.seed,
        "class_weights": class_weights.tolist(),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "final_metrics": {
            k.replace("eval_", ""): round(v, 4) if isinstance(v, float) else v
            for k, v in eval_results.items()
            if k.startswith("eval_")
        },
    }
    config_path = final_model_dir / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"  Model saved to     : {final_model_dir}")
    print(f"  Config saved to    : {config_path}")

    # List saved files
    saved_files = list(final_model_dir.iterdir())
    print(f"  Saved artefacts    : {len(saved_files)} files")
    for fp in sorted(saved_files):
        size_kb = fp.stat().st_size / 1024
        print(f"    {fp.name:40s} ({size_kb:,.1f} KB)")

    # -- Summary --------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE -- Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")
    print(f"  Best macro-F1 : {eval_results.get('eval_macro_f1', 'N/A')}")
    print(f"  Model dir     : {final_model_dir}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
