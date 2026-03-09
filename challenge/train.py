#!/usr/bin/env python3
"""
MedGemma Fine-tuning Script with LoRA

Fine-tunes MedGemma-4b-it on radiology report generation using LoRA.
Supports ReXGradient-160K dataset or custom CSV files.

Usage:
    # Download model and dataset
    python train.py --download-model --download-dataset

    # Quick test run
    python train.py --train-size 100 --val-size 20 --epochs 1

    # Full training
    python train.py --epochs 3 --batch-size 4 --learning-rate 2e-4

    # Custom CSV training
    python train.py --data-path ./custom_data.csv --findings-col "Findings" --impression-col "Impression"
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

import mlflow

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma with LoRA for radiology report generation")

    # Model & Data
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/medgemma-4b-it",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./models/medgemma-4b-it",
        help="Local directory to download/load the model"
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="rajpurkarlab/ReXGradient-160K",
        help="HuggingFace dataset ID"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./datasets/ReXGradient-160K",
        help="Local directory to download/load the dataset"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to custom CSV file (overrides dataset-id)"
    )
    parser.add_argument(
        "--findings-col",
        type=str,
        default="findings",
        help="Column name for findings in custom CSV"
    )
    parser.add_argument(
        "--impression-col",
        type=str,
        default="impression",
        help="Column name for impression in custom CSV"
    )

    # Download options
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Download model from HuggingFace Hub to local directory"
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download dataset from HuggingFace Hub to local directory"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per device"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Warmup ratio"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=16,
        help="LoRA alpha"
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--target-modules",
        type=str,
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Target modules for LoRA"
    )

    # Data processing
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Dataset split for training"
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="validation",
        help="Dataset split for validation"
    )
    parser.add_argument(
        "--train-size",
        type=int,
        default=None,
        help="Limit training samples (for testing)"
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=None,
        help="Limit validation samples (for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Output & Logging
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for training results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./checkpoints",
        help="Directory for checkpoints"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="medgemma-finetune",
        help="MLFlow experiment name"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLFlow logging"
    )

    return parser.parse_args()


def download_model(model_id: str, model_dir: str, hf_token: str):
    """Download model from HuggingFace Hub to local directory."""
    print(f"📥 Downloading model from {model_id} to {model_dir}")
    print("   This may take a while for large models...")

    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True
    )
    print(f"✅ Model downloaded to {model_path}")
    return model_path


def download_dataset(dataset_id: str, dataset_dir: str, hf_token: str):
    """Download dataset from HuggingFace Hub to local directory."""
    print(f"📥 Downloading dataset from {dataset_id} to {dataset_dir}")

    dataset_path = snapshot_download(
        repo_id=dataset_id,
        repo_type="dataset",
        local_dir=dataset_dir,
        local_dir_use_symlinks=False,
        token=hf_token,
        resume_download=True
    )
    print(f"✅ Dataset downloaded to {dataset_path}")
    return dataset_path


def load_model_and_tokenizer(model_id: str, model_dir: str, hf_token: str):
    """Load MedGemma model and tokenizer from local directory or HuggingFace Hub."""
    # Check if model exists locally
    if Path(model_dir).exists() and list(Path(model_dir).glob("config.json")):
        print(f"🔄 Loading model from local directory: {model_dir}")
        load_path = model_dir
    else:
        print(f"🔄 Loading model from HuggingFace Hub: {model_id}")
        load_path = model_id

    print("🔄 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(load_path, token=hf_token)

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🔄 Loading model (this takes 1-2 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )
    print("✅ Model loaded successfully!")
    return model, tokenizer


def load_dataset(args):
    """Load training and validation datasets.

    Supports both ReXGradient-160K from HuggingFace and custom CSV files.
    """
    if args.data_path:
        return load_csv_dataset(args)
    else:
        return load_hf_dataset(args)


def load_hf_dataset(args):
    """Load ReXGradient-160K dataset from HuggingFace."""
    from datasets import load_from_disk, load_dataset

    # Try loading from local directory first
    if Path(args.dataset_dir).exists():
        print(f"📂 Loading dataset from local directory: {args.dataset_dir}")
        try:
            dataset = load_from_disk(args.dataset_dir)
            # Check if it has the expected splits
            if args.train_split in dataset:
                train_dataset = dataset[args.train_split]
            else:
                # If no splits, use full dataset
                train_dataset = dataset

            if args.val_split in dataset:
                val_dataset = dataset[args.val_split]
            else:
                # Create validation split from training data
                print("⚠️  No validation split found, using 10% of training data")
                split_dataset = train_dataset.train_test_split(test_size=0.1, seed=args.seed)
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["test"]
        except Exception as e:
            print(f"⚠️  Failed to load from local: {e}")
            print("📂 Loading from HuggingFace Hub...")
            dataset = load_dataset(args.dataset_id, token=args.hf_token)
            train_dataset = dataset[args.train_split]
            val_dataset = dataset[args.val_split]
    else:
        print(f"📂 Loading dataset from HuggingFace Hub: {args.dataset_id}")
        dataset = load_dataset(args.dataset_id, token=args.hf_token)
        train_dataset = dataset[args.train_split]
        val_dataset = dataset[args.val_split]

    # Check for findings/impression columns (case-insensitive)
    columns_lower = {col.lower(): col for col in train_dataset.column_names}
    findings_col = columns_lower.get("findings")
    impression_col = columns_lower.get("impression")

    if findings_col and impression_col:
        print(f"   Found '{findings_col}' and '{impression_col}' columns")
        # Rename to standard lowercase names
        train_dataset = train_dataset.rename_column(findings_col, "findings")
        train_dataset = train_dataset.rename_column(impression_col, "impression")
        val_dataset = val_dataset.rename_column(findings_col, "findings")
        val_dataset = val_dataset.rename_column(impression_col, "impression")
    elif "metadata" in train_dataset.column_names:
        print(f"   Found 'metadata' column, extracting findings and impression...")
        train_dataset = extract_from_metadata(train_dataset)
        val_dataset = extract_from_metadata(val_dataset)
    else:
        raise ValueError(f"Expected 'findings'/'impression' or 'metadata' columns, got: {train_dataset.column_names}")

    # Limit samples if specified
    if args.train_size:
        train_dataset = train_dataset.select(range(min(args.train_size, len(train_dataset))))
    if args.val_size:
        val_dataset = val_dataset.select(range(min(args.val_size, len(val_dataset))))

    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def extract_from_metadata(dataset):
    """Extract findings and impression from metadata column."""
    findings = []
    impressions = []

    for item in dataset:
        metadata = item.get("metadata", {})
        findings.append(metadata.get("findings", ""))
        impressions.append(metadata.get("impression", ""))

    # Create new dataset with extracted columns
    new_data = {
        "findings": findings,
        "impression": impressions
    }

    return HFDataset.from_dict(new_data)


def load_csv_dataset(args):
    """Load dataset from custom CSV file."""
    print(f"📂 Loading CSV dataset from {args.data_path}")

    df = pd.read_csv(args.data_path)
    print(f"   Total rows: {len(df)}")

    # Validate columns
    if args.findings_col not in df.columns:
        raise ValueError(f"Findings column '{args.findings_col}' not found in CSV. Available: {df.columns.tolist()}")
    if args.impression_col not in df.columns:
        raise ValueError(f"Impression column '{args.impression_col}' not found in CSV. Available: {df.columns.tolist()}")

    # Remove rows with missing values
    df = df.dropna(subset=[args.findings_col, args.impression_col])
    print(f"   Valid rows: {len(df)}")

    # Split into train/validation (90/10)
    train_df = df.sample(frac=0.9, random_state=args.seed)
    val_df = df.drop(train_df.index)

    # Limit samples if specified
    if args.train_size:
        train_df = train_df.head(args.train_size)
    if args.val_size:
        val_df = val_df.head(args.val_size)

    print(f"   Training samples: {len(train_df)}")
    print(f"   Validation samples: {len(val_df)}")

    train_dataset = HFDataset.from_pandas(train_df)
    val_dataset = HFDataset.from_pandas(val_df)

    # Rename columns to standard names
    train_dataset = train_dataset.rename_column(args.findings_col, "findings")
    train_dataset = train_dataset.rename_column(args.impression_col, "impression")
    val_dataset = val_dataset.rename_column(args.findings_col, "findings")
    val_dataset = val_dataset.rename_column(args.impression_col, "impression")

    return train_dataset, val_dataset


def prepare_lora(model, args):
    """Apply LoRA configuration to the model."""
    print(f"🔧 Applying LoRA configuration...")
    print(f"   Rank (r): {args.lora_r}")
    print(f"   Alpha: {args.lora_alpha}")
    print(f"   Dropout: {args.lora_dropout}")
    print(f"   Target modules: {args.target_modules}")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, lora_config


def format_prompt(findings: str, impression: str = None) -> str:
    """Format findings and impression into training prompt."""
    user_message = f"""You are a board-certified radiologist.
Given the "Findings" section of a chest X-ray report, write a concise, professional "Impression" section.
The Impression should:
- List the most important diagnoses first
- Use standardized terminology
- Be suitable for a referring physician

Findings:
{findings}"""

    if impression:
        assistant_message = impression
        return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n{assistant_message}<end_of_turn>"
    else:
        return f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize the dataset."""
    prompts = []
    for findings, impression in zip(examples["findings"], examples["impression"]):
        prompt = format_prompt(findings, impression)
        prompts.append(prompt)

    # Tokenize with padding to max_length to avoid batching issues
    tokenized = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",  # Pad to max_length to ensure uniform length
        return_tensors=None
    )

    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()

    # Add token_type_ids (required for Gemma3 training)
    if "token_type_ids" not in tokenized:
        # Create zero token_type_ids (single sequence)
        tokenized["token_type_ids"] = [[0] * len(ids) for ids in tokenized["input_ids"]]

    return tokenized


def prepare_datasets(train_dataset, val_dataset, tokenizer, max_length):
    """Prepare datasets for training."""
    print("🔄 Tokenizing datasets...")

    train_tokenized = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )

    val_tokenized = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer, max_length),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Tokenizing validation data"
    )

    print(f"   Training samples: {len(train_tokenized)}")
    print(f"   Validation samples: {len(val_tokenized)}")

    return train_tokenized, val_tokenized


def setup_mlflow(args, lora_config):
    """Setup MLFlow logging."""
    if args.no_mlflow:
        return None

    try:
        mlflow.set_experiment(args.mlflow_experiment)

        # Log parameters
        params = {
            "model_id": args.model_id,
            "dataset_id": args.dataset_id,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "max_length": args.max_length,
            "warmup_ratio": args.warmup_ratio,
            "seed": args.seed,
        }

        mlflow.log_params(params)
        print(f"✅ MLFlow logging enabled: {args.mlflow_experiment}")
        return mlflow

    except Exception as e:
        print(f"⚠️  Failed to setup MLFlow: {e}")
        print("   Continuing without MLFlow logging...")
        return None


class MetricsCallback:
    """Custom callback for computing metrics during training."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            logs = {}
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])


def save_model(model, tokenizer, output_dir: str):
    """Save the fine-tuned model."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"💾 Saving model to {output_dir}")

    # Save LoRA adapters
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Model saved to {output_dir}")


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Get HF token from args or environment
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    print("="*60)
    print("MedGemma Fine-tuning with LoRA")
    print("="*60)

    # Download model if requested
    if args.download_model:
        download_model(args.model_id, args.model_dir, hf_token)

    # Download dataset if requested
    if args.download_dataset and not args.data_path:
        download_dataset(args.dataset_id, args.dataset_dir, hf_token)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.model_dir, hf_token)

    # Load datasets
    train_dataset, val_dataset = load_dataset(args)

    # Apply LoRA
    model, lora_config = prepare_lora(model, args)

    # Prepare datasets for training
    train_tokenized, val_tokenized = prepare_datasets(
        train_dataset, val_dataset, tokenizer, args.max_length
    )

    # Setup MLFlow
    mlflow_client = setup_mlflow(args, lora_config)

    # Use default data collator since we're padding to max_length
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["mlflow"] if not args.no_mlflow else [],
        seed=args.seed,
        dataloader_pin_memory=False,
    )

    # Create metrics callback
    metrics_callback = MetricsCallback()

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        data_collator=data_collator,
        callbacks=[metrics_callback] if not args.no_mlflow else [],
    )

    # Train
    print("\n🚀 Starting training...")
    print("="*60)

    train_result = trainer.train()

    print("\n" + "="*60)
    print("✅ Training complete!")
    print("="*60)

    # Print training metrics
    print(f"\n📊 Training Metrics:")
    print(f"   Final Training Loss: {train_result.training_loss:.4f}")
    print(f"   Total Steps: {train_result.global_step}")

    # Final evaluation
    print("\n📊 Running final evaluation...")
    eval_result = trainer.evaluate()
    print(f"   Final Validation Loss: {eval_result['eval_loss']:.4f}")

    # Log to MLFlow
    if mlflow_client and not args.no_mlflow:
        mlflow.log_metrics({
            "final_train_loss": train_result.training_loss,
            "final_eval_loss": eval_result['eval_loss'],
            "total_steps": train_result.global_step,
        })
        mlflow.end_run()

    # Save model
    save_model(model, tokenizer, args.output_dir)

    print("\n✅ All done!")


if __name__ == "__main__":
    main()
