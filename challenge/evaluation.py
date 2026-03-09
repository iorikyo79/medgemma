#!/usr/bin/env python3
"""
MedGemma Radiology Report Evaluation Script

Evaluates MedGemma-4b-it on generating radiology impressions from findings.
Uses ROUGE and BERTScore metrics.

Usage:
    python evaluation.py --sample_size 50 --hf_token YOUR_TOKEN
"""

import os
import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MedGemma on radiology report generation")
    parser.add_argument(
        "--data-path",
        type=str,
        default="/kaggle/input/datasets/hamidmomand/radiological-report/Cleaned_Diagnostic_Data.csv",
        help="Path to the CSV file containing radiology reports"
    )
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
        help="HuggingFace dataset ID (for downloading)"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./datasets/ReXGradient-160K",
        help="Local directory to download/load the dataset"
    )
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
        "--sample-size",
        type=int,
        default=50,
        help="Number of reports to evaluate (default: 50)"
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate for each impression"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
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
    """Load MedGemma model and tokenizer from local directory or HuggingFace Hub.

    Supports loading LoRA adapters if adapter_config.json is present in model_dir.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    # Check if this is a LoRA adapter directory
    is_lora_adapter = Path(model_dir).exists() and list(Path(model_dir).glob("adapter_config.json"))

    if is_lora_adapter:
        print(f"🔄 Loading LoRA adapter from: {model_dir}")
        print(f"📌 Base model will be loaded from: {model_id}")

        # Load tokenizer from adapter directory
        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir, token=hf_token)

        # Load base model
        print(f"🔄 Loading base model from {model_id}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )

        # Load LoRA adapter
        print("🔄 Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()  # Merge adapter weights into base model
        print("✅ LoRA adapter loaded and merged successfully!")
    else:
        # Check if model exists locally
        if Path(model_dir).exists() and list(Path(model_dir).glob("config.json")):
            print(f"🔄 Loading model from local directory: {model_dir}")
            load_path = model_dir
        else:
            print(f"🔄 Loading model from HuggingFace Hub: {model_id}")
            load_path = model_id

        print("🔄 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(load_path, token=hf_token)

        print("🔄 Loading model (this takes 1-2 minutes)...")
        model = AutoModelForCausalLM.from_pretrained(
            load_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token
        )
        print("✅ Model loaded successfully!")

    return model, tokenizer


def load_dataset(data_path: str, sample_size: int, seed: int):
    """Load and sample the radiology dataset."""
    print(f"📂 Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    print(f"   Total reports: {len(df)}")

    # 'Details' 컬럼이 없으면 Indication + Comparison + Findings 를 합쳐 생성
    if "Details" not in df.columns:
        print("   ⚙️  'Details' 컬럼이 없어 Indication + Comparison + Findings 를 합쳐 생성합니다.")

        def build_details(row):
            parts = []
            if pd.notna(row.get("Indication")) and str(row.get("Indication", "")).strip():
                parts.append(f"Indication: {row['Indication']}")
            if pd.notna(row.get("Comparison")) and str(row.get("Comparison", "")).strip():
                parts.append(f"Comparison: {row['Comparison']}")
            if pd.notna(row.get("Findings")) and str(row.get("Findings", "")).strip():
                parts.append(f"Findings: {row['Findings']}")
            return "\n".join(parts)

        df["Details"] = df.apply(build_details, axis=1)

    # Sample reports
    df_sample = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    print(f"   Sampled {len(df_sample)} reports")
    return df_sample


def generate_impression(model, tokenizer, findings_text: str, max_new_tokens: int = 128):
    """Generate an impression from findings using MedGemma."""
    messages = [
        {
            "role": "user",
            "content": f"""You are a board-certified radiologist.
Given the "Findings" section of a chest X-ray report, write a concise, professional "Impression" section.
The Impression should:
- List the most important diagnoses first
- Use standardized terminology
- Be suitable for a referring physician

Findings:
{findings_text}"""
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


def generate_predictions(model, tokenizer, df_sample, max_new_tokens: int):
    """Generate impressions for all samples."""
    predictions = []
    references = []
    findings_list = []

    print("🤖 Generating impressions...")
    for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample), desc="Progress"):
        findings = row["Details"]
        true_impression = row["Impression"]

        # Skip if empty
        if pd.isna(findings) or pd.isna(true_impression):
            continue

        generated = generate_impression(model, tokenizer, findings, max_new_tokens)
        predictions.append(generated)
        references.append(true_impression)
        findings_list.append(findings)

    print(f"✅ Generated {len(predictions)} impressions")
    return predictions, references, findings_list


def calculate_rouge_scores(predictions, references):
    """Calculate ROUGE scores."""
    from rouge_score import rouge_scorer

    print("📊 Calculating ROUGE scores...")
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    results = {
        'rouge1': {'mean': np.mean(rouge1_scores), 'std': np.std(rouge1_scores)},
        'rouge2': {'mean': np.mean(rouge2_scores), 'std': np.std(rouge2_scores)},
        'rougeL': {'mean': np.mean(rougeL_scores), 'std': np.std(rougeL_scores)},
    }
    return results, rouge1_scores, rouge2_scores, rougeL_scores


def calculate_bert_score(predictions, references):
    """Calculate BERTScore."""
    from bert_score import score as bert_score

    print("📊 Calculating BERTScore...")
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)

    return {
        'precision': P.mean().item(),
        'recall': R.mean().item(),
        'f1': F1.mean().item(),
        'f1_std': F1.std().item(),
    }, F1


def save_results(output_dir: str, findings_list, references, predictions,
                 rouge1_scores, rouge2_scores, rougeL_scores, bert_f1_scores):
    """Save evaluation results to CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame({
        "findings": findings_list,
        "true_impression": references,
        "generated_impression": predictions,
        "rouge1": rouge1_scores,
        "rouge2": rouge2_scores,
        "rougeL": rougeL_scores,
        "bertscore_f1": bert_f1_scores.tolist()
    })

    results_path = output_path / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"💾 Saved results to {results_path}")

    # Save top 5 examples for quick inspection
    top5_path = output_path / "top5_examples.csv"
    results_df.head(5).to_csv(top5_path, index=False)
    print(f"💾 Saved top 5 examples to {top5_path}")

    return results_df


def print_summary(rouge_results, bert_results):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)

    print("\n🔹 ROUGE Scores:")
    print(f"   ROUGE-1: {rouge_results['rouge1']['mean']:.3f} (±{rouge_results['rouge1']['std']:.3f})")
    print(f"   ROUGE-2: {rouge_results['rouge2']['mean']:.3f} (±{rouge_results['rouge2']['std']:.3f})")
    print(f"   ROUGE-L: {rouge_results['rougeL']['mean']:.3f} (±{rouge_results['rougeL']['std']:.3f})")

    print("\n🔹 BERTScore:")
    print(f"   Precision: {bert_results['precision']:.3f}")
    print(f"   Recall:    {bert_results['recall']:.3f}")
    print(f"   F1:        {bert_results['f1']:.3f} (±{bert_results['f1_std']:.3f})")
    print("="*60 + "\n")


def print_best_worst_examples(results_df, rougeL_scores):
    """Print best and worst performing examples."""
    worst_idx = int(np.argmin(rougeL_scores))
    best_idx = int(np.argmax(rougeL_scores))

    print("🔴 WORST PERFORMING EXAMPLE")
    print(f"ROUGE-L: {rougeL_scores[worst_idx]:.3f}")
    print(f"\n📝 Findings:\n{results_df.iloc[worst_idx]['findings'][:500]}...")
    print(f"\n🤖 Generated:\n{results_df.iloc[worst_idx]['generated_impression']}")
    print(f"\n✅ Ground Truth:\n{results_df.iloc[worst_idx]['true_impression']}")

    print("\n" + "-"*60 + "\n")

    print("🟢 BEST PERFORMING EXAMPLE")
    print(f"ROUGE-L: {rougeL_scores[best_idx]:.3f}")
    print(f"\n📝 Findings:\n{results_df.iloc[best_idx]['findings'][:500]}...")
    print(f"\n🤖 Generated:\n{results_df.iloc[best_idx]['generated_impression']}")
    print(f"\n✅ Ground Truth:\n{results_df.iloc[best_idx]['true_impression']}")


def main():
    args = parse_args()

    # Get HF token from args or environment (optional for local loading)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    print("="*60)
    print("MedGemma Radiology Report Evaluation")
    print("="*60)

    # Download model if requested
    if args.download_model:
        download_model(args.model_id, args.model_dir, hf_token)

    # Download dataset if requested
    if args.download_dataset:
        download_dataset(args.dataset_id, args.dataset_dir, hf_token)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_id, args.model_dir, hf_token)

    # Load dataset
    df_sample = load_dataset(args.data_path, args.sample_size, args.seed)

    # Generate predictions
    predictions, references, findings_list = generate_predictions(
        model, tokenizer, df_sample, args.max_new_tokens
    )

    # Calculate metrics
    rouge_results, rouge1_scores, rouge2_scores, rougeL_scores = calculate_rouge_scores(
        predictions, references
    )
    bert_results, bert_f1_scores = calculate_bert_score(predictions, references)

    # Print summary
    print_summary(rouge_results, bert_results)

    # Save results
    results_df = save_results(
        args.output_dir, findings_list, references, predictions,
        rouge1_scores, rouge2_scores, rougeL_scores, bert_f1_scores
    )

    # Print best/worst examples
    print_best_worst_examples(results_df, rougeL_scores)

    print("✅ Evaluation complete!")


if __name__ == "__main__":
    main()
