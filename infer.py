#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinGym-style zero-shot inference with ProtT5 (Hugging Face).

This script reads one or more CSV files containing deep mutational scanning (DMS)
variants and evaluates how well ProtT5 zero-shot scores correlate with the
experimental DMS measurements.

Expected input columns (per row / variant):
  - `mutant`: mutation string like "A123G" (WT AA, 1-based position, mutant AA).
  - `mutated_sequence`: the full *mutant* protein sequence.
  - `DMS_score`: the experimental fitness/score for this variant.

Zero-shot score (masked marginal, log-odds):
  Δ = log P(mutant_aa | context) - log P(wildtype_aa | context)

For ProtT5 (a seq2seq model), we score P(aa|context) by:
  1) Encoder input: WT sequence with the mutated position replaced by `<extra_id_0>`.
  2) Decoder target prefix: `<extra_id_0> {AA} <extra_id_1>`.
  3) Read the decoder logits at the `{AA}` step and compute log-softmax.

Outputs:
  - For each input CSV: a new CSV with a `prott5_delta_logp` column.
  - A `summary.csv` aggregating per-file Spearman statistics.
"""

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import T5ForConditionalGeneration, T5Tokenizer

try:
    from importlib.metadata import distributions
except Exception:  # pragma: no cover

    def distributions():  # type: ignore[override]
        return []


AA20 = "ACDEFGHIKLMNPQRSTVWY"
REQUIRED_COLS = {"mutant", "mutated_sequence", "DMS_score"}


def compute_spearman(pred_scores, true_scores) -> tuple[float | None, float | None]:
    rho, pval = spearmanr(pred_scores, true_scores, nan_policy="omit")
    rho_val = None if rho is None or (isinstance(rho, float) and math.isnan(rho)) else float(rho)
    pval_val = None if pval is None or (isinstance(pval, float) and math.isnan(pval)) else float(pval)
    return rho_val, pval_val


def _fmt_float(x: float | None, *, fmt: str) -> str:
    return "nan" if x is None else format(x, fmt)


def collect_installed_packages() -> list[str]:
    items: list[str] = []
    for dist in distributions():
        name = None
        try:
            name = dist.metadata.get("Name")
        except Exception:
            name = None
        if not name:
            continue
        items.append(f"{name}=={dist.version}")
    return sorted(set(items), key=str.lower)


def print_runtime_environment() -> None:
    print("========== Runtime ==========")
    print(f"Python:        {sys.version.replace(os.linesep, ' ')}")
    print(f"Executable:    {sys.executable}")
    print(f"Platform:      {sys.platform}")
    print("Packages:")
    for item in collect_installed_packages():
        print(f"  - {item}")
    print("=============================\n")


def parse_mutant(mut_str: str) -> tuple[str, int, str]:
    wt_aa = mut_str[0]
    mut_aa = mut_str[-1]
    pos1 = int(mut_str[1:-1])
    return wt_aa, pos1, mut_aa


def recover_wt_sequence(mut_seq: str, wt_aa: str, pos1: int) -> str:
    return mut_seq[: pos1 - 1] + wt_aa + mut_seq[pos1:]


def resolve_csv_paths(*, data_dir: Path, csv: str | None) -> list[Path]:
    if csv is None:
        return sorted(p for p in data_dir.glob("*.csv") if p.is_file())

    candidate = Path(csv)
    if not candidate.is_absolute():
        candidate = (data_dir / candidate).resolve()

    if not candidate.exists():
        raise FileNotFoundError(f"CSV not found: {candidate}")

    return [candidate]


def load_dataset(*, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path.name}: {sorted(missing)}")
    return df


def _normalize_sequence(sequence: str) -> str:
    sequence = str(sequence).strip().upper().replace(" ", "")
    sequence = re.sub(r"[UZOB]", "X", sequence)
    if not sequence:
        raise ValueError("Empty sequence.")
    return sequence


def _spaced_tokens(tokens: list[str]) -> str:
    return " ".join(tokens)


def _encode_aa20_token_ids(tokenizer: T5Tokenizer) -> tuple[torch.Tensor, dict[str, int]]:
    token_ids: list[int] = []
    aa_to_col: dict[str, int] = {}
    for i, aa in enumerate(AA20):
        ids = tokenizer(aa, add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Tokenizer does not map amino acid {aa!r} to a single token: {ids}")
        token_ids.append(ids[0])
        aa_to_col[aa] = i
    return torch.tensor(token_ids, dtype=torch.long), aa_to_col


def _device_and_dtype() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def load_model(*, model_name: str) -> tuple[T5ForConditionalGeneration, T5Tokenizer, str]:
    device, dtype = _device_and_dtype()
    print(f"Loading ProtT5 model {model_name!r} from Hugging Face (device={device})")
    tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype).eval().to(device)
    return model, tokenizer, device


def iter_batches(items: list[tuple], batch_size: int) -> Iterable[list[tuple]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def preprocess_dataset(*, df: pd.DataFrame) -> tuple[list[tuple[int, str, str, int, str]], list[float]]:
    prepared: list[tuple[int, str, str, int, str]] = []
    true_scores: list[float] = []

    for row_idx, row in enumerate(df.itertuples(index=False)):
        mut_str = str(getattr(row, "mutant"))
        mut_seq = _normalize_sequence(getattr(row, "mutated_sequence"))
        score = float(getattr(row, "DMS_score"))

        wt_aa, pos1, mut_aa = parse_mutant(mut_str)

        if pos1 < 1 or pos1 > len(mut_seq):
            raise ValueError(f"Row {row_idx}: pos1 out of range ({pos1}) for sequence length {len(mut_seq)}")

        wt_seq = recover_wt_sequence(mut_seq, wt_aa, pos1)
        wt_seq = _normalize_sequence(wt_seq)

        tokens = list(wt_seq)
        tokens[pos1 - 1] = "<extra_id_0>"
        masked_text = _spaced_tokens(tokens)

        prepared.append((row_idx, masked_text, wt_aa, pos1, mut_aa))
        true_scores.append(score)

    return prepared, true_scores


def score_delta_logp(
    *,
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    device: str,
    batch_size: int,
    prepared: list[tuple[int, str, str, int, str]],
    progress_every: int,
) -> list[float]:
    aa20_token_ids, aa_to_col = _encode_aa20_token_ids(tokenizer)
    aa20_token_ids = aa20_token_ids.to(device)

    placeholder = "A"
    labels_text = f"<extra_id_0> {placeholder} <extra_id_1>"
    label_enc = tokenizer(labels_text, add_special_tokens=True, return_tensors="pt")
    labels_single = label_enc.input_ids.to(device)

    placeholder_ids = tokenizer(placeholder, add_special_tokens=False).input_ids
    if len(placeholder_ids) != 1:
        raise ValueError(f"Unexpected placeholder tokenization for {placeholder!r}: {placeholder_ids}")
    placeholder_id = placeholder_ids[0]
    labels_list = labels_single[0].tolist()
    if placeholder_id not in labels_list:
        raise ValueError(f"Could not locate placeholder id {placeholder_id} in labels {labels_list}")
    aa_step_index = labels_list.index(placeholder_id)

    preds: list[float] = [float("nan")] * len(prepared)
    total = len(prepared)
    processed = 0

    with torch.no_grad():
        for batch in iter_batches(prepared, batch_size):
            batch_texts = [masked_text for (_, masked_text, _, _, _) in batch]
            enc = tokenizer(batch_texts, add_special_tokens=True, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            labels = labels_single.expand(len(batch), -1)
            outputs = model(**enc, labels=labels)

            logits = outputs.logits[:, aa_step_index, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            aa20_log_probs = log_probs.index_select(dim=-1, index=aa20_token_ids)

            for j, (row_idx, _masked_text, wt_aa, _pos1, mut_aa) in enumerate(batch):
                if wt_aa not in aa_to_col or mut_aa not in aa_to_col:
                    preds[row_idx] = float("nan")
                else:
                    wt_col = aa_to_col[wt_aa]
                    mut_col = aa_to_col[mut_aa]
                    preds[row_idx] = float(aa20_log_probs[j, mut_col] - aa20_log_probs[j, wt_col])
                processed += 1

            if progress_every > 0 and (processed % progress_every == 0 or processed == total):
                print(f"  predicted {processed}/{total}")

    return preds


def create_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="ProtT5 zero-shot variant effect prediction via Δ = logP(mut|context) - logP(wt|context)."
    )
    p.add_argument(
        "--input_csv",
        default=None,
        help="Only process this CSV (basename under data_dir, or an absolute path). If omitted, process all CSVs in data_dir.",
    )
    p.add_argument("--data_dir", default="/opt/ml/processing/input/data")
    p.add_argument("--output_dir", default="/opt/ml/processing/output")
    p.add_argument("--output_suffix", default="_prott5_zeroshot.csv")
    p.add_argument("--progress_every", type=int, default=100, help="Print progress every N variants (0 disables).")

    p.add_argument("--model_name", default="Rostlab/prot_t5_xl_uniref50", help="Hugging Face model identifier.")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for model forward passes.")
    return p


def main() -> None:
    args = create_parser().parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_runtime_environment()

    csv_paths = resolve_csv_paths(data_dir=data_dir, csv=args.input_csv)
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {data_dir}")

    model, tokenizer, device = load_model(model_name=args.model_name)

    summaries: list[dict] = []

    for csv_path in csv_paths:
        df = load_dataset(csv_path=csv_path)
        prepared, true_scores = preprocess_dataset(df=df)

        pred_scores = score_delta_logp(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
            prepared=prepared,
            progress_every=args.progress_every,
        )

        rho, pval = compute_spearman(pred_scores, true_scores)
        df["prott5_delta_logp"] = pred_scores

        out_name = f"{csv_path.stem}{args.output_suffix}"
        out_path = output_dir / out_name
        df.to_csv(out_path, index=False)

        print("\n========== ProteinGym zero-shot ==========")
        print("Model:        ProtT5")
        print(f"HF repo:      {args.model_name}")
        print(f"CSV:          {csv_path.name}")
        print(f"Variants:     {len(df)}")
        print(f"Spearman ρ:   {_fmt_float(rho, fmt='.4f')}")
        print(f"P-value:      {_fmt_float(pval, fmt='.2e')}")
        print(f"Saved to:     {out_path}")
        print("==========================================\n")

        summaries.append(
            {
                "csv": csv_path.name,
                "variants": int(len(df)),
                "spearman_rho": rho,
                "p_value": pval,
                "output_csv": out_path.name,
            }
        )

    summary_path = output_dir / "summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
