#!/usr/bin/env python
"""RL fine-tuning pipeline for GPT-2 using PPO on synthetic seller email data.

This script ports the experimental workflow from the `RL_FT_v1.ipynb` notebook into a
reusable, configurable Python module so it can be executed end-to-end from the
command line. It performs the following high-level steps:

1. Load and clean the synthetic seller email corpus (JSON list of strings).
2. Generate sales-email prompts from the dataset to drive PPO queries.
3. Instantiate GPT-2 with a value head plus the PPO trainer from TRL.
4. Define a rich reward function based on sentiment, structure, CTA detection, etc.
5. Run PPO updates to nudge GPT-2 toward higher-scoring sales outreach emails.
6. Persist the fine-tuned model and tokenizer for later inference.

Example usage (PowerShell on Windows):

```powershell
python rl_finetune_gpt2.py \
  --data-path ..\local_RL_FT\scripts\data\seller_emails_v3.json \
  --output-dir ..\local_RL_FT\outputs \
  --model-name gpt2 \
  --num-epochs 1 \
  --batch-size 2
```

The default hyperparameters mirror the notebook, but they can be overridden from
the CLI or by editing the dataclass defaults below.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

try:  # optional dependency for grammar checking
    from language_tool_python import LanguageTool  # type: ignore
except ImportError:  # pragma: no cover - optional tool
    LanguageTool = None  # type: ignore

import nltk
from nltk.corpus import stopwords

# --------------------------------------------------------------------------------------
# Logging utilities
# --------------------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    """Configure root logger formatting and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger("rl_finetune_gpt2")


# --------------------------------------------------------------------------------------
# Dataclasses for configuration
# --------------------------------------------------------------------------------------


def default_reward_weights() -> Dict[str, float]:
    return {
        "length": 1.2,
        "politeness": 1.2,
        "sentiment": 0.7,
        "clarity": 0.6,
        "cta": 1.4,
        "personalization": 0.7,
        "grammar": 0.8,
        "value": 1.1,
        "spam": 0.8,
        "structure": 0.8,
        "instructional_tone": 0.8,
        "lexical_coherence": 1.2,
    }


@dataclass
class PPOFineTuneConfig:
    """Container for hyperparameters and paths used by the fine-tuning script."""

    data_path: Path
    output_dir: Path
    cleaned_csv_path: Optional[Path] = None
    model_name: str = "gpt2"
    tokenizer_name: Optional[str] = None
    save_steps: Optional[int] = None
    num_epochs: int = 4
    batch_size: int = 4
    max_new_tokens: int = 320
    top_p: float = 0.9
    temperature: float = 0.8
    learning_rate: float = 1e-5
    seed: int = 42
    max_length: int = 256
    reward_weights: Dict[str, float] = field(default_factory=default_reward_weights)
    device: Optional[str] = None  # auto-detected later
    save_model: bool = True

    def resolve(self) -> "PPOFineTuneConfig":
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name
        self.output_dir = self.output_dir.resolve()
        if self.cleaned_csv_path is None:
            self.cleaned_csv_path = self.output_dir / "seller_emails_clean.csv"
        self.cleaned_csv_path = self.cleaned_csv_path.resolve()
        self.data_path = self.data_path.resolve()
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.output_dir, exist_ok=True)
        return self


# --------------------------------------------------------------------------------------
# Reproducibility helpers
# --------------------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Data ingestion & prompt generation
# --------------------------------------------------------------------------------------


def load_seller_emails(file_path: Path) -> pd.DataFrame:
    """Load seller email corpus stored as JSON list of strings."""
    if not file_path.exists():
        raise FileNotFoundError(f"Seller email JSON not found: {file_path}")

    with file_path.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    if not isinstance(raw, list) or not all(isinstance(entry, str) for entry in raw):
        raise ValueError("Expected the JSON file to contain a list of strings (email bodies).")

    df = pd.DataFrame({"email_text": raw})
    # adding email length feature
    df["len_email_text"] = df["email_text"].str.len()
    # clean text by removing extra whitespace
    df["email_text"] = df["email_text"].str.replace(r"\s+", " ", regex=True).str.strip()
    # remove duplicates
    df = df.drop_duplicates(subset=["email_text"]).reset_index(drop=True)
    logger.info("Loaded %d seller emails from %s", len(df), file_path)
    return df


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("Saved cleaned dataset to %s", output_path)


# --------------------------------------------------------------------------------------
# Reward model utilities
# --------------------------------------------------------------------------------------


def ensure_nltk_resources() -> None:
    """Ensure required NLTK corpora are downloaded (stopwords & punkt)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:  # pragma: no cover
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:  # pragma: no cover
        logger.info("Downloading NLTK stopwords...")
        nltk.download("stopwords")


ensure_nltk_resources()
STOP_WORDS = set(stopwords.words("english"))


def load_sentiment_analyzer(use_gpu: bool = True):
    device = 0 if use_gpu and torch.cuda.is_available() else -1
    logger.info("Loading sentiment analyzer (%s)...", "GPU" if device == 0 else "CPU")
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )


def init_language_tool():
    if LanguageTool is None:
        logger.warning("language_tool_python not installed; grammar rewards disabled.")
        return None
    try:  # pragma: no cover - performs network access on first run
        tool = LanguageTool("en-US")
        logger.info("Initialized LanguageTool for grammar checking.")
        return tool
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to initialize LanguageTool: %s", exc)
        return None


CTA_PHRASES = [
    "call",
    "reply",
    "schedule",
    "meet",
    "connect",
    "reach out",
    "get in touch",
    "book a demo",
    "set up a meeting",
    "schedule a call",
    "contact us",
]

POLITE_TERMS = [
    "thank",
    "appreciate",
    "please",
    "hope",
    "kindly",
    "regards",
    "grateful",
    "welcome",
    "would you",
    "could you",
]

UNCLEAR_TERMS = [
    "utilize",
    "leverage",
    "synergy",
    "paradigm",
    "bandwidth",
    "ecosystem",
    "turnkey",
    "disruptive",
]

VALUE_TERMS_PATTERN = re.compile(
    r"\b(save|reduce|increase|boost|improve|growth|roi|cost|revenue|profit)\b",
    re.IGNORECASE,
)

SPAM_TERMS = [
    "free",
    "winner",
    "click here",
    "urgent",
    "act now",
    "limited time",
    "guarantee",
]

INSTRUCTIONAL_PATTERN = re.compile(
    r"(write (an|a) email|here'?s how|this guide|please follow)",
    re.IGNORECASE,
)


def has_cta_phrase(text: str) -> Tuple[bool, List[str]]:
    lower = text.lower()
    matches = []
    for phrase in CTA_PHRASES:
        if " " in phrase:
            if phrase in lower:
                matches.append(phrase)
        elif re.search(rf"\b{re.escape(phrase)}\b", lower):
            matches.append(phrase)
    return len(matches) > 0, matches


def evaluate_structure_and_intent(text: str) -> Tuple[float, str]:
    lower = text.lower()
    matched: List[str] = []
    greeting_match = re.findall(r"\b(hi|hello|dear)\b", lower)
    closing_match = re.findall(r"\b(regards|sincerely|thanks|best)\b", lower)
    product_match = re.findall(r"\b(product|solution|platform|offer|launch|introduc)\w*", lower)
    if greeting_match:
        matched.extend(greeting_match)
    if closing_match:
        matched.extend(closing_match)
    if product_match:
        matched.extend(product_match)
    has_greeting = bool(greeting_match)
    has_closing = bool(closing_match)
    has_product = bool(product_match)
    if has_greeting and has_closing and has_product:
        return 1.0, f"Proper structure detected: {matched}"
    if has_greeting or has_product:
        return 0.5, f"Partial email structure: {matched}"
    return -0.5, "Missing greeting/closing/product context"


def penalize_instructional_tone(text: str) -> Tuple[float, str]:
    matches = INSTRUCTIONAL_PATTERN.findall(text.lower())
    if matches:
        flat = [m[0] if isinstance(m, tuple) else m for m in matches]
        return -1.0, f"Instructional tone detected: {flat}"
    return 0.0, "No instructional tone detected"


def lexical_coherence(text: str) -> Tuple[float, str]:
    sentences = [s.strip().lower() for s in re.split(r"[.!?]", text) if s.strip()]
    if len(sentences) < 2:
        return 0.5, "Too short to assess coherence"
    overlaps: List[float] = []
    for idx in range(len(sentences) - 1):
        s1 = {w for w in re.findall(r"\b[a-z]+\b", sentences[idx]) if w not in STOP_WORDS}
        s2 = {w for w in re.findall(r"\b[a-z]+\b", sentences[idx + 1]) if w not in STOP_WORDS}
        if not s1 or not s2:
            continue
        overlap = len(s1 & s2) / len(s1 | s2)
        overlaps.append(overlap)
    if not overlaps:
        return 0.3, "Insufficient overlapping vocabulary"
    avg_overlap = float(np.mean(overlaps))
    if avg_overlap > 0.4:
        return 1.0, f"Good coherence (avg overlap={avg_overlap:.2f})"
    if avg_overlap > 0.2:
        return 0.5, f"Partial coherence (avg overlap={avg_overlap:.2f})"
    return -0.5, f"Low coherence (avg overlap={avg_overlap:.2f})"


def compute_reward(
    text: str,
    *,
    sentiment_analyzer=None,
    tool=None,
    weights: Optional[Dict[str, float]] = None,
    detailed: bool = False,
) -> Tuple[float, Dict[str, float]] | float:
    """Compute composite reward for a generated email."""
    text = (text or "").strip()
    lower = text.lower()
    w = weights or default_reward_weights()

    # Length
    length = len(text)
    if 100 <= length <= 300:
        length_r = 1.0
        length_reason = f"Ideal length ({length} chars)"
    elif 300 < length <= 450:
        length_r = 0.2
        length_reason = f"Slightly long ({length} chars)"
    elif length > 450:
        length_r = -0.7
        length_reason = f"Too long ({length} chars)"
    else:
        length_r = -0.7
        length_reason = f"Too short ({length} chars)"

    # Politeness
    polite_hits = [term for term in POLITE_TERMS if re.search(rf"\b{re.escape(term)}\b", lower)]
    polite_r = 0.5 * len(polite_hits)
    polite_reason = (
        f"Polite terms: {', '.join(polite_hits)}"
        if polite_hits
        else "No polite phrasing detected"
    )

    # Sentiment
    sentiment_r = 0.0
    sentiment_reason = "Sentiment analysis skipped"
    if sentiment_analyzer is not None:
        try:
            output = sentiment_analyzer(text[:512])
            if output and isinstance(output, list):
                label = output[0].get("label", "").upper()
                score = output[0].get("score", 0.0)
                if label.startswith("POS"):
                    sentiment_r = 1.0
                    sentiment_reason = f"Positive tone (score={score:.2f})"
                elif label.startswith("NEG"):
                    sentiment_r = -0.3
                    sentiment_reason = f"Negative tone (score={score:.2f})"
                else:
                    sentiment_reason = f"Neutral tone (score={score:.2f})"
        except Exception as exc:  # pragma: no cover
            sentiment_reason = f"Sentiment analysis failed: {exc}"

    # Clarity
    unclear_hits = [term for term in UNCLEAR_TERMS if re.search(rf"\b{term}\b", lower)]
    clarity_r = -0.3 * len(unclear_hits) if unclear_hits else 0.7
    clarity_reason = (
        f"Unclear buzzwords: {', '.join(unclear_hits)}"
        if unclear_hits
        else "Clear, accessible language"
    )

    # CTA
    has_cta, cta_hits = has_cta_phrase(text)
    cta_r = 1.0 if has_cta else -0.8
    cta_reason = (
        f"CTA detected: {', '.join(cta_hits)}"
        if has_cta
        else "Missing explicit call-to-action"
    )

    # Personalization
    personalization_terms = ["you", "your team", "your company", "dear", "hello"]
    personalized = any(term in lower for term in personalization_terms)
    personalization_r = 0.7 if personalized else -0.2
    personalization_reason = (
        "Personalized tone"
        if personalized
        else "No personalization markers"
    )

    # Grammar (optional)
    grammar_r = 0.0
    grammar_reason = "Grammar check skipped"
    if tool is not None:
        try:
            matches = tool.check(text)
            n_errors = len(matches)
            if n_errors == 0:
                grammar_r = 1.0
                grammar_reason = "No grammar issues"
            elif n_errors < 4:
                grammar_r = 0.5
                grammar_reason = f"Minor grammar issues ({n_errors})"
            else:
                grammar_r = -0.3
                grammar_reason = f"Significant grammar issues ({n_errors})"
        except Exception as exc:  # pragma: no cover
            grammar_reason = f"Grammar check failed: {exc}"

    # Value proposition
    value_hits = VALUE_TERMS_PATTERN.findall(text)
    value_r = min(len(value_hits) * 0.5, 1.0)
    value_reason = (
        f"Value props detected: {', '.join(set(map(str.lower, value_hits)))}"
        if value_hits
        else "No value proposition terms"
    )

    # Spam avoidance
    spam_hits = [term for term in SPAM_TERMS if re.search(rf"\b{term}\b", lower)]
    spam_r = -0.8 * len(spam_hits) if spam_hits else 0.5
    spam_reason = (
        f"Spammy terms present: {', '.join(spam_hits)}"
        if spam_hits
        else "No spam terms detected"
    )

    # Structure
    structure_r, structure_reason = evaluate_structure_and_intent(text)

    # Instructional tone penalty
    instruction_r, instruction_reason = penalize_instructional_tone(text)

    # Lexical coherence
    coherence_r, coherence_reason = lexical_coherence(text)

    total_reward = (
        w["length"] * length_r
        + w["politeness"] * polite_r
        + w["sentiment"] * sentiment_r
        + w["clarity"] * clarity_r
        + w["cta"] * cta_r
        + w["personalization"] * personalization_r
        + w["grammar"] * grammar_r
        + w["value"] * value_r
        + w["spam"] * spam_r
        + w["structure"] * structure_r
        + w["instructional_tone"] * instruction_r
        + w["lexical_coherence"] * coherence_r
    )
    total_reward = float(np.clip(total_reward, -4.0, 6.0))

    if not detailed:
        return total_reward

    breakdown = {
        "length": length_r,
        "politeness": polite_r,
        "sentiment": sentiment_r,
        "clarity": clarity_r,
        "cta": cta_r,
        "personalization": personalization_r,
        "grammar": grammar_r,
        "value": value_r,
        "spam": spam_r,
        "structure": structure_r,
        "instructional_tone": instruction_r,
        "lexical_coherence": coherence_r,
        "reasons": {
            "length": length_reason,
            "politeness": polite_reason,
            "sentiment": sentiment_reason,
            "clarity": clarity_reason,
            "cta": cta_reason,
            "personalization": personalization_reason,
            "grammar": grammar_reason,
            "value": value_reason,
            "spam": spam_reason,
            "structure": structure_reason,
            "instructional_tone": instruction_reason,
            "lexical_coherence": coherence_reason,
        },
    }
    return total_reward, breakdown


# --------------------------------------------------------------------------------------
# Prompt generation
# --------------------------------------------------------------------------------------

PROMPT_TEMPLATES = [
    "Write a concise, professional sales email introducing this idea: \"{snippet}\"",
    "Compose a friendly B2B sales email based on this concept: \"{snippet}\"",
    "Generate a product outreach email using this information: \"{snippet}\"",
    "Write an enterprise sales email centered on: \"{snippet}\"",
    "Craft a formal product introduction email about: \"{snippet}\"",
]


def generate_prompts_from_emails(df: pd.DataFrame, *, text_column: str = "email_text") -> List[str]:
    texts = df[text_column].dropna().tolist()
    prompts: List[str] = []
    for email in texts:
        email = email.strip()
        sentences = nltk.sent_tokenize(email)
        core = [
            s for s in sentences if not re.match(r"^(hi|hello|dear|regards|thank|best)", s.strip().lower())
        ]
        snippet = ""
        for sentence in core:
            if len(sentence.split()) >= 6:
                snippet = sentence.strip()
                break
        if not snippet:
            snippet = email[:120]
        template = random.choice(PROMPT_TEMPLATES)
        prompts.append(template.format(snippet=snippet))
    logger.info("Generated %d prompts from dataset", len(prompts))
    return prompts


# --------------------------------------------------------------------------------------
# Model preparation
# --------------------------------------------------------------------------------------


def load_model_and_tokenizer(config: PPOFineTuneConfig):
    logger.info("Loading tokenizer: %s", config.tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    tokenizer.padding_side = "left"
    print(f"Tokenizer pad token before adjustment: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad token to eos token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

    logger.info("Loading base model for PPO value head: %s", config.model_name)
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ppo_model = ppo_model.to(config.device)
    logger.info("Model loaded on %s", config.device)
    return ppo_model, tokenizer


def create_ppo_trainer(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    config: PPOFineTuneConfig,
) -> PPOTrainer:
    logger.info("Initialising PPO trainer")
    ppo_config = PPOConfig(
        model_name=None,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.batch_size,
        optimize_cuda_cache=True,
    )
    trainer = PPOTrainer(config=ppo_config, model=model, tokenizer=tokenizer)
    return trainer


# --------------------------------------------------------------------------------------
# PPO training loop
# --------------------------------------------------------------------------------------


def prepare_queries(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> List[torch.Tensor]:
    prompt_lengths = attention_mask.sum(dim=1)
    queries: List[torch.Tensor] = []
    for row, length in zip(input_ids, prompt_lengths):
        length_int = int(length.item())
        start_idx = row.size(0) - length_int  # account for left padding
        queries.append(row[start_idx:].clone())
    return queries


def slice_responses(
    generated_ids: torch.Tensor,
    prompt_token_count: int,
    prompt_lengths: Sequence[int],
) -> List[torch.Tensor]:
    responses: List[torch.Tensor] = []
    for idx in range(generated_ids.size(0)):
        start = prompt_token_count
        resp_ids = generated_ids[idx, start:]
        if resp_ids.numel() == 0:
            resp_ids = generated_ids.new_tensor([generated_ids[idx, -1].item()])
        responses.append(resp_ids)
    return responses


def run_ppo_training(
    config: PPOFineTuneConfig,
    model: AutoModelForCausalLMWithValueHead,
    tokenizer: AutoTokenizer,
    trainer: PPOTrainer,
    prompts: Sequence[str],
    sentiment_analyzer,
    grammar_tool,
) -> None:
    device = torch.device(config.device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    gen_kwargs = dict(
        max_new_tokens=config.max_new_tokens,
        pad_token_id=pad_token_id,
        do_sample=True,
        top_p=config.top_p,
        temperature=config.temperature,
    )

    logger.info("Starting PPO fine-tuning for %d epochs", config.num_epochs)
    for epoch in range(config.num_epochs):
        logger.info("Epoch %d/%d", epoch + 1, config.num_epochs)
        progress = tqdm(range(0, len(prompts), config.batch_size), desc=f"epoch {epoch+1}")
        for idx in progress:
            batch_prompts = list(prompts[idx : idx + config.batch_size])
            enc = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=config.max_length,
            )
            input_ids = enc.input_ids.to(device)
            attention_mask = enc.attention_mask.to(device)

            queries = prepare_queries(input_ids, attention_mask)

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            prompt_token_count = input_ids.size(1)
            prompt_lengths = attention_mask.sum(dim=1).tolist()
            responses = slice_responses(generated, prompt_token_count, prompt_lengths)
            response_texts = [
                tokenizer.decode(resp, skip_special_tokens=True) for resp in responses
            ]

            rewards: List[torch.Tensor] = []
            for text, resp_ids in zip(response_texts, responses):
                total_reward = compute_reward(
                    text,
                    sentiment_analyzer=sentiment_analyzer,
                    tool=grammar_tool,
                    weights=config.reward_weights,
                )
                reward_tensor = torch.full(
                    (resp_ids.size(0),),
                    float(total_reward),
                    device=device,
                    dtype=torch.float32,
                )
                rewards.append(reward_tensor)

            print({"query": batch_prompts, "response": response_texts})
            stats = trainer.step(queries, responses, rewards)

            rewards_for_log = torch.stack([reward.mean() for reward in rewards]).to(device)
            batch_log = {"query": batch_prompts, "response": response_texts}
            print({"query": batch_prompts, "response": response_texts})
            trainer.log_stats(stats, batch_log, rewards_for_log)

        logger.info("Completed epoch %d", epoch + 1)

    if config.save_model:
        save_dir = config.output_dir / "ppo_gpt2"
        save_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Saving fine-tuned model to %s", save_dir)
        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)


# --------------------------------------------------------------------------------------
# Argument parsing & CLI entrypoint
# --------------------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "seller_emails_v3.json",
        help="Path to JSON file containing a list of seller emails.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory to store cleaned dataset, logs, and model artifacts.",
    )
    parser.add_argument("--model-name", type=str, default="gpt2", help="Hugging Face model id for PPO training.")
    parser.add_argument("--tokenizer-name", type=str, default=None, help="Tokenizer id (defaults to model-name).")
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--no-save", action="store_true", help="Skip saving the fine-tuned model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser


# --------------------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=args.verbose)

    config = PPOFineTuneConfig(
        data_path=args.data_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_length=args.max_length,
        save_model=not args.no_save,
    ).resolve()

    logger.info("Configuration: %s", config)

    set_seed(config.seed)

    df = load_seller_emails(config.data_path)
    save_cleaned_dataset(df, config.cleaned_csv_path)

    prompts = generate_prompts_from_emails(df)

    model, tokenizer = load_model_and_tokenizer(config)
    trainer = create_ppo_trainer(model, tokenizer, config)

    sentiment_analyzer = load_sentiment_analyzer(use_gpu=config.device == "cuda")
    grammar_tool = init_language_tool()

    run_ppo_training(
        config,
        model,
        tokenizer,
        trainer,
        prompts,
        sentiment_analyzer,
        grammar_tool,
    )

    logger.info("RL fine-tuning complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
