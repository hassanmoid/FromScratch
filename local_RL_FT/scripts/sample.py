# ============================================
# 🧩 Step 5A — Reward Modeling Functions (Fully Explained)
# ============================================
from transformers import pipeline
import numpy as np
import torch
import re
import pandas as pd

def load_sentiment_analyzer(use_gpu=True):
    """
    Load sentiment analyzer (DistilBERT fine-tuned on SST-2).
    """
    device = 0 if use_gpu and torch.cuda.is_available() else -1
    print(f"🔍 Loading sentiment analyzer on {'GPU' if device == 0 else 'CPU'}...")
    analyzer = pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=device)
    return analyzer


def compute_reward(text: str, sentiment_analyzer=None, tool=None, weights=None, detailed=False):
    """
    Compute and explain the reward breakdown for a given email.
    Categories:
      1. Length Reward
      2. Politeness Reward
      3. Sentiment Reward
      4. Clarity
      5. CTA (Call-to-Action)
      6. Personalization
      7. Grammar
      8. Value Proposition
      9. Spam Avoidance
      10. Structure
    """
    txt = (text or "").strip()
    txt_lower = txt.lower()
    weights = weights or {"length": 1.0, "politeness": 1.0, "sentiment": 1.0}

    # --- 1️⃣ Length ---
    length = len(txt)
    if 100 <= length <= 300:
        length_r = 1.0
        length_reason = f"✅ Ideal length ({length} chars between 100–300)."
    elif length > 300:
        length_r = 0.5
        length_reason = f"⚠️ Slightly long ({length} chars > 300)."
    else:
        length_r = -0.5
        length_reason = f"⚠️ Too short ({length} chars < 100)."

    # --- 2️⃣ Politeness ---
    polite_terms = ["thank", "appreciate", "please", "hope", "kindly", "regards", "grateful", "welcome", "would you", "could you"]
    found_terms = [term for term in polite_terms if re.search(rf"\b{term}\b", txt_lower)]
    polite_r = 0.5 * len(found_terms)
    if len(found_terms) > 0:
        polite_reason = f"✅ Found polite terms: {', '.join(found_terms)} (+{polite_r:.1f})."
    else:
        polite_reason = "⚠️ No polite words detected."

    # --- 3️⃣ Sentiment ---
    sentiment_r = 0.0
    sentiment_reason = ""
    if sentiment_analyzer is not None:
        try:
            out = sentiment_analyzer(txt[:512])
            if out and isinstance(out, list) and "label" in out[0]:
                label = out[0]["label"].upper()
                score = out[0].get("score", 0)
                if label.startswith("POS"):
                    sentiment_r = 1.0
                    sentiment_reason = f"✅ Positive tone detected (score={score:.2f})."
                elif label.startswith("NEG"):
                    sentiment_r = -0.3
                    sentiment_reason = f"⚠️ Negative tone detected (score={score:.2f})."
                else:
                    sentiment_reason = f"😐 Neutral tone detected (score={score:.2f})."
        except Exception as e:
            sentiment_reason = f"⚠️ Sentiment analysis failed: {e}"
            sentiment_r = 0.0

    # --- 4️⃣ Clarity ---
    unclear_terms = ["utilize", "leverage", "synergy", "paradigm", "bandwidth", "ecosystem", "turnkey", "disruptive"]
    found_terms = [term for term in unclear_terms if re.search(rf"\b{term}\b", txt_lower)]
    clarity_r = -0.3 * len(unclear_terms) if unclear_terms else 0.7
    clarity_reason = f"⚠️ Found jargon: {', '.join(unclear_terms)}" if unclear_terms else "✅ Clear and simple language."

    # --- 5️⃣ CTA (Call-to-Action) ---
    cta_phrases = ["schedule a call", "book a demo", "let’s connect", "reply", "meet", "call", "let me know", "get started", "sign up", "try it now"]
    has_cta = any(p in txt_lower for p in cta_phrases)
    cta_r = 1.0 if has_cta else -0.2
    cta_reason = "✅ Contains clear call-to-action words like " + ", ".join([p for p in cta_phrases if p in txt_lower]) if has_cta else "⚠️ No clear call-to-action."

    # --- 6️⃣ Personalization ---
    personalization_terms = ["you", "your team", "your company", "dear", "hello"]
    personalized = any(t in txt_lower for t in personalization_terms)
    personalization_r = 0.7 if personalized else 0.0
    personalization_reason = "✅ Personalized tone with terms like " + ", ".join([t for t in personalization_terms if t in txt_lower]) if personalized else "⚠️ No personalization detected."

    # --- 7️⃣ Grammar ---
    grammar_r, grammar_reason = 0.0, ""
    if tool is not None:
        try:
            correction = tool.check(txt)
            n_errors = len(correction)
            if n_errors == 0:
                grammar_r = 1.0
                grammar_reason = "✅ No grammatical errors detected."
            elif n_errors < 3:
                grammar_r = 0.5
                grammar_reason = f"⚠️ Few grammatical errors detected ({n_errors} issues)."
            else:
                grammar_r = -0.3
                grammar_reason = f"❌ Many grammatical errors detected ({n_errors} issues)."
        except Exception as e:
            grammar_reason = f"⚠️ Grammar check failed: {e}"
            grammar_r = 0.0

    # --- 8️⃣ Value Proposition ---
    value_terms = re.findall(r"\b(save|reduce|increase|boost|improve|growth|roi|cost|revenue|profit)\b", txt, re.I)
    value_r = min(len(value_terms) * 0.3, 1.0)
    value_reason = f"✅ Value terms found: {', '.join(set(value_terms))}." if value_terms else "⚠️ No value proposition terms."

    # --- 9️⃣ Spam Avoidance ---
    spam_terms = ["free", "winner", "click here", "urgent", "act now", "limited time", "guarantee"]
    found_spam = [term for term in spam_terms if re.search(rf"\b{term}\b", txt_lower)]
    spam_r = -0.5 * len(found_spam) if found_spam else 0.5
    spam_reason = f"❌ Spammy terms found: {', '.join(found_spam)}." if found_spam else "✅ No spammy terms detected."

    # --- 🔟 Structure ---
    has_greeting = bool(re.search(r"\b(dear|hi|hello|greetings|to whom it may concern)\b", txt_lower))
    has_closing = bool(re.search(r"(regards|sincerely|best)", txt_lower))
    if has_greeting and has_closing:
        structure_r, structure_reason = 1.0, "✅ Proper greeting and closing."
    elif has_greeting or has_closing:
        structure_r, structure_reason = 0.3, "⚠️ Missing either greeting or closing."
    else:
        structure_r, structure_reason = -0.2, "⚠️ Missing both greeting and closing."

    # --- Weighted total ---
    total_reward = (
        weights["length"] * length_r
        + weights["politeness"] * polite_r
        + weights["sentiment"] * sentiment_r
        + weights["clarity"] * clarity_r
        + weights["cta"] * cta_r
        + weights["personalization"] * personalization_r
        + weights["grammar"] * grammar_r
        + weights["value"] * value_r
        + weights["spam"] * spam_r
        + weights["structure"] * structure_r
    )
    
    total_reward = float(np.clip(total_reward, -3.0, 5.0))

    if detailed:
        return total_reward, {
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
                "structure": structure_reason
            }
        }
    return total_reward


def test_reward_function(df, reward_fn, n_samples=3, **kwargs):
    """
    Evaluate reward on sample seller emails with complete reasoning.
    """
    print("\n🧮 Testing Reward Function with Explanations:\n")
    sample_emails = df["email_text"].sample(n_samples, random_state=42)

    weights = kwargs.get("weights", {"length": 1.0, "politeness": 1.0, "sentiment": 1.0})

    for i, email in enumerate(sample_emails, 1):
        total, breakdown = reward_fn(email, detailed=True, **kwargs)
        reasons = breakdown["reasons"]

        print(f"📧 Email #{i}")
        print(f"Excerpt: {email[:180]}...\n")

        print("🧾 Reward Components & Explanations:")
        print(f"  ├─ Length Reward:     {breakdown['length']:+.2f} - {reasons['length']}")
        print(f"  ├─ Politeness Reward: {breakdown['politeness']:+.2f} - {reasons['politeness']}")
        print(f"  └─ Sentiment Reward:  {breakdown['sentiment']:+.2f} - {reasons['sentiment']}")
        print(f"  ├─ Clarity Reward:    {breakdown['clarity']:+.2f} - {reasons['clarity']}")
        print(f"  ├─ CTA Reward:        {breakdown['cta']:+.2f} - {reasons['cta']}")
        print(f"  ├─ Personalization:   {breakdown['personalization']:+.2f} - {reasons['personalization']}")
        print(f"  ├─ Grammar Reward:    {breakdown['grammar']:+.2f} - {reasons['grammar']}")
        print(f"  ├─ Value Prop Reward: {breakdown['value']:+.2f} - {reasons['value']}")
        print(f"  ├─ Spam Avoidance:    {breakdown['spam']:+.2f} - {reasons['spam']}")
        print(f"  └─ Structure Reward:  {breakdown['structure']:+.2f} - {reasons['structure']}")
        print(f"  • Weights Used:      {weights}")

        # Explicit calculation formula
        calc_str = (
            f"({weights['length']}×{breakdown['length']:.2f}) + "
            f"({weights['politeness']}×{breakdown['politeness']:.2f}) + "
            f"({weights['sentiment']}×{breakdown['sentiment']:.2f}) + "
            f"({weights['clarity']}×{breakdown['clarity']:.2f}) + "
            f"({weights['cta']}×{breakdown['cta']:.2f}) + "
            f"({weights['personalization']}×{breakdown['personalization']:.2f}) + "
            f"({weights['grammar']}×{breakdown['grammar']:.2f}) + "
            f"({weights['value']}×{breakdown['value']:.2f}) + "
            f"({weights['spam']}×{breakdown['spam']:.2f}) + "
            f"({weights['structure']}×{breakdown['structure']:.2f})"
        )
        intermediate_sum = (
            weights["length"] * breakdown["length"]
            + weights["politeness"] * breakdown["politeness"]
            + weights["sentiment"] * breakdown["sentiment"]
            + weights["clarity"] * breakdown["clarity"]
            + weights["cta"] * breakdown["cta"]
            + weights["personalization"] * breakdown["personalization"]
            + weights["grammar"] * breakdown["grammar"]
            + weights["value"] * breakdown["value"]
            + weights["spam"] * breakdown["spam"]
            + weights["structure"] * breakdown["structure"]
        )

        print("\n🧮 Calculation:")
        print(f"  = {calc_str}")
        print(f"  = {intermediate_sum:.2f}")
        print(f"  → Final Clipped Reward: {total:.2f}")
        print("-" * 70)


# ============================================
# ⚙️ Step 5B — Run Reward Evaluation
# ============================================

# ============================================
# ⚙️ Step 5B — Run Reward Evaluation (Extended)
# ============================================
from language_tool_python import LanguageTool

# ✅ Load sentiment model once (cached)
if "sentiment_analyzer" not in globals():
    sentiment_analyzer = load_sentiment_analyzer(use_gpu=True)

# ✅ Load LanguageTool only once (cached)
if "tool" not in globals():
    print("🧠 Initializing LanguageTool (cached once)...")
    tool = LanguageTool('en-US', config={"cache": True})

# Define reward weights
reward_weights = {
    "length": 1.0,
    "politeness": 1.2,
    "sentiment": 0.8,
    "clarity": 0.6,
    "cta": 1.2,
    "personalization": 0.8,
    "grammar": 1.0,
    "value": 1.0,
    "spam": 0.7,
    "structure": 0.6
}

df_seller = pd.DataFrame({
    "email_text": [
        "Hi there,\n\nI hope this message finds you well. I wanted to introduce you to our new software solution that can help streamline your business operations. Please let me know if you're interested in learning more.\n\nBest regards,\nJohn Doe",
        "Dear Sir/Madam,\n\nWe are pleased to offer you a special discount on our products. This is a limited-time offer, so act fast!\n\nSincerely,\nJane Smith",
        "Hello,\n\nOur team has developed an innovative platform that can significantly enhance your marketing efforts. I would love to schedule a demo at your convenience.\n\nThank you,\nEmily Johnson"
    ]
})
# Evaluate on sample emails (now with component explanations)
test_reward_function(
    df_seller,
    compute_reward,
    n_samples=5,
    sentiment_analyzer=sentiment_analyzer,
    weights=reward_weights
)
