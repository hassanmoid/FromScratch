"""
Test script for outcome-based RL fine-tuned model
"""

import os
import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"test_outcome_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================================
# CONFIGURATION
# ================================================================================

REWARD_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "reward_model")
GEN_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ppo_gpt2_outcome")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.95,
    "do_sample": True,
    "repetition_penalty": 1.1,
    "pad_token_id": 50256
}

# Test contexts
TEST_CONTEXTS = [
    "Customer: CTO at a Healthcare firm was referred by a partner for a telehealth app.",
    "Customer: marketing director at a Finance firm asked about pricing for a payment gateway.",
    "Customer: sales head at a E-commerce firm attended a webinar for a inventory sync system.",
    "Customer: founder at a Manufacturing firm cold prospect for a supply chain tool.",
    "Customer: operations manager at a Real Estate firm was referred by a partner for a CRM suite.",
    "Customer: CFO at a IT Services firm looking for solutions for a cloud migration service.",
    "Customer: procurement head at a Education firm attended a webinar for a learning management system.",
    "Customer: admin manager at a Travel firm requested demo for a booking automation platform.",
]

# ================================================================================
# MODEL LOADING
# ================================================================================

def load_models(reward_path: str, gen_path: str):
    """Load both reward model and generation model"""
    logger.info("="*80)
    logger.info("LOADING MODELS")
    logger.info("="*80)
    logger.info(f"Device: {DEVICE}")
    
    # Check if models exist
    if not os.path.exists(reward_path):
        logger.error(f"Reward model not found: {reward_path}")
        logger.error("Please train the model first using rl_ppo_outcome_based.py")
        return None, None, None, None
    
    if not os.path.exists(gen_path):
        logger.error(f"Generation model not found: {gen_path}")
        logger.error("Please train the model first using rl_ppo_outcome_based.py")
        return None, None, None, None
    
    # Load reward model
    logger.info("Loading reward model...")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_path)
    reward_model.to(DEVICE)
    reward_model.eval()
    logger.info("✓ Reward model loaded")
    
    # Load generation model
    logger.info("Loading generation model...")
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_path)
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model = AutoModelForCausalLM.from_pretrained(gen_path)
    gen_model.to(DEVICE)
    gen_model.eval()
    logger.info(f"✓ Generation model loaded. Parameters: {gen_model.num_parameters():,}")
    logger.info("="*80)
    
    return reward_model, reward_tokenizer, gen_model, gen_tokenizer

# ================================================================================
# EMAIL GENERATION & EVALUATION
# ================================================================================

def create_prompt(context: str) -> str:
    """Create formatted prompt"""
    return f"""Write a professional sales email based on the following context.

Context: {context}

Email:"""

def generate_email(model, tokenizer, context: str) -> str:
    """Generate email for given context"""
    prompt = create_prompt(context)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            **GENERATION_CONFIG
        )
    
    # Decode
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only email part
    if "Email:" in full_text:
        email = full_text.split("Email:")[-1].strip()
    else:
        email = full_text
    
    return email

def predict_outcome(reward_model, reward_tokenizer, text: str) -> tuple:
    """Predict outcome and get reward"""
    # Tokenize
    inputs = reward_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Predict
    with torch.no_grad():
        outputs = reward_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]
        predicted_class = torch.argmax(logits, dim=-1).item()
    
    # Map to reward
    outcome_to_reward = {0: -1.0, 1: 0.5, 2: 2.0}
    reward = outcome_to_reward[predicted_class]
    
    # Get probabilities for each outcome
    outcome_probs = {
        "no_reply": probs[0].item(),
        "reply": probs[1].item(),
        "conversion": probs[2].item()
    }
    
    return predicted_class, reward, outcome_probs

# ================================================================================
# TESTING
# ================================================================================

def run_tests(reward_model, reward_tokenizer, gen_model, gen_tokenizer, contexts: list):
    """Run tests on multiple contexts"""
    logger.info("="*80)
    logger.info("RUNNING TESTS")
    logger.info("="*80)
    logger.info(f"Number of test contexts: {len(contexts)}\n")
    
    results = []
    
    for idx, context in enumerate(contexts, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST {idx}/{len(contexts)}")
        logger.info(f"{'='*80}")
        logger.info(f"Context:\n{context}\n")
        
        # Generate email
        logger.info("Generating email...")
        email = generate_email(gen_model, gen_tokenizer, context)
        
        logger.info(f"\nGenerated Email:\n{email}\n")
        
        # Predict outcome
        predicted_outcome, reward, outcome_probs = predict_outcome(
            reward_model, reward_tokenizer, email
        )
        
        # Outcome mapping
        outcome_names = {0: "No Reply", 1: "Reply", 2: "Conversion"}
        
        logger.info(f"Predicted Outcome: {outcome_names[predicted_outcome]} (class {predicted_outcome})")
        logger.info(f"Reward: {reward:.2f}")
        logger.info(f"Outcome Probabilities:")
        logger.info(f"  - No Reply: {outcome_probs['no_reply']:.3f}")
        logger.info(f"  - Reply: {outcome_probs['reply']:.3f}")
        logger.info(f"  - Conversion: {outcome_probs['conversion']:.3f}")
        
        # Analysis
        word_count = len(email.split())
        has_question = '?' in email
        has_cta = any(word in email.lower() for word in ['call', 'demo', 'connect', 'schedule', 'discuss'])
        
        logger.info(f"\nAnalysis:")
        logger.info(f"  - Word count: {word_count}")
        logger.info(f"  - Contains question: {has_question}")
        logger.info(f"  - Has call-to-action: {has_cta}")
        
        results.append({
            'context': context,
            'email': email,
            'predicted_outcome': predicted_outcome,
            'reward': reward,
            'outcome_probs': outcome_probs,
            'word_count': word_count,
            'has_question': has_question,
            'has_cta': has_cta
        })
    
    return results

# ================================================================================
# SUMMARY
# ================================================================================

def print_summary(results: list):
    """Print summary statistics"""
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    total_tests = len(results)
    avg_reward = sum(r['reward'] for r in results) / total_tests
    avg_word_count = sum(r['word_count'] for r in results) / total_tests
    pct_with_questions = sum(r['has_question'] for r in results) / total_tests * 100
    pct_with_cta = sum(r['has_cta'] for r in results) / total_tests * 100
    
    # Outcome distribution
    outcome_counts = {0: 0, 1: 0, 2: 0}
    for r in results:
        outcome_counts[r['predicted_outcome']] += 1
    
    outcome_names = {0: "No Reply", 1: "Reply", 2: "Conversion"}
    
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Average reward: {avg_reward:.2f}")
    logger.info(f"Average word count: {avg_word_count:.1f}")
    logger.info(f"Emails with questions: {pct_with_questions:.1f}%")
    logger.info(f"Emails with CTA: {pct_with_cta:.1f}%")
    logger.info(f"\nPredicted Outcome Distribution:")
    for outcome_id, count in outcome_counts.items():
        pct = count / total_tests * 100
        logger.info(f"  - {outcome_names[outcome_id]}: {count} ({pct:.1f}%)")
    logger.info("="*80)

# ================================================================================
# INTERACTIVE MODE
# ================================================================================

def interactive_mode(reward_model, reward_tokenizer, gen_model, gen_tokenizer):
    """Interactive mode for testing custom contexts"""
    logger.info("\n" + "="*80)
    logger.info("INTERACTIVE MODE")
    logger.info("="*80)
    logger.info("Enter custom contexts to generate emails. Type 'quit' to exit.\n")
    
    while True:
        context = input("\nEnter context (or 'quit' to exit): ").strip()
        
        if context.lower() in ['quit', 'exit', 'q']:
            logger.info("Exiting interactive mode.")
            break
        
        if not context:
            logger.info("Please enter a valid context.")
            continue
        
        logger.info(f"\nGenerating email for context: {context}")
        email = generate_email(gen_model, gen_tokenizer, context)
        
        predicted_outcome, reward, outcome_probs = predict_outcome(
            reward_model, reward_tokenizer, email
        )
        
        outcome_names = {0: "No Reply", 1: "Reply", 2: "Conversion"}
        
        print("\n" + "="*80)
        print("GENERATED EMAIL:")
        print("="*80)
        print(email)
        print("="*80)
        print(f"Predicted Outcome: {outcome_names[predicted_outcome]} (Reward: {reward:.2f})")
        print(f"Probabilities - No Reply: {outcome_probs['no_reply']:.3f}, "
              f"Reply: {outcome_probs['reply']:.3f}, "
              f"Conversion: {outcome_probs['conversion']:.3f}")
        print("="*80)

# ================================================================================
# MAIN
# ================================================================================

def main():
    """Main execution"""
    logger.info("="*80)
    logger.info("EMAIL GENERATION MODEL - OUTCOME-BASED TEST")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")
    
    # Load models
    reward_model, reward_tokenizer, gen_model, gen_tokenizer = load_models(
        REWARD_MODEL_PATH,
        GEN_MODEL_PATH
    )
    
    if reward_model is None or gen_model is None:
        logger.error("Failed to load models. Exiting.")
        return
    
    # Run predefined tests
    results = run_tests(reward_model, reward_tokenizer, gen_model, gen_tokenizer, TEST_CONTEXTS)
    
    # Print summary
    print_summary(results)
    
    # Ask user if they want interactive mode
    print("\n" + "="*80)
    response = input("Would you like to try interactive mode? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        interactive_mode(reward_model, reward_tokenizer, gen_model, gen_tokenizer)
    
    logger.info("\n" + "="*80)
    logger.info("Testing completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
