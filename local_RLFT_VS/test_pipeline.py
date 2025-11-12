"""
Test script for the trained RL-PPO email generation model

This script loads the trained model and generates emails for various test contexts
"""

import os
import torch
import logging
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"test_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ================================================================================
# CONFIGURATION
# ================================================================================

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "ppo_gpt2_simple")
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

# Test contexts covering different scenarios
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

def load_model_and_tokenizer(model_path: str):
    """Load the trained model and tokenizer"""
    logger.info("="*80)
    logger.info("LOADING TRAINED MODEL")
    logger.info("="*80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Device: {DEVICE}")
    
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        logger.error("Please train the model first using rl_mistral_ppo_pipeline.py")
        return None, None
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("✓ Tokenizer loaded")
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    model.to(DEVICE)
    model.eval()
    logger.info(f"✓ Model loaded. Parameters: {model.num_parameters():,}")
    logger.info("="*80)
    
    return model, tokenizer

# ================================================================================
# EMAIL GENERATION
# ================================================================================

def create_prompt(context: str) -> str:
    """Create a formatted prompt for the model"""
    prompt = f"""Write a professional sales email based on the following context.

Context: {context}

Email:"""
    return prompt

def generate_email(model, tokenizer, context: str) -> str:
    """Generate an email for the given context"""
    # Create prompt
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
    
    # Extract only the email part (after "Email:")
    if "Email:" in full_text:
        email = full_text.split("Email:")[-1].strip()
    else:
        email = full_text
    
    return email

# ================================================================================
# TESTING
# ================================================================================

def run_tests(model, tokenizer, test_contexts: list):
    """Run tests on multiple contexts"""
    logger.info("="*80)
    logger.info("RUNNING TESTS")
    logger.info("="*80)
    logger.info(f"Number of test contexts: {len(test_contexts)}\n")
    
    results = []
    
    for idx, context in enumerate(test_contexts, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST {idx}/{len(test_contexts)}")
        logger.info(f"{'='*80}")
        logger.info(f"Context:\n{context}\n")
        
        # Generate email
        logger.info("Generating email...")
        email = generate_email(model, tokenizer, context)
        
        logger.info(f"\nGenerated Email:\n{email}\n")
        
        # Analyze email
        word_count = len(email.split())
        has_question = '?' in email
        has_cta = any(word in email.lower() for word in ['call', 'demo', 'connect', 'schedule', 'discuss'])
        
        logger.info(f"Analysis:")
        logger.info(f"  - Word count: {word_count}")
        logger.info(f"  - Contains question: {has_question}")
        logger.info(f"  - Has call-to-action: {has_cta}")
        
        results.append({
            'context': context,
            'email': email,
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
    avg_word_count = sum(r['word_count'] for r in results) / total_tests
    pct_with_questions = sum(r['has_question'] for r in results) / total_tests * 100
    pct_with_cta = sum(r['has_cta'] for r in results) / total_tests * 100
    
    logger.info(f"Total tests: {total_tests}")
    logger.info(f"Average word count: {avg_word_count:.1f}")
    logger.info(f"Emails with questions: {pct_with_questions:.1f}%")
    logger.info(f"Emails with CTA: {pct_with_cta:.1f}%")
    logger.info("="*80)

# ================================================================================
# INTERACTIVE MODE
# ================================================================================

def interactive_mode(model, tokenizer):
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
        email = generate_email(model, tokenizer, context)
        
        print("\n" + "="*80)
        print("GENERATED EMAIL:")
        print("="*80)
        print(email)
        print("="*80)

# ================================================================================
# MAIN
# ================================================================================

def main():
    """Main execution"""
    logger.info("="*80)
    logger.info("EMAIL GENERATION MODEL - TEST PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80 + "\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    if model is None or tokenizer is None:
        logger.error("Failed to load model. Exiting.")
        return
    
    # Run predefined tests
    results = run_tests(model, tokenizer, TEST_CONTEXTS)
    
    # Print summary
    print_summary(results)
    
    # Ask user if they want to try interactive mode
    print("\n" + "="*80)
    response = input("Would you like to try interactive mode? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        interactive_mode(model, tokenizer)
    
    logger.info("\n" + "="*80)
    logger.info("Testing completed!")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

if __name__ == "__main__":
    main()
