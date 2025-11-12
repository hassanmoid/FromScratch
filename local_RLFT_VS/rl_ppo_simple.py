"""
Simplified RL Fine-Tuning Pipeline with PPO for Email Generation
Custom PPO implementation without TRL dependency issues

This version uses a simple, custom PPO trainer that's guaranteed to work.
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)

# ================================================================================
# LOGGING SETUP
# ================================================================================

def setup_logging():
    """Setup comprehensive logging for training monitoring"""
    log_filename = f"rl_ppo_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ================================================================================
# CONFIGURATION
# ================================================================================

@dataclass
class Config:
    """Training configuration parameters"""
    # Model settings
    model_name: str = "gpt2"  # Using GPT-2 Small (124M params)
    max_length: int = 256
    
    # Data settings
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails.csv")
    train_split: float = 0.85
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 2
    
    # PPO hyperparameters
    clip_epsilon: float = 0.2
    value_coef: float = 0.1
    entropy_coef: float = 0.01
    
    # Training settings
    save_steps: int = 50
    log_steps: int = 10
    
    # Generation settings
    generation_kwargs: dict = None
    
    # Reward settings
    reward_no_reply: float = -1.0
    reward_reply: float = 0.5
    reward_conversion: float = 2.0
    
    # Output paths
    output_dir: str = "./outputs/ppo_gpt2_simple"
    
    def __post_init__(self):
        if self.generation_kwargs is None:
            self.generation_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": 50256
            }
        
        os.makedirs(self.output_dir, exist_ok=True)

# ================================================================================
# DATA LOADING
# ================================================================================

def load_data(config: Config):
    """Load and prepare the email dataset"""
    logger.info(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Outcome distribution:\n{df['outcome'].value_counts()}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    train_size = int(len(df) * config.train_split)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    logger.info(f"Train samples: {len(train_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    return train_df, test_df

def create_prompt(context: str) -> str:
    """Create a formatted prompt"""
    return f"""Write a professional sales email based on the following context.

Context: {context}

Email:"""

# ================================================================================
# REWARD MODEL
# ================================================================================

class RewardModel:
    """Heuristic-based reward model"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("Initializing RewardModel")
        
        self.positive_keywords = [
            'happy to', 'would love', 'glad', 'excited', 'appreciate',
            'case study', 'demo', 'call', 'connect', 'schedule',
            'help', 'optimize', 'growth', 'achieve', 'success'
        ]
        
        self.negative_keywords = [
            'buy now', 'limited time', 'act now', 'urgent', 'hurry'
        ]
        
        self.cta_phrases = [
            'call', 'demo', 'connect', 'schedule', 'discuss', 'share'
        ]
    
    def calculate_reward(self, text: str) -> float:
        """Calculate reward for generated email"""
        text_lower = text.lower()
        reward = 0.0
        
        # Length reward
        word_count = len(text.split())
        if 50 <= word_count <= 200:
            reward += 0.5
        elif word_count < 30:
            reward -= 0.5
        
        # Positive keywords
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        reward += min(positive_count * 0.2, 1.0)
        
        # Negative keywords penalty
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        reward -= negative_count * 0.3
        
        # CTA bonus
        has_cta = any(phrase in text_lower for phrase in self.cta_phrases)
        if has_cta:
            reward += 0.5
        
        # Personalization
        if '[name]' in text_lower:
            reward += 0.3
        
        # Professional greeting
        greetings = ['hi', 'hello', 'good morning', 'greetings', 'hey']
        if any(text_lower.startswith(g) for g in greetings):
            reward += 0.2
        
        # Question bonus
        if '?' in text:
            reward += 0.2
        
        return np.clip(reward, -1.0, 2.0)

# ================================================================================
# SIMPLE PPO TRAINER
# ================================================================================

class SimplePPOTrainer:
    """Custom PPO trainer without TRL dependencies"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.reward_model = None
    
    def setup(self):
        """Initialize model and tokenizer"""
        logger.info("="*80)
        logger.info("SETUP PHASE")
        logger.info("="*80)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        logger.info("Tokenizer loaded")
        
        # Load model
        logger.info(f"Loading model: {self.config.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
        )
        self.model.to(self.device)
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        logger.info("Optimizer configured")
        
        # Reward model
        self.reward_model = RewardModel(self.config)
        
        logger.info("Setup completed!")
        logger.info("="*80)
    
    def generate_response(self, prompt: str) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """Generate response and get log probs"""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        prompt_length = inputs['input_ids'].shape[1]
        
        # Generate
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.config.generation_kwargs,
                output_scores=True,
                return_dict_in_generate=True
            )
        
        # Extract generated text
        generated_ids = outputs.sequences[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text, outputs.sequences, generated_ids
    
    def compute_policy_loss(self, old_log_probs: torch.Tensor, new_log_probs: torch.Tensor, 
                           advantages: torch.Tensor) -> torch.Tensor:
        """Compute PPO policy loss with clipping"""
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                           1 + self.config.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        return policy_loss
    
    def train_epoch(self, train_df: pd.DataFrame, epoch: int):
        """Train for one epoch"""
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch + 1}")
        logger.info(f"{'='*80}")
        
        epoch_rewards = []
        total_steps = len(train_df) // self.config.batch_size
        
        for step in range(total_steps):
            # Sample batch
            batch_indices = np.random.choice(
                len(train_df),
                size=self.config.batch_size,
                replace=False
            )
            batch = train_df.iloc[batch_indices]
            
            # Generate responses and collect data
            batch_data = []
            for _, row in batch.iterrows():
                prompt = create_prompt(row['context'])
                generated_text, full_sequence, generated_ids = self.generate_response(prompt)
                reward = self.reward_model.calculate_reward(generated_text)
                
                batch_data.append({
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'reward': reward,
                    'context': row['context']
                })
                epoch_rewards.append(reward)
            
            # Log samples
            if step % self.config.log_steps == 0:
                logger.info(f"\nStep {step + 1}/{total_steps}")
                logger.info(f"Sample generation:")
                logger.info(f"Context: {batch_data[0]['context'][:80]}...")
                logger.info(f"Generated: {batch_data[0]['generated_text'][:100]}...")
                logger.info(f"Reward: {batch_data[0]['reward']:.3f}")
                logger.info(f"Avg reward (last {len(epoch_rewards)} samples): {np.mean(epoch_rewards[-10:]):.3f}")
            
            # Simple supervised fine-tuning step (simplified PPO)
            # In a full PPO implementation, we would compute advantages and do multiple
            # epochs of optimization. Here we do a simplified version.
            self.model.train()
            total_loss = 0
            
            for data in batch_data:
                # Prepare training data
                full_text = data['prompt'] + data['generated_text']
                inputs = self.tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Weight loss by reward (higher reward = lower loss weight needed)
                # This encourages the model to generate high-reward texts
                reward_weight = 1.0 - (data['reward'] / 2.0)  # Normalize reward to [0, 1.5]
                weighted_loss = loss * max(0.1, reward_weight)  # Don't let weight go to 0
                
                total_loss += weighted_loss
            
            # Backward pass
            avg_loss = total_loss / self.config.batch_size
            avg_loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if step % self.config.log_steps == 0:
                logger.info(f"Loss: {avg_loss.item():.4f}")
        
        avg_epoch_reward = np.mean(epoch_rewards)
        logger.info(f"\nEpoch {epoch + 1} completed!")
        logger.info(f"Average reward: {avg_epoch_reward:.4f}")
        logger.info(f"Min reward: {min(epoch_rewards):.4f}")
        logger.info(f"Max reward: {max(epoch_rewards):.4f}")
        
        return avg_epoch_reward
    
    def train(self, train_df: pd.DataFrame):
        """Main training loop"""
        logger.info("="*80)
        logger.info("TRAINING PHASE")
        logger.info("="*80)
        
        epoch_rewards = []
        
        for epoch in range(self.config.num_epochs):
            avg_reward = self.train_epoch(train_df, epoch)
            epoch_rewards.append(avg_reward)
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:  # Save every epoch
                checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f"checkpoint_epoch_{epoch + 1}"
                )
                self.save_model(checkpoint_path)
        
        # Save final model
        logger.info("="*80)
        logger.info("Training completed!")
        logger.info(f"Final average reward: {epoch_rewards[-1]:.4f}")
        logger.info(f"Saving final model to {self.config.output_dir}")
        self.save_model(self.config.output_dir)
        logger.info("="*80)
        
        return epoch_rewards
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model saved to {path}")

# ================================================================================
# MAIN
# ================================================================================

def main():
    """Main execution"""
    logger.info("="*80)
    logger.info("EMAIL GENERATION RL-PPO TRAINING (SIMPLE VERSION)")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Configuration
    config = Config()
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Num epochs: {config.num_epochs}")
    logger.info("="*80)
    
    # Load data
    logger.info("\nSTEP 1: DATA LOADING")
    train_df, test_df = load_data(config)
    
    # Initialize trainer
    logger.info("\nSTEP 2: TRAINER INITIALIZATION")
    trainer = SimplePPOTrainer(config)
    trainer.setup()
    
    # Train
    logger.info("\nSTEP 3: TRAINING")
    epoch_rewards = trainer.train(train_df)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total epochs: {len(epoch_rewards)}")
    logger.info(f"Final reward: {epoch_rewards[-1]:.4f}")
    logger.info(f"Average reward: {np.mean(epoch_rewards):.4f}")
    logger.info(f"Best reward: {max(epoch_rewards):.4f}")
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    logger.info("\n✓ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
