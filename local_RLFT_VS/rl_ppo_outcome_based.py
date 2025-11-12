"""
RL Fine-Tuning with PPO for Email Generation - Outcome-Based Rewards
Using trained DistilBERT reward model based on actual conversion outcomes

This pipeline:
1. Trains a DistilBERT classifier to predict outcomes (0, 1, 2) from email text
2. Uses this trained model as the reward function
3. Fine-tunes GPT-2 with PPO to maximize predicted conversion outcomes
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset

# ================================================================================
# LOGGING SETUP
# ================================================================================

def setup_logging():
    """Setup comprehensive logging"""
    log_filename = f"rl_ppo_outcome_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    """Training configuration"""
    # Data
    data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emails.csv")
    
    # Reward Model Training
    reward_model_name: str = "distilbert-base-uncased"
    reward_model_epochs: int = 3
    reward_model_batch_size: int = 8
    reward_model_lr: float = 2e-5
    
    # Generation Model
    gen_model_name: str = "gpt2"
    
    # RL Training
    rl_learning_rate: float = 1e-5
    rl_batch_size: int = 4
    rl_num_epochs: int = 3
    gradient_accumulation_steps: int = 2
    
    # Generation settings
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.95
    
    # Output paths
    reward_model_dir: str = "./outputs/reward_model"
    gen_model_dir: str = "./outputs/ppo_gpt2_outcome"
    
    def __post_init__(self):
        os.makedirs(self.reward_model_dir, exist_ok=True)
        os.makedirs(self.gen_model_dir, exist_ok=True)

# ================================================================================
# DATA LOADING
# ================================================================================

def load_data(config: Config):
    """Load and prepare email dataset"""
    logger.info(f"Loading data from {config.data_path}")
    df = pd.read_csv(config.data_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Outcome distribution:\n{df['outcome'].value_counts()}")
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# ================================================================================
# REWARD MODEL TRAINING
# ================================================================================

class RewardModelTrainer:
    """Train DistilBERT to predict email outcomes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        
    def prepare_data(self, df: pd.DataFrame):
        """Prepare data for reward model training"""
        logger.info("Preparing data for reward model training")
        
        # Split data
        train_df, val_df = train_test_split(
            df, 
            test_size=0.15, 
            random_state=42,
            stratify=df['outcome']
        )
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Val samples: {len(val_df)}")
        logger.info(f"Train outcome distribution:\n{train_df['outcome'].value_counts()}")
        
        # Create datasets
        train_dataset = Dataset.from_pandas(train_df[['email_text', 'outcome']])
        val_dataset = Dataset.from_pandas(val_df[['email_text', 'outcome']])
        
        return train_dataset, val_dataset
    
    def tokenize_function(self, examples):
        """Tokenize email texts"""
        return self.tokenizer(
            examples['email_text'],
            truncation=True,
            padding='max_length',
            max_length=256
        )
    
    def train(self, df: pd.DataFrame):
        """Train the reward model"""
        logger.info("="*80)
        logger.info("TRAINING REWARD MODEL")
        logger.info("="*80)
        
        # Load tokenizer and model
        logger.info(f"Loading {self.config.reward_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.reward_model_name)
        
        # 3 classes: outcome 0, 1, 2
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.reward_model_name,
            num_labels=3
        )
        self.model.to(self.device)
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_data(df)
        
        # Tokenize
        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        
        # Rename 'outcome' to 'labels' for Trainer
        train_dataset = train_dataset.rename_column('outcome', 'labels')
        val_dataset = val_dataset.rename_column('outcome', 'labels')
        
        # Set format
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.reward_model_dir,
            num_train_epochs=self.config.reward_model_epochs,
            per_device_train_batch_size=self.config.reward_model_batch_size,
            per_device_eval_batch_size=self.config.reward_model_batch_size,
            learning_rate=self.config.reward_model_lr,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            accuracy = (predictions == labels).mean()
            return {"accuracy": accuracy}
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train
        logger.info("Starting reward model training...")
        trainer.train()
        
        # Save
        logger.info(f"Saving reward model to {self.config.reward_model_dir}")
        trainer.save_model(self.config.reward_model_dir)
        self.tokenizer.save_pretrained(self.config.reward_model_dir)
        
        # Evaluate
        logger.info("Evaluating reward model...")
        metrics = trainer.evaluate()
        logger.info(f"Final metrics: {metrics}")
        
        logger.info("="*80)
        logger.info("REWARD MODEL TRAINING COMPLETED")
        logger.info("="*80)
        
        return self.model, self.tokenizer

# ================================================================================
# REWARD MODEL (INFERENCE)
# ================================================================================

class OutcomeRewardModel:
    """Use trained DistilBERT to compute rewards"""
    
    def __init__(self, model_path: str, device: torch.device):
        logger.info(f"Loading reward model from {model_path}")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        logger.info("Reward model loaded successfully")
        
        # Map predicted class to reward
        # outcome 0 (no reply) -> -1.0
        # outcome 1 (reply) -> 0.5
        # outcome 2 (conversion) -> 2.0
        self.outcome_to_reward = {
            0: -1.0,
            1: 0.5,
            2: 2.0
        }
    
    def compute_reward(self, text: str) -> float:
        """Compute reward for generated email"""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).item()
        
        # Map to reward
        reward = self.outcome_to_reward[predicted_class]
        
        return reward, predicted_class
    
    def compute_rewards_batch(self, texts: List[str]) -> List[float]:
        """Compute rewards for batch of texts"""
        rewards = []
        for text in texts:
            reward, _ = self.compute_reward(text)
            rewards.append(reward)
        return rewards

# ================================================================================
# RL TRAINING WITH PPO
# ================================================================================

class SimpleRLTrainer:
    """Simple RL trainer with outcome-based rewards"""
    
    def __init__(self, config: Config, reward_model: OutcomeRewardModel):
        self.config = config
        self.reward_model = reward_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.optimizer = None
    
    def setup(self):
        """Initialize generation model"""
        logger.info("="*80)
        logger.info("SETUP RL TRAINING")
        logger.info("="*80)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.gen_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.gen_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        # Load model
        logger.info(f"Loading model: {self.config.gen_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(self.config.gen_model_name)
        self.model.to(self.device)
        logger.info(f"Model loaded. Parameters: {self.model.num_parameters():,}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.rl_learning_rate
        )
        
        logger.info("Setup completed!")
        logger.info("="*80)
    
    def create_prompt(self, context: str) -> str:
        """Create prompt from context"""
        return f"""Write a professional sales email based on the following context.

Context: {context}

Email:"""
    
    def generate_response(self, prompt: str) -> str:
        """Generate email response"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Decode only generated part
        prompt_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return generated_text
    
    def train_epoch(self, df: pd.DataFrame, epoch: int):
        """Train for one epoch"""
        logger.info(f"\n{'='*80}")
        logger.info(f"EPOCH {epoch + 1}")
        logger.info(f"{'='*80}")
        
        epoch_rewards = []
        epoch_predicted_outcomes = []
        total_steps = len(df) // self.config.rl_batch_size
        
        for step in range(total_steps):
            # Sample batch
            batch_indices = np.random.choice(
                len(df),
                size=self.config.rl_batch_size,
                replace=False
            )
            batch = df.iloc[batch_indices]
            
            # Generate responses
            batch_data = []
            for _, row in batch.iterrows():
                prompt = self.create_prompt(row['context'])
                generated_text = self.generate_response(prompt)
                reward, predicted_outcome = self.reward_model.compute_reward(generated_text)
                
                batch_data.append({
                    'prompt': prompt,
                    'generated_text': generated_text,
                    'reward': reward,
                    'predicted_outcome': predicted_outcome,
                    'actual_outcome': row['outcome'],
                    'context': row['context']
                })
                
                epoch_rewards.append(reward)
                epoch_predicted_outcomes.append(predicted_outcome)
            
            # Log samples
            if step % 10 == 0:
                logger.info(f"\nStep {step + 1}/{total_steps}")
                logger.info(f"Sample generation:")
                logger.info(f"Context: {batch_data[0]['context'][:80]}...")
                logger.info(f"Generated: {batch_data[0]['generated_text'][:150]}...")
                logger.info(f"Predicted outcome: {batch_data[0]['predicted_outcome']}")
                logger.info(f"Reward: {batch_data[0]['reward']:.3f}")
                logger.info(f"Avg reward: {np.mean(epoch_rewards[-10:]):.3f}")
            
            # Training step (simplified PPO - reward-weighted supervised learning)
            self.model.train()
            total_loss = 0
            
            for data in batch_data:
                # Prepare training data
                full_text = data['prompt'] + data['generated_text']
                inputs = self.tokenizer(
                    full_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                # Weight loss by reward (higher reward = encourage this generation)
                # Normalize reward to [0, 1] range: (-1 to 2) -> (0 to 1)
                normalized_reward = (data['reward'] + 1.0) / 3.0  # Now in [0, 1]
                loss_weight = 1.0 - normalized_reward  # High reward = low loss weight
                weighted_loss = loss * max(0.1, loss_weight)
                
                total_loss += weighted_loss
            
            # Backward pass
            avg_loss = total_loss / self.config.rl_batch_size
            avg_loss.backward()
            
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            if step % 10 == 0:
                logger.info(f"Loss: {avg_loss.item():.4f}")
        
        # Epoch summary
        avg_reward = np.mean(epoch_rewards)
        outcome_dist = pd.Series(epoch_predicted_outcomes).value_counts()
        
        logger.info(f"\nEpoch {epoch + 1} completed!")
        logger.info(f"Average reward: {avg_reward:.4f}")
        logger.info(f"Predicted outcome distribution:\n{outcome_dist}")
        logger.info(f"Min reward: {min(epoch_rewards):.4f}")
        logger.info(f"Max reward: {max(epoch_rewards):.4f}")
        
        return avg_reward
    
    def train(self, df: pd.DataFrame):
        """Main training loop"""
        logger.info("="*80)
        logger.info("RL TRAINING PHASE")
        logger.info("="*80)
        
        epoch_rewards = []
        
        for epoch in range(self.config.rl_num_epochs):
            avg_reward = self.train_epoch(df, epoch)
            epoch_rewards.append(avg_reward)
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config.gen_model_dir,
                f"checkpoint_epoch_{epoch + 1}"
            )
            self.save_model(checkpoint_path)
        
        # Save final model
        logger.info("="*80)
        logger.info("RL Training completed!")
        logger.info(f"Final average reward: {epoch_rewards[-1]:.4f}")
        self.save_model(self.config.gen_model_dir)
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
    logger.info("EMAIL GENERATION RL TRAINING - OUTCOME-BASED REWARDS")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Configuration
    config = Config()
    logger.info("Configuration:")
    logger.info(f"  Data: {config.data_path}")
    logger.info(f"  Reward model: {config.reward_model_name}")
    logger.info(f"  Generation model: {config.gen_model_name}")
    logger.info(f"  RL learning rate: {config.rl_learning_rate}")
    logger.info(f"  RL batch size: {config.rl_batch_size}")
    logger.info(f"  RL epochs: {config.rl_num_epochs}")
    logger.info("="*80)
    
    # Load data
    logger.info("\nSTEP 1: DATA LOADING")
    df = load_data(config)
    
    # Train reward model
    logger.info("\nSTEP 2: TRAIN REWARD MODEL")
    reward_trainer = RewardModelTrainer(config)
    reward_model_nn, reward_tokenizer = reward_trainer.train(df)
    
    # Initialize reward model for inference
    logger.info("\nSTEP 3: INITIALIZE REWARD MODEL FOR RL")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reward_model = OutcomeRewardModel(config.reward_model_dir, device)
    
    # Test reward model on some samples
    logger.info("\nTesting reward model on training samples:")
    test_samples = df.sample(5, random_state=42)
    for idx, row in test_samples.iterrows():
        reward, pred_outcome = reward_model.compute_reward(row['email_text'])
        logger.info(f"Email excerpt: {row['email_text'][:100]}...")
        logger.info(f"  Actual outcome: {row['outcome']}, Predicted: {pred_outcome}, Reward: {reward:.2f}")
    
    # Train generation model with RL
    logger.info("\nSTEP 4: RL TRAINING")
    rl_trainer = SimpleRLTrainer(config, reward_model)
    rl_trainer.setup()
    epoch_rewards = rl_trainer.train(df)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total epochs: {len(epoch_rewards)}")
    logger.info(f"Final reward: {epoch_rewards[-1]:.4f}")
    logger.info(f"Average reward: {np.mean(epoch_rewards):.4f}")
    logger.info(f"Best reward: {max(epoch_rewards):.4f}")
    logger.info(f"Reward model saved to: {config.reward_model_dir}")
    logger.info(f"Generation model saved to: {config.gen_model_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    logger.info("\n✓ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
