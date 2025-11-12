"""
End-to-End RL Fine-Tuning Pipeline with PPO for Email Generation
Using GPT-2 Small (124M parameters) for full fine-tuning without quantization or LoRA

This pipeline trains a model to generate high-converting sales emails using 
Reinforcement Learning with Proximal Policy Optimization (PPO).

Reward Structure:
- Outcome 0 (No Reply): Negative reward (-1.0)
- Outcome 1 (Reply): Moderate reward (+0.5)
- Outcome 2 (Reply + Conversion): High reward (+2.0)
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from dataclasses import dataclass

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    GenerationConfig
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset

# ================================================================================
# LOGGING SETUP
# ================================================================================

def setup_logging():
    """Setup comprehensive logging for training monitoring"""
    log_filename = f"rl_ppo_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
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
    data_path: str = "emails.csv"
    train_split: float = 0.85
    
    # PPO hyperparameters
    learning_rate: float = 1.5e-5
    batch_size: int = 8
    mini_batch_size: int = 4
    ppo_epochs: int = 4
    gradient_accumulation_steps: int = 2
    
    # Training settings
    total_episodes: int = 500  # Total number of training episodes
    save_steps: int = 50
    log_steps: int = 10
    
    # Generation settings
    generation_kwargs: dict = None
    
    # Reward settings
    reward_no_reply: float = -1.0
    reward_reply: float = 0.5
    reward_conversion: float = 2.0
    
    # Output paths
    output_dir: str = "./outputs/ppo_gpt2_email_model"
    checkpoint_dir: str = "./outputs/checkpoints"
    
    def __post_init__(self):
        if self.generation_kwargs is None:
            self.generation_kwargs = {
                "max_new_tokens": 128,
                "temperature": 0.9,
                "top_k": 50,
                "top_p": 0.95,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": 50256  # GPT-2 EOS token
            }
        
        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

# ================================================================================
# DATA LOADING AND PREPROCESSING
# ================================================================================

class EmailDataProcessor:
    """Process and prepare email data for RL training"""
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("Initializing EmailDataProcessor")
        
    def load_and_prepare_data(self) -> Tuple[Dataset, Dataset, pd.DataFrame]:
        """Load CSV data and prepare train/test splits"""
        logger.info(f"Loading data from {self.config.data_path}")
        
        # Load CSV
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Columns: {df.columns.tolist()}")
        logger.info(f"Outcome distribution:\n{df['outcome'].value_counts()}")
        
        # Shuffle data
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Split data
        train_size = int(len(df) * self.config.train_split)
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Convert to HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        return train_dataset, test_dataset, df
    
    def create_prompt(self, context: str) -> str:
        """Create a formatted prompt for the model"""
        prompt = f"""Write a professional sales email based on the following context.

Context: {context}

Email:"""
        return prompt
    
    def prepare_training_samples(self, dataset: Dataset, tokenizer) -> List[Dict]:
        """Prepare samples with prompts for training"""
        logger.info("Preparing training samples with prompts")
        
        samples = []
        for idx, example in enumerate(dataset):
            prompt = self.create_prompt(example['context'])
            
            # Tokenize prompt
            tokens = tokenizer.encode(prompt, return_tensors="pt")
            
            sample = {
                'query': prompt,
                'tokens': tokens,
                'context': example['context'],
                'reference_email': example['email_text'],
                'outcome': example['outcome']
            }
            samples.append(sample)
            
            if idx < 3:  # Log first 3 samples for inspection
                logger.info(f"\n{'='*80}")
                logger.info(f"Sample {idx + 1}:")
                logger.info(f"Context: {example['context']}")
                logger.info(f"Reference Email: {example['email_text']}")
                logger.info(f"Outcome: {example['outcome']}")
                logger.info(f"Prompt length: {tokens.shape[1]} tokens")
        
        logger.info(f"Prepared {len(samples)} training samples")
        return samples

# ================================================================================
# REWARD FUNCTION
# ================================================================================

class RewardModel:
    """
    Reward model that scores generated emails based on desired outcomes
    
    This is a heuristic-based reward model that evaluates email quality based on:
    1. Length appropriateness
    2. Professional language patterns
    3. Call-to-action presence
    4. Personalization elements
    """
    
    def __init__(self, config: Config):
        self.config = config
        logger.info("Initializing RewardModel")
        
        # Keywords that indicate high-quality emails
        self.positive_keywords = [
            'happy to', 'would love', 'glad', 'excited', 'appreciate',
            'case study', 'demo', 'call', 'connect', 'schedule',
            'help', 'optimize', 'growth', 'achieve', 'success'
        ]
        
        # Keywords that indicate low-quality emails
        self.negative_keywords = [
            'buy now', 'limited time', 'act now', 'urgent', 'hurry',
            'click here', 'guarantee', 'risk-free', 'free trial'
        ]
        
        self.call_to_action_phrases = [
            'call', 'demo', 'connect', 'schedule', 'discuss', 'share',
            'show', 'interested', 'feedback'
        ]
        
    def calculate_reward(self, generated_text: str, context: str) -> float:
        """
        Calculate reward for a generated email
        
        Reward components:
        - Base reward for reasonable length
        - Bonus for positive keywords
        - Penalty for negative keywords
        - Bonus for call-to-action
        - Bonus for personalization
        """
        text_lower = generated_text.lower()
        reward = 0.0
        
        # 1. Length reward (encourage 50-200 words)
        word_count = len(generated_text.split())
        if 50 <= word_count <= 200:
            length_reward = 0.5
        elif 30 <= word_count < 50 or 200 < word_count <= 250:
            length_reward = 0.2
        elif word_count < 30:
            length_reward = -0.5
        else:
            length_reward = -0.3
        
        reward += length_reward
        
        # 2. Positive keyword bonus
        positive_count = sum(1 for kw in self.positive_keywords if kw in text_lower)
        positive_reward = min(positive_count * 0.2, 1.0)  # Cap at 1.0
        reward += positive_reward
        
        # 3. Negative keyword penalty
        negative_count = sum(1 for kw in self.negative_keywords if kw in text_lower)
        negative_penalty = negative_count * -0.3
        reward += negative_penalty
        
        # 4. Call-to-action bonus
        has_cta = any(phrase in text_lower for phrase in self.call_to_action_phrases)
        if has_cta:
            reward += 0.5
        
        # 5. Personalization bonus (check for [Name] or similar)
        if '[name]' in text_lower or 'your company' in text_lower:
            reward += 0.3
        
        # 6. Professional greeting bonus
        greetings = ['hi', 'hello', 'good morning', 'greetings', 'hey']
        if any(text_lower.startswith(g) for g in greetings):
            reward += 0.2
        
        # 7. Question mark bonus (engagement)
        if '?' in generated_text:
            reward += 0.2
        
        # Normalize reward to be mostly in [-1, 2] range matching outcome rewards
        reward = np.clip(reward, -1.0, 2.0)
        
        return reward
    
    def compute_rewards(self, queries: List[str], responses: List[str]) -> List[torch.Tensor]:
        """Compute rewards for a batch of generated emails"""
        rewards = []
        
        for query, response in zip(queries, responses):
            # Extract context from query
            context = query.split("Context: ")[1].split("\n\nEmail:")[0] if "Context: " in query else ""
            
            # Calculate reward
            reward_value = self.calculate_reward(response, context)
            rewards.append(torch.tensor(reward_value))
        
        return rewards

# ================================================================================
# PPO TRAINER
# ================================================================================

class EmailPPOTrainer:
    """PPO Trainer for email generation task"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.ppo_trainer = None
        self.reward_model = None
        
    def setup(self):
        """Initialize model, tokenizer, and PPO trainer"""
        logger.info("="*80)
        logger.info("SETUP PHASE")
        logger.info("="*80)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"  # Important for generation
        logger.info("Tokenizer loaded successfully")
        
        # Load base model
        logger.info(f"Loading base model: {self.config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,  # Full precision, no quantization
        )
        logger.info(f"Base model loaded. Parameters: {base_model.num_parameters():,}")
        
        # Wrap with value head for PPO
        logger.info("Wrapping model with value head for PPO")
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
        self.model.to(self.device)
        logger.info("Model wrapped and moved to device")
        
        # PPO Configuration
        logger.info("Setting up PPO configuration")
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size,
            mini_batch_size=self.config.mini_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        
        # Initialize PPO Trainer
        logger.info("Initializing PPO Trainer")
        self.ppo_trainer = PPOTrainer(
            config=ppo_config,
            model=self.model,
            tokenizer=self.tokenizer,
        )
        logger.info("PPO Trainer initialized successfully")
        
        # Initialize reward model
        self.reward_model = RewardModel(self.config)
        
        logger.info("Setup completed successfully!")
        logger.info("="*80)
        
    def generate_responses(self, queries_tokens: List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[str]]:
        """Generate email responses for given query tokens"""
        response_tensors = []
        response_texts = []
        
        for query_tensor in queries_tokens:
            # Move to device
            query_tensor = query_tensor.to(self.device)
            
            # Generate
            with torch.no_grad():
                response_tensor = self.ppo_trainer.generate(
                    query_tensor,
                    **self.config.generation_kwargs
                )
            
            # Extract only the generated part (remove prompt)
            generated_tokens = response_tensor[0][query_tensor.shape[1]:]
            response_tensors.append(generated_tokens)
            
            # Decode
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response_texts.append(response_text)
        
        return response_tensors, response_texts
    
    def train(self, train_samples: List[Dict]):
        """Main training loop"""
        logger.info("="*80)
        logger.info("TRAINING PHASE")
        logger.info("="*80)
        logger.info(f"Total episodes: {self.config.total_episodes}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Mini batch size: {self.config.mini_batch_size}")
        logger.info("="*80)
        
        total_episodes = self.config.total_episodes
        batch_size = self.config.batch_size
        
        # Training metrics
        episode_rewards = []
        
        for episode in range(total_episodes):
            # Sample batch
            batch_indices = np.random.choice(len(train_samples), size=batch_size, replace=False)
            batch_samples = [train_samples[i] for i in batch_indices]
            
            # Prepare queries
            queries = [sample['query'] for sample in batch_samples]
            queries_tokens = [sample['tokens'] for sample in batch_samples]
            
            # Generate responses
            logger.info(f"\nEpisode {episode + 1}/{total_episodes}: Generating responses...")
            response_tensors, response_texts = self.generate_responses(queries_tokens)
            
            # Log some generated emails
            if episode % self.config.log_steps == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"EPISODE {episode + 1} - Sample Generations")
                logger.info(f"{'='*80}")
                for i in range(min(2, len(response_texts))):
                    logger.info(f"\nSample {i + 1}:")
                    logger.info(f"Context: {batch_samples[i]['context'][:100]}...")
                    logger.info(f"Generated Email:\n{response_texts[i]}")
                    logger.info(f"Reference Email:\n{batch_samples[i]['reference_email']}")
                    logger.info(f"Reference Outcome: {batch_samples[i]['outcome']}")
            
            # Calculate rewards
            logger.info(f"Episode {episode + 1}/{total_episodes}: Computing rewards...")
            rewards = self.reward_model.compute_rewards(queries, response_texts)
            
            # Convert rewards to tensors on correct device
            rewards = [r.to(self.device) for r in rewards]
            
            # Log rewards
            reward_values = [r.item() for r in rewards]
            avg_reward = np.mean(reward_values)
            episode_rewards.append(avg_reward)
            
            logger.info(f"Episode {episode + 1}/{total_episodes}: Average Reward = {avg_reward:.4f}")
            logger.info(f"Reward range: [{min(reward_values):.4f}, {max(reward_values):.4f}]")
            
            # PPO update step
            logger.info(f"Episode {episode + 1}/{total_episodes}: Running PPO update...")
            
            # Prepare batch for PPO
            query_tensors = [q.squeeze(0).to(self.device) for q in queries_tokens]
            response_tensors = [r.to(self.device) for r in response_tensors]
            
            # Run PPO step
            stats = self.ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log statistics
            if episode % self.config.log_steps == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"EPISODE {episode + 1} - Training Statistics")
                logger.info(f"{'='*80}")
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"{key}: {value:.4f}")
                
                # Log moving average reward
                if len(episode_rewards) >= 10:
                    moving_avg = np.mean(episode_rewards[-10:])
                    logger.info(f"Moving Average Reward (last 10 episodes): {moving_avg:.4f}")
            
            # Save checkpoint
            if (episode + 1) % self.config.save_steps == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir,
                    f"checkpoint_episode_{episode + 1}"
                )
                logger.info(f"\nSaving checkpoint to {checkpoint_path}")
                self.save_model(checkpoint_path)
        
        # Save final model
        logger.info("="*80)
        logger.info("Training completed!")
        logger.info(f"Final average reward: {episode_rewards[-1]:.4f}")
        logger.info(f"Saving final model to {self.config.output_dir}")
        self.save_model(self.config.output_dir)
        logger.info("="*80)
        
        return episode_rewards
    
    def save_model(self, path: str):
        """Save model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Model and tokenizer saved to {path}")

# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function"""
    logger.info("="*80)
    logger.info("EMAIL GENERATION RL-PPO TRAINING PIPELINE")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Initialize configuration
    config = Config()
    logger.info("Configuration:")
    logger.info(f"  Model: {config.model_name}")
    logger.info(f"  Max length: {config.max_length}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Total episodes: {config.total_episodes}")
    logger.info(f"  Output directory: {config.output_dir}")
    logger.info("="*80)
    
    # Load and prepare data
    logger.info("\nSTEP 1: DATA LOADING AND PREPARATION")
    data_processor = EmailDataProcessor(config)
    train_dataset, test_dataset, full_df = data_processor.load_and_prepare_data()
    
    # Initialize trainer
    logger.info("\nSTEP 2: MODEL AND TRAINER INITIALIZATION")
    trainer = EmailPPOTrainer(config)
    trainer.setup()
    
    # Prepare training samples
    logger.info("\nSTEP 3: PREPARING TRAINING SAMPLES")
    train_samples = data_processor.prepare_training_samples(train_dataset, trainer.tokenizer)
    
    # Train model
    logger.info("\nSTEP 4: TRAINING")
    episode_rewards = trainer.train(train_samples)
    
    # Final summary
    logger.info("="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(f"Total episodes: {len(episode_rewards)}")
    logger.info(f"Final reward: {episode_rewards[-1]:.4f}")
    logger.info(f"Average reward: {np.mean(episode_rewards):.4f}")
    logger.info(f"Best reward: {max(episode_rewards):.4f}")
    logger.info(f"Model saved to: {config.output_dir}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    logger.info("\n✓ Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
