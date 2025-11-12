# RL Fine-Tuning with PPO for Email Generation

This project implements an end-to-end Reinforcement Learning pipeline using Proximal Policy Optimization (PPO) to fine-tune GPT-2 for generating high-converting sales emails.

## Overview

The pipeline trains GPT-2 to write professional sales emails that maximize conversion rates (replies and lead conversions) based on customer context.

**Key Features:**
- Full fine-tuning (no LoRA, no quantization)
- Custom reward model based on email quality heuristics
- Comprehensive logging at each step
- GPU/CPU compatible (uses GPT-2 Small - 124M parameters for local training)
- Simple, custom PPO implementation without complex dependencies

## Files

- `rl_ppo_simple.py` - Main training pipeline (simplified, most reliable)
- `rl_mistral_ppo_pipeline.py` - Original TRL-based pipeline (may have version issues)
- `test_pipeline.py` - Test script for generating emails with trained model
- `emails.csv` - Training data with context, email_text, and outcomes
- `requirements.txt` - Python dependencies
- `run_training.bat` - Windows batch file to run training

## Data Format

The `emails.csv` contains:
- `context`: Customer information and engagement status
- `email_text`: The email that was sent
- `outcome`: 
  - 0 = No reply
  - 1 = Reply only  
  - 2 = Reply + Lead conversion (target)

## Installation

```bash
pip install -r requirements.txt
```

## Running Training

### Method 1: Direct Python Execution (Recommended)

Open a new terminal/command prompt and run:

```bash
cd d:\Learnings\FromScratch\local_RLFT_VS
C:\Users\moidhassan\AppData\Local\anaconda3\python.exe rl_ppo_simple.py
```

### Method 2: Using Batch File

Double-click `run_training.bat` or run:

```bash
d:\Learnings\FromScratch\local_RLFT_VS\run_training.bat
```

### Method 3: Using Conda Environment

```bash
conda activate base
cd d:\Learnings\FromScratch\local_RLFT_VS
python rl_ppo_simple.py
```

## Training Configuration

Default settings in `rl_ppo_simple.py`:

```python
model_name: "gpt2"          # GPT-2 Small (124M parameters)
learning_rate: 1e-5
batch_size: 4
num_epochs: 3
clip_epsilon: 0.2           # PPO clipping parameter
```

You can modify these in the `Config` class.

## Monitoring Training

Training progress is logged to:
1. Console output (real-time)
2. Log file: `rl_ppo_simple_YYYYMMDD_HHMMSS.log`

The log includes:
- Data loading statistics
- Model initialization details
- Per-step sample generations
- Reward values
- Training loss
- Epoch summaries

## Reward Model

The custom reward model evaluates emails based on:

1. **Length appropriateness** (50-200 words optimal)
2. **Positive keywords** ("happy to", "would love", "case study", etc.)
3. **Negative keyword penalties** ("buy now", "limited time", etc.)
4. **Call-to-action presence** ("call", "demo", "connect", "schedule")
5. **Personalization** ("[Name]", "your company")
6. **Professional greeting** ("Hi", "Hello", "Good morning")
7. **Engagement questions** (presence of '?')

Rewards are normalized to range [-1.0, 2.0] matching the outcome labels.

## Output

Trained models are saved to:
- `./outputs/ppo_gpt2_simple/` - Final model
- `./outputs/ppo_gpt2_simple/checkpoint_epoch_X/` - Intermediate checkpoints

## Testing the Trained Model

After training completes, test the model:

```bash
C:\Users\moidhassan\AppData\Local\anaconda3\python.exe test_pipeline.py
```

This will:
1. Load the trained model
2. Generate emails for 8 test contexts
3. Display analysis (word count, CTA presence, etc.)
4. Offer interactive mode for custom contexts

## Expected Training Time

On CPU:
- ~20-30 minutes per epoch
- Total: ~1-1.5 hours for 3 epochs

On GPU (if available):
- ~5-10 minutes per epoch
- Total: ~15-30 minutes for 3 epochs

## Troubleshooting

### Issue: "FileNotFoundError: emails.csv"
**Solution**: Make sure you're in the correct directory when running the script.

### Issue: "Out of memory"
**Solution**: Reduce `batch_size` in the Config class (try 2 or 1).

### Issue: Training seems stuck
**Solution**: Check the log file to see actual progress. Initial model loading can take 1-2 minutes.

### Issue: KeyboardInterrupt during import
**Solution**: This can happen with VS Code terminals. Try running from a standard Windows Command Prompt or PowerShell window opened separately.

## Architecture

### Training Pipeline:
1. **Data Loading**: Load and split email dataset
2. **Model Setup**: Load GPT-2 and tokenizer
3. **Training Loop**:
   - Sample batch of contexts
   - Generate emails for each context
   - Calculate rewards using heuristic model
   - Compute weighted loss (higher reward = lower loss weight)
   - Backpropagate and update model
4. **Save Model**: Save checkpoints and final model

### Generation Process:
1. Format context into prompt template
2. Tokenize prompt
3. Generate continuation with sampling parameters
4. Decode and extract email text

## Future Enhancements

Possible improvements:
1. Use actual conversion data for reward model
2. Implement full PPO with value function and advantages
3. Add A/B testing framework
4. Fine-tune on larger model (GPT-2 Medium/Large)
5. Implement curriculum learning (easy → hard examples)

## Notes

- The pipeline uses a simplified PPO approach with reward-weighted supervised learning
- For production use, consider implementing full PPO with proper advantage estimation
- The reward model is heuristic-based; ideally train a reward model on human preferences
- GPU usage will significantly speed up training

## Contact & Support

For issues or questions, check the log files first. They contain detailed information about each step of the training process.
