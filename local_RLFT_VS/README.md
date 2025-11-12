# RL Fine-Tuning Pipeline for Mistral 7B with PPO

This directory contains a complete end-to-end Reinforcement Learning fine-tuning pipeline using Proximal Policy Optimization (PPO) to train Mistral 7B to write better sales emails that maximize customer conversion.

## 📋 Overview

**Goal**: Fine-tune Mistral 7B to generate sales emails that maximize the probability of:
- Getting a reply from the customer (outcome = 1)
- Converting the customer to a lead (outcome = 2)

**Method**: Use PPO (Proximal Policy Optimization) with a custom reward function that considers:
- Actual email outcomes from the dataset
- Email quality indicators (personalization, CTA, professionalism)
- Email length optimization
- Text coherence

## 🗂️ Data Format

The pipeline expects a CSV file with the following columns:
- `context`: Customer information and product details
- `email_text`: The actual email sent to the customer
- `outcome`: 0 (no reply), 1 (replied), 2 (replied + converted)

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Run the Pipeline

```powershell
python rl_mistral_ppo_pipeline.py --data-path emails.csv --output-dir ./outputs --ppo-steps 100
```

### 3. Monitor Progress

The pipeline will:
- Log detailed progress to `rl_mistral_ppo.log`
- Save checkpoints every 20 steps to `./outputs/checkpoint-*/`
- Log sample generations every 5 steps
- Display real-time metrics in the console

## ⚙️ Configuration Options

```powershell
python rl_mistral_ppo_pipeline.py \
  --data-path emails.csv \              # Path to your email dataset
  --output-dir ./outputs \              # Output directory
  --model-name mistralai/Mistral-7B-Instruct-v0.2 \  # Model to fine-tune
  --ppo-steps 100 \                     # Number of PPO training steps
  --batch-size 4 \                      # Batch size (reduce if OOM)
  --learning-rate 1.41e-5 \             # Learning rate
  --use-4bit \                          # Use 4-bit quantization (saves memory)
  --seed 42                             # Random seed for reproducibility
```

## 📊 Output Structure

After training, the output directory will contain:

```
outputs/
├── final_model/                    # Final trained model
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
├── checkpoint-20/                  # Intermediate checkpoints
├── checkpoint-40/
├── checkpoint-60/
├── training_metrics.csv            # Training metrics over time
├── test_results.csv               # Evaluation results on test set
└── rl_mistral_ppo.log             # Detailed training logs
```

## 🔍 Key Features

### 1. **Comprehensive Logging**
- Step-by-step progress tracking
- Sample email generations at regular intervals
- Detailed reward breakdowns
- Real-time metrics monitoring

### 2. **Reward Function**
The reward model considers multiple factors:
- **Outcome Reward** (60% weight): Based on actual email outcome (0, 1, or 2)
- **Quality Reward** (10% weight): Checks for:
  - Professional greetings
  - Clear call-to-action
  - Personalization elements
  - Value propositions
  - Avoids spam indicators
- **Length Penalty** (1% weight): Penalizes overly long or short emails
- **Coherence**: Basic structure validation

### 3. **Memory Efficient**
- Uses 4-bit quantization (reduces memory by ~75%)
- LoRA adapters for parameter-efficient fine-tuning
- Gradient accumulation support
- Batch size control

### 4. **Checkpointing**
- Saves model every 20 steps by default
- Allows resuming training if interrupted
- Keeps track of best performing checkpoints

### 5. **Evaluation**
- Automatic test set evaluation after training
- Generates sample emails with detailed analysis
- Saves all results to CSV for further analysis

## 📈 Monitoring Training

### Console Output
```
===================================================================================
RL Fine-Tuning Configuration
===================================================================================
  data_path: emails.csv
  model_name: mistralai/Mistral-7B-Instruct-v0.2
  ppo_steps: 100
  batch_size: 4
  ...
===================================================================================

Loading data from emails.csv
Loaded 102 email samples
Outcome distribution:
  Outcome 0: 68 samples (66.7%)
  Outcome 1: 10 samples (9.8%)
  Outcome 2: 24 samples (23.5%)

--- Step 5 ---
Mean Reward: 0.4523
  Outcome Reward: 0.3125
  Quality Reward: 0.1234
  Length Reward: 0.0164
PPO Loss: 0.0234
```

### Log File
Check `rl_mistral_ppo.log` for detailed logs including:
- Full configuration
- Sample email generations
- Reward breakdowns
- Error messages (if any)

## 🧪 Testing the Pipeline

Use the test script to verify the pipeline works:

```powershell
python test_pipeline.py
```

This will:
1. Check all dependencies are installed
2. Verify the data file is accessible
3. Test data loading
4. Test model loading (in simulation mode)
5. Test reward calculation
6. Validate the complete pipeline

## 🎯 Expected Results

After training for 100 steps, you should see:
- **Mean Reward**: Increasing trend from ~0.3 to ~0.5+
- **Outcome Reward**: Improvement in alignment with high-outcome emails
- **Quality Reward**: Better structured emails with CTAs and personalization
- **Generated Emails**: More professional, engaging, and conversion-focused

## 💡 Tips for Best Results

1. **Start Small**: Begin with 50-100 PPO steps to test the pipeline
2. **Monitor Memory**: If you get OOM errors, reduce `--batch-size` to 2 or 1
3. **Adjust Learning Rate**: If training is unstable, reduce `--learning-rate`
4. **Inspect Samples**: Review logged samples every 5 steps to see progress
5. **Use Checkpoints**: If training crashes, you can resume from checkpoints

## 🐛 Troubleshooting

### Out of Memory (OOM)
```powershell
# Reduce batch size
python rl_mistral_ppo_pipeline.py --batch-size 1

# Or use 8-bit quantization instead
python rl_mistral_ppo_pipeline.py --no-use-4bit --use-8bit
```

### Slow Training
- Use GPU if available (script automatically detects CUDA)
- Reduce `--ppo-steps` for faster testing
- Increase `--mini-batch-size` if memory allows

### Model Loading Issues
- Ensure you have Hugging Face access to Mistral models
- Login with: `huggingface-cli login`
- Check internet connection for model download

## 📚 Technical Details

### Architecture
- **Base Model**: Mistral 7B Instruct v0.2
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
  - Rank: 16
  - Alpha: 32
  - Target modules: q_proj, k_proj, v_proj, o_proj
- **RL Algorithm**: PPO (Proximal Policy Optimization)
  - PPO Epochs: 4
  - Gradient accumulation: 1
- **Quantization**: 4-bit (NF4) with double quantization

### Generation Parameters
- Max new tokens: 150
- Temperature: 0.7
- Top-k: 50
- Top-p: 0.95

## 📝 Citation

If you use this pipeline, please cite:

```
@misc{mistral-rl-ppo-2025,
  author = {Moid Hassan},
  title = {RL Fine-Tuning Pipeline for Mistral 7B with PPO},
  year = {2025},
  url = {https://github.com/hassanmoid/FromScratch}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- More sophisticated reward models
- Integration with wandb for better tracking
- Support for other models (Llama, GPT, etc.)
- Advanced evaluation metrics
- A/B testing framework

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

---

**Happy Fine-Tuning! 🚀**
