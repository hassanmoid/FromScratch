# RL Fine-Tuning with Outcome-Based Rewards

## 🎯 Simplified Approach

This implementation uses **only the actual outcome data** (0, 1, 2) from your CSV to train the reward model. No heuristics or complex reward engineering.

### How It Works:

1. **Train Reward Model**: DistilBERT classifier learns to predict outcomes from email text
   - Input: Email text
   - Output: Predicted outcome (0=No Reply, 1=Reply, 2=Conversion)
   - Maps to rewards: -1.0, 0.5, 2.0 respectively

2. **RL Fine-Tuning**: GPT-2 generates emails and gets feedback from reward model
   - Generates emails based on context
   - Reward model predicts likely outcome
   - Model learns to generate emails that maximize conversion probability

## 📁 Files

- `rl_ppo_outcome_based.py` - Main training pipeline (outcome-based rewards)
- `test_outcome_model.py` - Test script for the trained model
- `emails.csv` - Training data (context, email_text, outcome)
- `requirements.txt` - Dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd d:\Learnings\FromScratch\local_RLFT_VS
C:\Users\moidhassan\AppData\Local\anaconda3\python.exe rl_ppo_outcome_based.py
```

This will:
- Train DistilBERT reward model (~5-10 minutes)
- Fine-tune GPT-2 with RL (~30-60 minutes on CPU, ~10-15 minutes on GPU)
- Save both models to `./outputs/`

### 3. Test the Model

```bash
C:\Users\moidhassan\AppData\Local\anaconda3\python.exe test_outcome_model.py
```

## 📊 Training Process

### Phase 1: Reward Model Training
```
Data: 100 emails → Train/Val split (85/15)
Model: DistilBERT-base-uncased
Task: 3-class classification (outcomes 0, 1, 2)
Epochs: 3
Expected accuracy: ~60-80% (depends on data quality)
```

### Phase 2: RL Fine-Tuning
```
Model: GPT-2 Small (124M parameters)
Training: Reward-weighted supervised learning (simplified PPO)
Epochs: 3
Batch size: 4
Outcome: Model learns to generate high-converting emails
```

## 🎯 Reward Structure

**Simple and data-driven:**

```python
Outcome 0 (No Reply)     → Reward: -1.0
Outcome 1 (Reply)        → Reward:  0.5
Outcome 2 (Conversion)   → Reward:  2.0
```

The reward model (trained DistilBERT) predicts which outcome a generated email will likely have.

## 📈 Monitoring Training

Training creates detailed logs:
- `rl_ppo_outcome_YYYYMMDD_HHMMSS.log`

Logs include:
- Reward model training metrics (accuracy, loss)
- Sample email generations at each step
- Predicted outcomes and rewards
- Epoch summaries

## 🔍 Example Output

```
Context: "Customer: CTO at a Healthcare firm was referred by a partner for a telehealth app."

Generated Email:
"Hi there,

Our telehealth app has helped several healthcare companies optimize their workflow. 
I wanted to reach out regarding your interest in our solution. We've seen great 
success with similar organizations.

Would you be open to a quick call this week to discuss how we can help?

Best regards"

Predicted Outcome: Conversion (class 2)
Reward: 2.0
Probabilities:
  - No Reply: 0.15
  - Reply: 0.25
  - Conversion: 0.60
```

## 💡 Key Advantages

1. **No Complex Heuristics**: Uses only your actual conversion data
2. **Data-Driven**: Learns what actually leads to conversions
3. **Interpretable**: Clear mapping from predicted outcome to reward
4. **Efficient**: DistilBERT is fast and lightweight
5. **Scalable**: Easy to retrain as you get more outcome data

## 🛠️ Configuration

Edit the `Config` class in `rl_ppo_outcome_based.py`:

```python
@dataclass
class Config:
    # Reward Model Training
    reward_model_name: str = "distilbert-base-uncased"
    reward_model_epochs: int = 3
    reward_model_batch_size: int = 8
    
    # Generation Model
    gen_model_name: str = "gpt2"
    
    # RL Training
    rl_learning_rate: float = 1e-5
    rl_batch_size: int = 4
    rl_num_epochs: int = 3
```

## 📁 Output Structure

```
outputs/
├── reward_model/          # Trained DistilBERT classifier
│   ├── config.json
│   ├── pytorch_model.bin
│   └── tokenizer files...
└── ppo_gpt2_outcome/      # RL fine-tuned GPT-2
    ├── config.json
    ├── pytorch_model.bin
    └── tokenizer files...
```

## 🧪 Testing

The test script (`test_outcome_model.py`) provides:

1. **Predefined Tests**: 8 diverse contexts
2. **Outcome Predictions**: Shows predicted outcome & probabilities
3. **Analysis**: Word count, CTA presence, questions
4. **Interactive Mode**: Test with your own contexts

## 📊 Expected Results

After training, you should see:
- **Reward Model Accuracy**: 60-80% (on 100 samples)
- **Average Reward Improvement**: Increases from ~0.0 to ~0.5-1.0
- **Outcome Distribution**: More emails predicted as "Conversion" (class 2)

## 🔧 Troubleshooting

### Low Reward Model Accuracy (<50%)
- Dataset may be too small (100 samples)
- Consider collecting more labeled data
- Check if email_text quality is good

### Model Not Improving
- Increase `rl_num_epochs` (try 5-10)
- Decrease `rl_learning_rate` (try 5e-6)
- Increase `rl_batch_size` if you have GPU memory

### Out of Memory
- Reduce `reward_model_batch_size` to 4
- Reduce `rl_batch_size` to 2

## 🚀 Next Steps

1. **Collect More Data**: More outcome labels = better reward model
2. **A/B Test**: Use generated emails in real campaigns
3. **Update Reward Model**: Retrain periodically with new outcomes
4. **Experiment with Models**: Try different base models (GPT-2 Medium, etc.)

## 📝 Technical Details

### Reward Model Architecture
```
DistilBERT (66M params)
  ↓
[CLS] token representation
  ↓
Classification head (3 classes)
  ↓
Outcome prediction (0, 1, or 2)
```

### RL Training Algorithm
```
1. Sample context from dataset
2. Generate email with current GPT-2
3. Predict outcome with reward model
4. Compute reward from outcome
5. Update GPT-2 with reward-weighted loss
6. Repeat
```

## 📚 References

- **Outcome-Based Learning**: Uses actual business metrics
- **Simplified PPO**: Reward-weighted supervised learning
- **Transfer Learning**: Leverages pre-trained DistilBERT and GPT-2

---

**Note**: This approach is much simpler and more interpretable than heuristic-based rewards. It directly optimizes for what matters: actual conversion outcomes!
