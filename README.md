# ViAMR: Fine-tuning LLMs for Abstract Meaning Representation in Vietnamese

🇻🇳 **Vietnamese AMR Parser for VLSP 2025 Competition**

## 📍 Overview

This project implements a Vietnamese Abstract Meaning Representation (AMR) parser developed for the VLSP 2025 competition. The system converts Vietnamese sentences into their semantic AMR representations using state-of-the-art language models with supervised fine-tuning (SFT) and reinforcement learning approaches (GRPO).

## 🎯 Features

* **Vietnamese AMR Parsing**: Convert Vietnamese sentences to PENMAN-format AMR graphs
* **Multiple Training Approaches**:
  * Supervised Fine-Tuning (SFT)
  * Group Relative Policy Optimization (GRPO) with reinforcement learning
* **Advanced Post-processing**: Comprehensive AMR validation and correction
* **Evaluation Metrics**: Automated scoring and evaluation system
* **DeepSpeed Integration**: Efficient training with ZeRO optimization

## 🏗️ Architecture

```
VLSP2025/amr/src/
├── main.py                 # Main inference pipeline
├── infer.py               # Model inference utilities
├── data_loader.py         # Data loading and preprocessing
├── data_processing.py     # Advanced data processing
├── train_sft.py          # Supervised fine-tuning
├── train_grpo.py         # GRPO reinforcement learning training
├── postprocessing.py     # AMR validation and correction
├── prompt.py             # System prompts and templates
├── reward.py             # Reward functions for RL training
├── get_score.py          # Evaluation and scoring
├── config/               # Training configurations
│   └── ds_zero2.json     # DeepSpeed ZeRO stage 2 config
└── scripts/              # Training and inference scripts
    ├── train_sft.sh      # SFT training script
    ├── train_grpo.sh     # GRPO training script
    ├── infer.sh          # Inference script
    ├── get_score.sh      # Evaluation script
    └── main.sh           # Main pipeline script
```

## 🚀 Setup and Usage

### 1. Installation

```bash
# Navigate to the AMR source directory
cd VLSP2025/amr/src

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Process and split training data
python data_processing.py
python split_train_test.py
```

### 3. Training Models

#### Supervised Fine-Tuning (SFT)

```bash
# Train with supervised fine-tuning
bash scripts/train_sft.sh
```

#### GRPO Reinforcement Learning

```bash
# Train with Group Relative Policy Optimization
bash scripts/train_grpo.sh
```

### 4. Inference

```bash
# Run AMR parsing inference
bash scripts/infer.sh

# Or run the main pipeline
bash scripts/main.sh
```

### 5. Evaluation

```bash
# Evaluate model performance
bash scripts/get_score.sh
```

## 📊 Key Components

### AMR Parser ([`infer.py`](src/infer.py))

The main parsing component using [`QwenReasoner`](src/infer.py) class:

```python
class QwenReasoner:
    def inference(self, prompt: str, max_new_tokens: int = 2048, is_extract_amr: bool = False) -> str
```

### Post-processing ([`postprocessing.py`](src/postprocessing.py))

Advanced AMR validation and correction functions:

* [`remove_single_prop_nodes`](src/postprocessing.py) - Remove single property nodes
* [`has_duplicate_nodes`](src/postprocessing.py) - Check for duplicate variable names
* [`dedup_and_tidy`](src/postprocessing.py) - Remove duplicate roles and clean formatting
* [`balance_parens`](src/postprocessing.py) - Fix parentheses balance
* [`fix_amr_vars`](src/postprocessing.py) - Correct variable declarations

### Prompting System ([`prompt.py`](src/prompt.py))

Structured prompts with Vietnamese-specific instructions:

```python
SYSTEM_PROMPT = '''
Bạn là một mô hình ngôn ngữ lớn chuyên về phân tích cú pháp ngữ nghĩa cho tiếng Việt. 
Nhiệm vụ của bạn là chuyển đổi một câu tiếng Việt đầu vào thành biểu diễn AMR hoàn chỉnh.
'''
```

## 🛠️ Configuration

### Training Configuration

* **DeepSpeed**: [`config/ds_zero2.json`](src/config/ds_zero2.json) - ZeRO stage 2 optimization
* **Model Support**: Qwen2.5, LLaMA3, and other transformer models
* **RL Training**: GRPO algorithm with custom reward functions

### Key Parameters

* **Max Sequence Length**: 2048 tokens
* **Training Approaches**: SFT + GRPO reinforcement learning
* **Output Format**: PENMAN notation AMR graphs
* **Language**: Vietnamese with underthesea tokenization

## 📈 Model Training

### Supervised Fine-Tuning

Uses [`train_sft.py`](src/train_sft.py) to train the model on Vietnamese sentence-AMR pairs with standard cross-entropy loss.

### Reinforcement Learning (GRPO)

Uses [`train_grpo.py`](src/train_grpo.py) with:
* Custom reward functions from [`reward.py`](src/reward.py)
* Group Relative Policy Optimization
* AMR quality-based rewards

## 🔍 Evaluation

The evaluation system ([`get_score.py`](src/get_score.py)) provides:
* AMR graph accuracy metrics
* Semantic similarity scoring
* Structure validation checks
* Performance benchmarking

## 📝 Usage Example

```python
from infer import QwenReasoner
from postprocessing import process_amr_general

# Initialize the AMR parser
reasoner = QwenReasoner(model_path="path/to/model")

# Parse Vietnamese sentence to AMR
sentence = "Tôi đang học tiếng Việt."
amr_result = reasoner.inference(sentence)

# Post-process the result
cleaned_amr = process_amr_general(amr_result)
print(cleaned_amr)
```

## 🤝 Contributing

This project is developed for the VLSP 2025 competition. The system focuses on Vietnamese language processing and AMR semantic representation.

## 📚 References

* Vietnamese Language Processing
* Abstract Meaning Representation (AMR)
* PENMAN Notation
* Group Relative Policy Optimization (GRPO)
