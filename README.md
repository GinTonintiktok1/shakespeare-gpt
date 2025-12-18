# 🎭 Shakespeare GPT

AI-powered Shakespeare text generator with Streamlit UI.

## 🚀 Quick Start

\\\ash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
\\\

## 📊 Model Info

- **Parameters**: 1.26M
- **Architecture**: 6-layer Transformer
- **Train Loss**: 1.12
- **Val Loss**: 1.50
- **Speed**: ~40 char/sec (CPU)

## 🎬 Example Output

\\\
ROMEO:
O, that I were a glove upon that hand,
That I might touch that cheek!
\\\

## 🛠️ Tech Stack

- PyTorch 2.0
- Streamlit 1.28
- Python 3.10+


## 🧠 Training

### Train From Scratch

The complete training pipeline is available in the `notebooks/` folder.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GinTonintiktok1/shakespeare-gpt/blob/master/notebooks/training_shakespeare_gpt.ipynb)

**Training specs:**
- Dataset: Shakespeare complete works (~1.1M chars)
- Hardware: Google Colab (free T4 GPU)
- Duration: ~3 hours
- Optimizer: AdamW (lr=3e-4)
- Epochs: 5
- Batch size: 64

**Results:**
- Train loss: 1.12
- Validation loss: 1.50
- Model size: 1.26M parameters (5 MB)

### Reproduce Training


