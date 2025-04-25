# Quantifying Device Type and Handedness Biases in a Remote Parkinson's Disease AI-Powered Assessment

This project investigates algorithmic fairness in AI models for detecting Parkinson’s Disease based on interaction data collected through a remote assessment platform. It includes stages for data generation, model training, testing, and bias evaluation across demographic factors such as race, sex, device type, and dominant hand.

---

## 📁 Project Structure

```
quantifying-bias-remote-pd-website
   ├── analysis/                  # Output plots and figures
   ├── data/                      # Raw, Intermediate and Processed datasets
   ├── src/                       # Source code for models, fairness, and utilities
   ├── weights/                   # Saved model weights
   ├── config.py                  # Project configuration
   ├── main.py                    # Main CLI script
   ├── Part 1 - Data Preparation.ipynb
   ├── Part 2 - Training and Testing the AI Model.ipynb
   ├── Part 3 - Quantify Bias After Race Upsample.ipynb
   ├── Part 3 - Quantify Bias Before Race Upsample.ipynb
   ├── requirements.txt
```

---

## 🛠️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/zerinnasrintumpa/quantifying-bias-remote-pd-website.git
   cd quantifying-bias-remote-pd-website
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 How to Run the Project

All major actions can be run using the CLI interface via `main.py`:

### 📊 Generate Data

```bash
python main.py --generate-data
```

### 🤖 Train Models

```bash
python main.py --train-model
```

### 🧪 Test on Held-out Set

```bash
python main.py --test-model
```

### ⚖️ Evaluate Fairness

```bash
# Without race upsampling
python main.py --evaluate-model-bias race_upsample_false

# With race upsampling
python main.py --evaluate-model-bias race_upsample_true
```

---

## 📓 Notebooks

- `Part 1`: Extract and preprocess features
- `Part 2`: Train and evaluate ML models
- `Part 3`: Analyze fairness before and after race upsampling

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
