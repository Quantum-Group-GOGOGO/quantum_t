# quantum_t

A quantitative-trading framework built with **Python** and **PyTorch**, designed for high-frequency futures data.  
It supports full-cycle development — data collection, preprocessing, model training (LSTM / MLP / Transformer), and back-testing — in a modular and reproducible way. Data collection needs to work with IB_gateway using a proper username and password.

---

## 🚀 Features

- **Data Collection**  
  Fetch and aggregate high-frequency futures data (e.g., 1-min bars) for training and back-testing.

- **Preprocessing Pipeline**  
  Clean, resample, and normalize market data with configurable window sizes and rolling statistics.

- **Model Architectures**  
  Implemented in PyTorch, including:
  - LSTM encoders for sequence learning  
  - MLP and Transformer layers for representation and prediction  
  - Custom loss functions for noisy, non-stationary data

- **Backtesting Engine**  
  Evaluate predictive models in a simulated futures environment with adjustable latency, spread, and execution parameters.

- **Modular Design**  
  Each component (data, model, training, evaluation) can run independently or be orchestrated through a unified pipeline.

---

## 📁 Project Structure

```
quantum_t/
├─ DataCollection/         # Data fetch & aggregation scripts
├─ DataPreprocess/         # Cleaning, feature engineering & normalization
├─ models/                 # PyTorch model definitions
├─ training/               # Training loop & experiment configs
├─ backtest/               # Backtesting & evaluation
├─ utils/                  # Common utilities & logging
├─ environment/            # Environment setup (requirements, Docker)
├─ Docs/                   # Documentation, experiment notes
└─ README.md               # Project overview
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Quantum-Group-GOGOGO/quantum_t.git
cd quantum_t

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🧠 Usage Example

### 1️⃣ Data Collection
```bash
python DataCollection/fetch_data.py --symbol NQ --bar_size '1 min' --days 30
```

### 2️⃣ Training
```bash
python training/train_model.py
```

### 3️⃣ Backtesting
```bash
python backtest/run_backtest.py --model checkpoints/model_latest.pt
```

---

## 🧩 Dependencies

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- pandas, numpy, matplotlib  
- tqdm, pyyaml, scikit-learn

(See `requirements.txt` for full list)

---

## 🤝 Author

Wentian Wang littlenova223@gmail.com

---

## 📜 License

This project is released under the **MIT License**.  
See the [`LICENSE`](LICENSE) file for details.

---


✳️ Maintainer: [Quantum Group](https://github.com/Quantum-Group-GOGOGO)
