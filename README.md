# 📰 Fake News Detection

A full-stack machine learning project that classifies news articles as **real** or **fake** using advanced **Natural Language Processing (NLP)** techniques. This end-to-end pipeline includes classical machine learning, transformer-based deep learning, data preprocessing, model evaluation, and a web-based user interface.

---

## 🚀 Project Overview

- **Goal**: Build a robust and scalable pipeline that can accurately detect fake news using both traditional and modern machine learning approaches.
- **Dataset**: [Kaggle – Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Approach**:
  - ✅ Classical ML with TF-IDF + Logistic Regression / Naive Bayes
  - ✅ Ensemble models like XGBoost and Random Forest
  - ✅ Advanced NLP with pretrained transformers (BERT)
  - ✅ Interactive web app using Streamlit for live predictions

---

## 🧱 Architecture Overview

```mermaid
graph TD;
    A[Raw Data (Kaggle)] --> B[Preprocessing (Cleaning, Tokenizing)];
    B --> C[Feature Engineering (TF-IDF / Embeddings)];
    C --> D[Model Training (LogReg / XGBoost / BERT)];
    D --> E[Evaluation (Metrics, Reports)];
    D --> F[App Interface (Streamlit)];
```

---

## 🗃️ Folder Structure

```
fake-news-detection/
├── data/
│   ├── raw/               # Original dataset from Kaggle
│   ├── interim/           # Combined and labeled dataset
│   └── processed/         # Cleaned and transformed data (TF-IDF, etc.)
│
├── notebooks/             # Jupyter notebooks for EDA, modeling, experimentation
├── models/                # Trained models and vectorizers
├── src/                   # Python modules for core logic
│   ├── data/              # Data loading and processing scripts
│   ├── features/          # Feature extraction methods
│   ├── models/            # Training and evaluation logic
│   └── utils/             # Logging, visualization, and helper functions
│
├── app/                   # Streamlit or Flask-based prediction app
├── scripts/               # CLI scripts for automation
├── tests/                 # Unit and integration tests
├── logs/                  # Training logs
├── reports/               # Evaluation outputs (graphs, reports)
├── configs/               # YAML configuration files
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 📊 Dataset Details

- **Source**: Kaggle ([Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset))
- **Files**:
  - `Fake.csv` – Articles labeled as fake
  - `True.csv` – Articles labeled as real
- **Columns**:
  - `title`, `text`, `subject`, `date`

---

## 🧠 Techniques & Tools

| Category              | Tools / Techniques                                  |
|-----------------------|------------------------------------------------------|
| Text Preprocessing    | NLTK, regex, stopword removal, lemmatization         |
| Feature Engineering   | TF-IDF, Bag-of-Words, BERT embeddings                |
| Models                | Logistic Regression, Naive Bayes, XGBoost, BERT      |
| Evaluation            | Accuracy, Precision, Recall, F1-score, ROC Curve     |
| Deployment            | Streamlit, Flask, Docker (optional)                  |
| Configuration         | YAML-based modular setup                             |
| Tracking & Logging    | Logging module, output reports, experiment logs      |

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/jobet1130/fake-news-detection.git
cd fake-news-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download the dataset manually from:
> https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Place `Fake.csv` and `True.csv` into the `data/raw/` directory.

---

## 🧪 Quick Start Guide

### Preprocess the Data
```bash
python scripts/preprocess_data.py
```

### Train the Model
```bash
python scripts/train.py --config configs/train_config.yaml
```

### Evaluate the Model
```bash
python scripts/evaluate.py
```

### Launch Web Application
```bash
cd app/
streamlit run app.py
```

---

## 📈 Example Output

Outputs are saved in the `reports/` folder:

- 📊 Confusion Matrix
- 📝 Classification Report
- 📉 ROC-AUC Curve
- 📚 Model comparison charts

---

## 🔐 Configuration Example (`train_config.yaml`)

```yaml
model:
  name: xgboost
  params:
    n_estimators: 200
    max_depth: 4
    learning_rate: 0.1

data:
  max_features: 5000
  test_size: 0.2
  random_state: 42

training:
  epochs: 10
  early_stopping: true
```

---

## 📌 Future Enhancements

- [ ] Integration with MLOps (e.g., MLflow or DVC)
- [ ] API deployment with FastAPI
- [ ] Incorporate news publisher metadata
- [ ] Real-time Twitter signal analysis
- [ ] Docker and Kubernetes deployment pipeline

---

## 👨‍💻 Contributors

- Jobet Casquejo – [`@jobet1130`](https://github.com/jobet1130)

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
