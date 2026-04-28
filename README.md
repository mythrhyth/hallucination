#  Detection and Benchmarking Reasoning Hallucinations in LLMs

This repository focuses on **detecting, injecting, and benchmarking hallucinations** in reasoning steps of **Large Language Models (LLMs)**. The project provides a modular framework for hallucination injection, detection, and evaluation — aiming to create interpretable and auditable LLM systems.

---

## 📂 Project Structure

```
Detection-and-Benchmarking-Reasoning-Hallucinations-in-LLMs/
│
├── Injection-Model/          # Module for hallucination injection into reasoning traces
│
├── detectionModel/           # Models and scripts for hallucination detection
│
├── app/                      # Backend app (e.g., FastAPI / Flask) to run detection/injection pipeline
│
├── data/                     # Datasets, JSON reasoning traces, and hallucination-labeled data
│
├── notebooks/                # Research notebooks for experiments and exploratory analysis
│
├── app_.py                   # Entry point for the web / API interface
│
├── __init__.py               # Marks directory as Python package
│
├── requirements.txt          # Python dependencies
│
├── README.md                 # Project documentation
│
├── .gitignore                # Ignored files and directories

```

---

## 🎯 Objective

To build a **benchmarking and detection system** that can:
1. **Inject** synthetic hallucinations into reasoning chains.
2. **Detect** hallucinations automatically using various ML/NLP models.
3. **Benchmark** detection performance across different LLMs and datasets.

---

## 🧩 Key Components

### 🔹 Injection Model
- Injects hallucinations based on **semantic drift**, **sentiment bias**, or **logical inconsistency**.
- Can be configured for controlled experiments (e.g., sentiment-based injection).

### 🔹 Detection Model
- Classifies reasoning steps as **faithful** or **hallucinated**.

- Includes **explainability metrics** and **attention visualization**.

### 🔹 Backend (app/)
- Provides FAST API for:
  - Uploading reasoning traces.
  - Running hallucination injection/detection.
  - Visualizing results.

### 🔹 Data
- JSON-based datasets with labeled reasoning chains.
- Supports datasets from reasoning benchmarks (e.g., GSM8K, CoT datasets).

### 🔹 Notebooks
- Experimental analysis and visualizations.
- Benchmarking results and evaluation metrics.

---

## ⚙️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/fuseai-fellowship/Detection-and-Benchmarking-Reasoning-Hallucinations-in-LLMs.git
cd Detection-and-Benchmarking-Reasoning-Hallucinations-in-LLMs
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the backend app:
```bash
python -m spacy download en_core_web_sm
python app.py
```

### Example workflow:
1. Upload reasoning traces via API or web UI.
2. Choose hallucination injection type (e.g., sentiment, logic).
3. Run detection model to identify injected or natural hallucinations.
4. View benchmarking metrics and visualizations.

---

## 📊 Evaluation Metrics
- **Precision, Recall, F1-score**
- **Faithfulness Score**
- **Logical Consistency**
- **Explainability Metrics (e.g., SHAP, Attention heatmaps)**

---

## 🔬 Research Focus

This repository is part of ongoing research on:
- **Reasoning hallucination taxonomy**
- **Auditable and explainable hallucination detection**
- **Sentiment-guided hallucination modeling**
- **Benchmark creation for reasoning LLMs**

---

## 🤝 Contributors
- [ISHWOR POKHREL RHYTHM BHETWAL/ SAJJAN ACHARYA]
- Special thanks to the Fusemachines AI Fellowship & Research team.


