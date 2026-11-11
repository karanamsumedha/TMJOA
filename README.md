#  TMJ-OA Risk Prediction using Survival Analysis, RAG & Generative AI

##  Overview

This project presents an **intelligent AI system for predicting the progression risk of Temporomandibular Joint Osteoarthritis (TMJ-OA)** using a combination of:

*  Survival Analysis (Cox Model)
*  Explainable AI (feature contributions)
*  Retrieval-Augmented Generation (RAG)
*  Generative AI explanations
*  Agent-based pipeline

Unlike traditional models, this system not only predicts **risk** but also explains **why the risk exists** and supports it with **relevant medical evidence**.

---

## 🎯 Problem Statement

TMJ-OA is a **degenerative joint disorder** where early prediction of progression is crucial.

However:

* Clinical datasets are complex and high-dimensional
* Predictions are often **not interpretable**
* No integration of **AI reasoning + medical knowledge**

👉 This project solves that by building an **end-to-end explainable AI pipeline**.

---

## 🧠 Key Highlights

✔ Predicts TMJ progression risk
✔ Converts risk into **human-understandable explanations**
✔ Identifies **key contributing features**
✔ Retrieves **medical evidence using RAG**
✔ Supports **custom patient input**
✔ Implements a **multi-stage AI agent pipeline**
✔ Includes **confidence estimation + safety layer**

---

## 🏗️ System Pipeline

```id="pipeline-diagram"
Raw Clinical + Radiomic Data
           ↓
Data Preprocessing & Feature Engineering
           ↓
Survival Model (Cox Regression)
           ↓
Risk Score Prediction
           ↓
Feature Contribution Analysis
           ↓
RAG (Retrieve Medical Evidence)
           ↓
Generative AI Explanation
           ↓
Agent Pipeline Output
```

---

## 📂 Project Structure

```id="folder-structure"
IMPLEMENTATION2/
│
├── data/
│   ├── raw/                # Original datasets
│   ├── processed/          # Cleaned + merged data
│   └── knowledge/          # TMJ medical knowledge (RAG)
│
├── src/
│   ├── data_preprocessing.py
│   ├── survival_preprocessing.py
│   ├── cox_model.py
│   ├── shape_explainer.py
│   ├── genai_explainer.py
│   ├── rag_module.py
│   └── agent_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* 🐍 Python
* 📊 Pandas, NumPy
* 📈 Lifelines (Cox Model)
* 🤖 Generative AI (LLM-based logic)
* 🔎 FAISS (Vector Database)
* 🧠 LangChain (RAG pipeline)
* 📉 Matplotlib (visualization)

---

## 🚀 How It Works

### 1️⃣ Data Processing

* Merges clinical + radiomic datasets
* Cleans and standardizes features
* Creates survival variables (`event`, `time`)

---

### 2️⃣ Survival Modeling

* Uses **Cox Proportional Hazards Model**
* Predicts **relative risk of TMJ progression**

---

### 3️⃣ Explainability Layer

* Calculates **feature contributions**
* Identifies top risk factors per patient

---

### 4️⃣ RAG (Retrieval-Augmented Generation)

* Uses TMJ-related medical text
* Retrieves **relevant evidence for each feature**
* Ensures explanations are **grounded in knowledge**

---

### 5️⃣ Generative AI Layer

* Converts model outputs into:

  * Natural language explanations
  * Clinical-style reasoning

---

### 6️⃣ Agent Pipeline

* Combines all components:

  * Prediction
  * Explanation
  * Evidence retrieval
  * Safety filtering
* Produces final output

---

## 🧪 Running the Project

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 2: Preprocess Data

```bash
python src/data_preprocessing.py
```

---

### Step 3: Prepare Survival Dataset

```bash
python src/survival_preprocessing.py
```

---

### Step 4: Train Model

```bash
python src/cox_model.py
```

---

### Step 5: Run Full System

```bash
python src/agent_pipeline.py
```

---

## 🧾 Sample Output

```id="sample-output"
Patient ID: 1

Risk Score: 0.88
Risk Probability: 0.71
Confidence: MEDIUM

Explanation:
The patient is at moderate risk of TMJ progression due to bone degradation and joint dysfunction...

Evidence:
- Reduction in trabecular bone volume is linked to TMJ progression...
- Age-related degeneration increases structural damage...
```

---

## 🧑‍⚕️ Custom Patient Input

The system allows testing with new patient data:

```id="custom-input"
Enter age: 60
Enter pain score: 9
Jaw clicking: 1
...
```

👉 Outputs real-time prediction + explanation

---

## ⚠️ Limitations

* Small dataset size (~66 patients)
* Limited event distribution
* Uses selected features (not full dataset)
* Predictions are **relative risk (not diagnosis)**

---

## 🔮 Future Scope

* Integrate deep learning on medical imaging
* Expand dataset for higher accuracy
* Deploy as a web app (Streamlit)
* Fine-tune LLM for medical explanations
* Improve feature selection techniques

---

## 🎓 Academic Contribution

This project demonstrates:

* Survival Analysis in healthcare
* Explainable AI
* Retrieval-Augmented Generation (RAG)
* Agent-based AI systems

---

## 📜 Disclaimer

This system is for **educational and research purposes only**.
It is not intended for medical diagnosis or treatment.

---

## 👩‍💻 Authors

**Kavana H**
**Karanam Sumedha**
Generative AI & Applications Project (UE23CS342BA9)

---
