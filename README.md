# Personalized AAC Assistant using RAG and LLaMA 3

**Authors**: Mohan Kakarla, Udit Brahmadevara    
**Domain**: Human-Centered AI | Natural Language Processing | Retrieval Augmented Generation | Assistive Communication | Personalization  

---

## 🧠 Abstract

This project explores an **Augmentative and Alternative Communication (AAC)** system powered by a **Retrieval-Augmented Generation (RAG)** pipeline and a **fine-tuned LLaMA 3 8B-Instruct** model. It is designed to support individuals with limited verbal communication by suggesting **personalized, context-aware responses** that reflect their unique persona and communication style.



## 🎯 Project Objectives

- 🧠 Aid non-speaking individuals by generating meaningful verbal suggestions based on intent and personal communication profiles.
- 🔎 Use **RAG** to retrieve relevant past narratives or experiences to ground model outputs.
- 🤖 Employ a **fine-tuned LLaMA 3 8B-Instruct** LLM to ensure fluent, coherent, and identity-preserving communication.
- 🧩 Build an accessible, modular, and visually intuitive AAC interface using **Streamlit**.

---

## 🧰 Core Technologies

| Component             | Description                                                               |
|-----------------------|---------------------------------------------------------------------------|
| 🧠 LLM Backbone        | Fine-tuned LLaMA 3 8B-Instruct model (hosted externally on Kaggle)        |
| 🧩 RAG Framework       | FAISS-based similarity search with Sentence-BERT embeddings               |
| 📄 User Profiles       | JSON-based profiles storing communication style, history, tone, summary  |
| 💬 Frontend            | Streamlit chat UI with multiple responses and user-influence inputs       |
| 🗃️ Text Chunking       | RecursiveCharacterTextSplitter with dynamic chunk size and overlap        |

---

## 🔍 Key Features

- 👥 **Multiple user personas**: Select between different tones (empathetic, humorous, concise, etc.).
- 🧾 **Customizable prompts**: Add "influence" like *Be humorous*, *Ask a question*, *Agree*.
- 🔁 **Multiple LLM responses**: Generate and choose from multiple options for each input.
- ⚙️ **Chunked RAG retrieval**: Customize chunking for better retrieval of personal context.
- 🛠️ **Debugging panel**: Inspect prompt formatting and retrieved narratives live.

---
## 🏗️ System Architecture
![System Architecture](media/architecture.png) 
- **Frontend**: Streamlit interface
   ![UI Screenshot](media/ui.png)
- **Backend**:  
  - Finetuned LLaMA model (loaded via HuggingFace or local path)  
  - FAISS-based vector store to store and retrieve user-specific memory  
  - Personalization layer (prompt engineering with profiles)  
- **Data Format**: JSON file for user histories Refer to [`user_profiles.json`](user_profiles.json) for:  

   
  

---
## 📄 Academic Report

Refer to [`report.pdf`](report.pdf) for:

- Background on AAC systems and user-centered design.
- Model fine-tuning pipeline, hyperparameters, and evaluation.
- Dataset preparation and tokenizer configuration.
- System usability insights, limitations, and future work.

---

## 🗂️ File Structure

```
├── app.py                  # Streamlit web app (production entrypoint)
├── Code.ipynb              # Core development notebook
├── Training.ipynb          # Fine-tuning LLaMA 3 notebook
├── user_profiles.json      # Communication style + memory settings
├── report.pdf              # Academic report
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
├── Demo.mp4                # Walkthrough video
└── media/
    ├── architecture.png    # System architecture diagram
    └── ui.png              # UI screenshot
```

---

## 💾 Finetuned Model

🚨 Due to GitHub storage limits, the fine-tuned LLaMA 3 8B-Instruct model is hosted externally:

📥 [Download Finetuned Model (~5GB)](https://www.kaggle.com/datasets/mohankumarkakarla/finetuned/data)

> After downloading, place the model inside:

```
./models/llm_finetuned/
```

Or for **Kaggle/Streamlit Cloud**, set `MERGED_MODEL_PATH` in `app.py` to the model directory location.

---

## 🚀 Running the App

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit frontend

```bash
streamlit run app.py
```

> ✅ **Note**: If running inside a notebook (e.g., Kaggle), the last cell in `Code.ipynb` should launch:

```python
!streamlit run /kaggle/input/stream/fast_app.py
```

---

## 📜 License

This project is open-sourced for **academic and research purposes only**. Please refer to [`LICENSE`](LICENSE) for more information.

---
