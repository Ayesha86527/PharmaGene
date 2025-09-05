# 🧬🩺 PharmaGene - Prescribe With Care

**AI for Safer, Personalized Healthcare**  
PharmaGene is an AI-powered medical agent that helps clinicians reduce **medication errors** by analyzing patient records, prescriptions, and genetic markers.  
By combining **vector-based retrieval**, **trusted medical sources** (FDA, WHO, PubMed), and reasoning with **OpenAI GPT-OSS-120B**, PharmaGene delivers **real-time, explainable, and personalized prescribing insights**.

## ✨ Features
- 📂 **Patient Record Analysis** – Process reports, prescriptions, and health history.  
- 🧬 **Pharmacogenomic Checks** – Identify drug-gene interactions and contraindications.  
- 🔎 **Trusted Knowledge Retrieval** – Pulls evidence from FDA, WHO, and medical guidelines.  
- ⚡ **Real-Time Reasoning** – AI-powered insights using GPT-OSS-120B.  
- 💻 **Interactive Interface** – Lightweight Streamlit frontend for clinicians.

## 🚀 Live Demo  
👉 [Try PharmaGene on Streamlit](https://pharmagene-fahxxmrjfwhaqxwdtnkifw.streamlit.app/)  

## 🛠️ Tech Stack
- **Python** – Core language  
- **LangChain & LangGraph** – Tool orchestration and reasoning  
- **FAISS** – Vector database for patient record storage & retrieval  
- **Streamlit** – Web UI for interactive use  
- **OpenAI GPT-OSS-120B** – Core reasoning model

## ⚙️ Installation & Setup

Clone the repository:
```bash
git clone https://github.com/Ayesha86527/PharmaGene.git
cd PharmaGene

# Create a virtual environment & activate:
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows

#Install dependencies:
pip install -r requirements.txt

# Set up environment variables (create a .env file in the root directory):
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key

# Run the Streamlit app:
streamlit run app.py

# Your app will be live at:
👉 http://localhost:8501

