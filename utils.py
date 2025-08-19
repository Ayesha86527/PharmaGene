# Import relevant functionality
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pypdf
from langchain_groq import ChatGroq
from langchain.tools import tool
import numpy as np
import uuid
import streamlit as st

# Configuring API Keys

tavily_api_key = st.secrets["TAVILY_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Extraction function
def extract_search_results(raw_results):
    extracted_results = []
    for item in raw_results:
        # Combining key information
        structured_query = f"URL: {item.get('url', '')}\nTitle: {item.get('title', '')}\nContent: {item.get('content', '')}\n---\n"
        extracted_results.append(structured_query)
    return '\n'.join(extracted_results)

# Tool 1

# Create the original search object
fact_based_search = TavilySearch(
    search_depth="basic",
    max_results=1,
    tavily_api_key=tavily_api_key,
    include_domains=["clinpgx.org","cpicpgx.org/guidelines/","fda.gov/drugs/science-and-research-drugs/table-pharmacogenomic-biomarkers-drug-labeling"
    "go.drugbank.com"],
    include_raw_content=False,
    include_images=False,
    max_tokens=2000,
    include_answer=False
)

@tool
def tavily_fact_based_search(query: str) -> str:
    """Use this tool for core pharmacogenomics knowledge such as drug–gene interactions, genetic markers, and fundamental concepts.
       For Example: “How does CYP2C19 variation affect clopidogrel response?"""
    try:
        # Get raw results from Tavily
        result = fact_based_search.invoke({"query": query})
        raw_results = result.get('results', [])

        # Apply your extraction function
        extracted_data = extract_search_results(raw_results)
        return extracted_data

    except Exception as e:
        return f"Search error: {str(e)}"
    

# Tool 2

# Create the original search object
clinical_guidelines_search = TavilySearch(
    search_depth="basic",
    max_results=1,
    tavily_api_key=tavily_api_key,
    include_domains=["pubmed.ncbi.nlm.nih.gov","clinicaltrials.gov","nccn.org/guidelines/",
                     "who.int/groups/expert-committee-on-selection-and-use-of-essential-medicines/essential-medicines-lists"],
    include_raw_content=False,
    include_images=False,
    max_tokens=2000,
    include_answer=False
)

@tool
def tavily_clinical_guidelines_search(query: str) -> str:
    """Use this tool to retrieve clinical guidelines and official recommendations from trusted bodies like NCCN, WHO, ClinicalTrials.gov
      For Example: “What are the pharmacogenomic guidelines for warfarin dosing?"""
    try:
        # Get raw results from Tavily
        result = clinical_guidelines_search.invoke({"query": query})
        raw_results = result.get('results', [])

        # Apply your extraction function
        extracted_data = extract_search_results(raw_results)
        return extracted_data

    except Exception as e:
        return f"Search error: {str(e)}"

# Tool 3

# Create the original search object
safety_data_search = TavilySearch(
    search_depth="basic",
    max_results=1,
    tavily_api_key=tavily_api_key,
    include_domains=["fda.gov/drugs/fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard","medlineplus.gov/"],
    include_raw_content=False,
    include_images=False,
    max_tokens=2000,
    include_answer=False
)

@tool
def tavily_safety_data_search(query: str) -> str:
    """Use this tool to find real-world evidence, case studies, post-marketing safety data, and adverse drug reaction reports.
      For Example: “Are there any safety alerts about carbamazepine in Asian populations?"""
    try:
        # Get raw results from Tavily
        result = safety_data_search.invoke({"query": query})
        raw_results = result.get('results', [])

        # Apply your extraction function
        extracted_data = extract_search_results(raw_results)
        return extracted_data

    except Exception as e:
        return f"Search error: {str(e)}"


# Tool 4

embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def document_loader(pdf_filename):
    loader = PyPDFLoader(pdf_filename)
    pages = loader.load_and_split()
    return pages

def split_text():
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
  return text_splitter

def create_chunks(text,text_splitter):
    texts = text_splitter.create_documents([text])
    return texts

def remove_extra_spaces(text):
    raw_text=' '.join(text.split())
    return raw_text

def create_embeddings(chunks):
    text_contents = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(text_contents)
    return embeddings, text_contents

def create_vector_store(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.astype('float32'))
    return index

def retrieval(index, user_prompt, text_contents):
    query_embedding = embedding_model.encode([user_prompt])
    k = min(5, len(text_contents))
    distances, indices = index.search(query_embedding.astype('float32'), k)
    retrieved_info = [text_contents[idx] for idx in indices[0]]
    context = "\n".join(retrieved_info)
    return context


def chat_completion(user_input):
  input_message = [{
    "role": "system",
    "content": """You are a helpful pharmacogenomics assistant. Your role is to assist doctors and clinicians in prescribing the right medications to patients
based on their medical history, genetics, allergies, and conditions.

You have access to the following tools. Each tool has a specific purpose, and you should only use the tool(s) that directly match the query:

1. tavily_fact_based_search:
   - Use ONLY for core pharmacogenomics knowledge such as drug–gene interactions, genetic markers, and fundamental concepts.

2. tavily_clinical_guidelines_search:
   - Use ONLY to retrieve clinical guidelines and official recommendations from trusted bodies like NCCN, WHO, or FDA.

3. tavily_safety_data_search:
   - Use ONLY to find real-world evidence, case studies, post-marketing safety data, and adverse drug reaction reports.

4. query_patient_records:
   - Use ONLY to search through the patient’s uploaded reports, genetic test results, or medical history stored in the vector database.

Tool Usage Rules:
- Select the tool(s) strictly based on the type of question.
- If the query is about a patient’s specific data, check `query_patient_records` first.
- If the query is about general pharmacogenomics knowledge, use `tavily_fact_based_search`.
- If the query is about treatment standards or protocols, use `tavily_clinical_guidelines_search`.
- If the query is about risks, safety, or real-world usage, use `tavily_safety_data_search`.
- If multiple tools are clearly required, combine them in ONE step only. Never loop between tools.
- **IMPORTANT:** When using a tool, ensure the arguments are provided as a valid JSON object.
- If none of the tools match, respond:
  "I’m sorry, but I couldn’t find relevant information for [drug/gene/symptom/etc.]."

Output Guidelines:
- Always give structured, concise responses.
- Include source URLs from the tool output.
- Do NOT speculate outside pharmacogenomics or patient safety context.
- Do NOT re-query the same tool repeatedly for the same question.
- Always provide the exact source url with the answer from where you have found the answer.
- If the user asks the same question in a particular session which was asked before in the session that you do not need to use the tools again just return the
answer from the session memory.

Stop Condition:
- Once you have gathered enough from the selected tool(s) to answer, STOP and return the result.

"""
}, {
    "role": "user",
    "content": user_input,
}]
  return input_message








