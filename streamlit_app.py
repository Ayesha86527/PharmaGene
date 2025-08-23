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
import streamlit as st
import tempfile
import os
import uuid

# Configuring API Keys
tavily_api_key = st.secrets["TAVILY_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

SYSTEM_MESSAGE = """You are a helpful pharmacogenomics assistant. You can search for drug information, clinical guidelines, safety data, and analyze patient records."""

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
    include_domains=["clinpgx.org","cpicpgx.org/guidelines/","fda.gov/drugs/science-and-research-drugs/table-pharmacogenomic-biomarkers-drug-labeling",
    "go.drugbank.com"],
    include_raw_content=False,
    include_images=False,
    max_tokens=2000,
    include_answer=False
)

@tool
def tavily_fact_based_search(query: str) -> str:
    """Use this tool for core pharmacogenomics knowledge such as drug–gene interactions, genetic markers, and fundamental concepts.
       For Example: "How does CYP2C19 variation affect clopidogrel response?"""
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
      For Example: "What are the pharmacogenomic guidelines for warfarin dosing?"""
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
      For Example: "Are there any safety alerts about carbamazepine in Asian populations?"""
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
# Global storage for vector databases (session-based)
VECTOR_STORAGE = {}
CURRENT_SESSION_KEY = None

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

def create_chunks(text, text_splitter):
    texts = text_splitter.create_documents([text])
    return texts

def remove_extra_spaces(text):
    raw_text = ' '.join(text.split())
    return raw_text

def create_embeddings(chunks):
    text_contents = [doc.page_content for doc in chunks]
    embeddings = embedding_model.encode(text_contents)
    return embeddings, text_contents

def create_vector_store(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index

def retrieval(index, user_prompt, text_contents):
    query_embedding = embedding_model.encode([user_prompt])
    k = 5
    distances, indices = index.search(query_embedding, k)
    retrieved_info = [text_contents[idx] for idx in indices[0]]
    context = "\n".join(retrieved_info)
    return context

# Functions to manage storage with session key
def set_current_session(session_key):
    """Set the current session key - call this from streamlit_app.py"""
    global CURRENT_SESSION_KEY
    CURRENT_SESSION_KEY = session_key

def get_current_session():
    global CURRENT_SESSION_KEY
    if CURRENT_SESSION_KEY is None:
        return "default"
    return CURRENT_SESSION_KEY

@tool
def load_patient_records(pdf_path: str) -> str:
    """Use this tool to load the patient's medical records and create vector database to store them as embeddings for retrieval"""
    try:
        pages = document_loader(pdf_path)
        full_text = ""
        for page in pages:
            full_text += remove_extra_spaces(page.page_content) + "\n"
        
        text_splitter = split_text()
        chunks = create_chunks(full_text, text_splitter)
        embeddings, text_contents = create_embeddings(chunks)
        
        # Store in global dict with current session key
        session_key = get_current_session()
        VECTOR_STORAGE[session_key] = {
            "vector_index": create_vector_store(embeddings),
            "document_contents": text_contents
        }
        
        return f"Patient records loaded successfully. Session: {session_key}"
    except Exception as e:
        return f"Error loading patient records: {str(e)}"

@tool
def search_patient_records(query: str) -> str:
    """Use this tool to search through the patient records."""
    session_key = get_current_session()
    
    if session_key not in VECTOR_STORAGE:
        return "No patient records loaded. Please upload and load patient records first."
    
    try:
        storage = VECTOR_STORAGE[session_key]
        context = retrieval(storage["vector_index"], query, storage["document_contents"])
        return f"Found information from patient records:\n{context}"
    except Exception as e:
        return f"Error searching patient records: {str(e)}"

st.title("PharmaGene")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize thread_id for this session (important for MemorySaver)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize memory (but not the agent yet)
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

# Create unique session key for this session
if "session_key" not in st.session_state:
    st.session_state.session_key = str(uuid.uuid4())

# Set the current session in utils.py so tools can access it
set_current_session(st.session_state.session_key)

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

uploaded_file = st.file_uploader("Upload the medical reports", type=["pdf", "docx", "txt"])

if uploaded_file is not None:
    try:
        # Get the correct file extension
        file_extension = os.path.splitext(uploaded_file.name)[1]
        
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension,  # Use actual file extension, not hardcoded .pdf
            prefix=uploaded_file.name.split('.')[0] + '_'
        ) as tmp_file:
            # Write the uploaded file content to temp file
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Process the document
        result = load_patient_records.invoke({"pdf_path": tmp_file_path})
        st.success(f"Status: {result}")
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def get_agent_executor():
    # Make sure session is set before creating agent
    set_current_session(st.session_state.session_key)
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=3000,
        timeout=None,
        max_retries=2,
        api_key=groq_api_key
    )
    tools = [tavily_fact_based_search, tavily_clinical_guidelines_search, tavily_safety_data_search, load_patient_records, search_patient_records]
    agent_executor = create_react_agent(model, tools, checkpointer=st.session_state.memory)
    return agent_executor

# Chat input 
if prompt := st.chat_input("Hey! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            try:
                # Get agent executor (not cached, so tools have access to current session state)
                agent_executor = get_agent_executor()
                
                # Configure for your agent
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                
                # Create the input for the agent
                input_messages = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ]
                
                # Invoke the agent
                response = agent_executor.invoke(
                    {"messages": input_messages}, 
                    config=config
                )
                
                # Extract the final response
                if response and "messages" in response:
                    response_content = response["messages"][-1].content
                else:
                    response_content = "I couldn't process your request. Please try again."
                
            except Exception as e:
                response_content = f"Something went wrong! Error Info: {str(e)}"
                st.error(response_content)
            
            # Display the response
            st.markdown(response_content)
            
            # Add assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response_content})

# Add a button to clear conversation history
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    try:
        # Create new thread ID for fresh conversation
        st.session_state.thread_id = str(uuid.uuid4())
    except Exception:
        pass
    st.rerun()


