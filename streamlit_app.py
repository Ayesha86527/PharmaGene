import streamlit as st
import uuid
import tempfile
import os
from utils import document_loader,split_text,remove_extra_spaces,create_chunks,create_embeddings,create_vector_store,retrieval,tavily_fact_based_search,tavily_clinical_guidelines_search,tavily_safety_data_search,chat_completion
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Configuring API Keys
tavily_api_key = st.secrets["TAVILY_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

st.title("App")

uploaded_doc = st.file_uploader("Upload patient records", type=["pdf", "docx", "txt"])

# Initialize session state variables
if "document_processed" not in st.session_state:
    st.session_state.document_processed = False
if "index" not in st.session_state:
    st.session_state.index = None
if "text_contents" not in st.session_state:
    st.session_state.text_contents = []

if uploaded_doc and not st.session_state.document_processed:
    with st.spinner("Processing document..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_doc.getvalue())
                tmp_file_path = tmp_file.name
            
            pages = document_loader(tmp_file_path)
            text_splitter = split_text()
            
            # Extract text content from pages and concatenate
            full_text = ""
            for page in pages:
                full_text += page.page_content
            
            raw_text = remove_extra_spaces(full_text)
            chunks = create_chunks(raw_text, text_splitter)
            embeddings, text_contents = create_embeddings(chunks)
            index = create_vector_store(embeddings)
            
            # Store in session state
            st.session_state.index = index
            st.session_state.text_contents = text_contents
            st.session_state.document_processed = True
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
            st.success("Document processed successfully!")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

@tool
def query_patient_records(user_query: str) -> str:
    """Use this tool to search through the patient's uploaded reports, genetic test results, or medical history stored in the vector database
     For Example: "Does this patient have any genetic marker for CYP2D6 metabolism issues?"""
    try:
        if st.session_state.index is not None and st.session_state.text_contents:
            context = retrieval(st.session_state.index, user_query, st.session_state.text_contents)
            return context
        else:
            return "No patient records have been uploaded yet. Please upload a document first."
    except Exception as e:
        return f"Search Error: {str(e)}"

@st.cache_resource
def initialize_agent():
    memory = MemorySaver()
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=3000,
        timeout=None,
        max_retries=2,
        api_key=groq_api_key
    )
    tools = [tavily_fact_based_search, tavily_clinical_guidelines_search, tavily_safety_data_search, query_patient_records]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    return agent_executor, memory

agent_executor, memory = initialize_agent()

def run_query(input_message, config):
    try:
        result = agent_executor.invoke({"messages": input_message}, config)
        return result
    except Exception as e:
        st.error(f"Agent execution error: {str(e)}")
        return {"messages": [{"content": f"Error: {str(e)}"}]}

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if prompt := st.chat_input("Hey! How can I help you today?"):
    # Adding user message to session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process with agent
    with st.chat_message("assistant"):
        with st.spinner("Generating Response..."):
            try:
                # Configuring for your agent
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                # Creating the input for the agent
                input_messages = chat_completion(prompt)

                # Invoking the agent
                response = run_query(input_messages, config=config)

                # Extracting the final response - more robust approach
                response_content = "I couldn't process your request. Please try again."
                
                try:
                    if response and "messages" in response and len(response["messages"]) > 0:
                        final_message = response["messages"][-1]
                        
                        # Try different ways to extract content
                        if hasattr(final_message, 'content'):
                            response_content = final_message.content
                        elif isinstance(final_message, dict):
                            response_content = final_message.get('content', str(final_message))
                        else:
                            response_content = str(final_message)
                            
                except Exception as e:
                    response_content = f"Error extracting response: {str(e)}"

            except Exception as e:
                response_content = f"Something went wrong! Error Info: {str(e)}"
                st.error(response_content)

            # Displaying the response
            st.markdown(response_content)

            # Adding assistant response to session state
            st.session_state.messages.append({"role": "assistant", "content": response_content})

# Button to clear conversation history
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.document_processed = False
    st.session_state.index = None
    st.session_state.text_contents = []
    try:
        # Creating new thread ID for fresh conversation
        st.session_state.thread_id = str(uuid.uuid4())
    except Exception:
        pass
    st.rerun()





