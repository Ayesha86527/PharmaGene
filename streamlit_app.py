# Import relevant functionality
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import streamlit as st
import tempfile
import os
import uuid
from utilss import SYSTEM_MESSAGE,tavily_fact_based_search, tavily_clinical_guidelines_search, tavily_safety_data_search, load_patient_records, search_patient_records,document_loader,split_text,create_chunks,create_embeddings,create_vector_store,remove_extra_spaces

# Configuring API Keys

tavily_api_key = st.secrets["TAVILY_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]


st.title("PharmaGene")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize thread_id for this session (important for MemorySaver)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialize memory (but not the agent yet)
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

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
        load_patient_records(tmp_file_path)
        st.success("Document processed successfully!")
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
    
    finally:
        # Clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def get_agent_executor():
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=3000,
        timeout=None,
        max_retries=2,
        api_key=groq_api_key
    )
    tools = [tavily_fact_based_search, tavily_clinical_guidelines_search, tavily_safety_data_search, load_patient_records,search_patient_records]
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

