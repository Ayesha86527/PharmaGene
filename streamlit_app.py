# Import relevant functionality
import streamlit as st
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
import tempfile
import uuid
import os
from utils import(
    tavily_fact_based_search, 
    tavily_clinical_guidelines_search, 
    tavily_safety_data_search,
    document_loader,
    split_text,
    remove_extra_spaces,
    create_chunks,
    create_embeddings,
    create_vector_store,
    create_patient_records_retrieval_tool,
    SYSTEM_MESSAGE,
    groq_api_key
)


#-------------STREAMLIT APP-----------
st.title("🧬🩺 PharmaGene - Prescribe with Care")

uploaded_doc = st.file_uploader("Upload patient's medical records", type=["pdf", "docx", "txt"])

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize thread_id for this session 
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "text_contents" not in st.session_state:
    st.session_state.text_contents = None

if uploaded_doc and st.session_state.vector_store is None:
    try:
        # Get the correct file extension
        file_extension = os.path.splitext(uploaded_doc.name)[1]
        
        # Create a temporary file with the correct extension
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=file_extension,
            prefix=uploaded_doc.name.split('.')[0] + '_'
        ) as tmp_file:
            # Write the uploaded file content to temp file
            tmp_file.write(uploaded_doc.getvalue())
            tmp_file_path = tmp_file.name
            pages = document_loader(tmp_file_path)
            text_splitter = split_text()
            all_text = " ".join([page.page_content for page in pages])
            text=remove_extra_spaces(all_text)
            chunks = create_chunks(text, text_splitter)
            embeddings, text_contents = create_embeddings(chunks)
            vector_store = create_vector_store(embeddings)

            st.session_state.vector_store = vector_store
            st.session_state.text_contents = text_contents

    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
    finally:
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def get_agent_executor():
    memory = MemorySaver()
    model = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        max_tokens=3000,
        timeout=None,
        max_retries=2,
        api_key=groq_api_key
    )
    
    # Create tools list - include patient records tool if available
    tools = [tavily_fact_based_search, tavily_clinical_guidelines_search, tavily_safety_data_search]
    
    if st.session_state.vector_store is not None and st.session_state.text_contents is not None:
        search_patient_records = create_patient_records_retrieval_tool(
            st.session_state.vector_store, 
            st.session_state.text_contents
        )
        tools.append(search_patient_records)
    
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
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
                # Get agent executor with current session tools
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


