import streamlit as st
import uuid
from utils import document_loader,split_text,remove_extra_spaces,create_chunks,create_embeddings,create_vector_store,retrieval,tavily_fact_based_search,tavily_clinical_guidelines_search,tavily_safety_data_search,chat_completion,run_query
from langchain_groq import ChatGroq
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Configuring API Keys

tavily_api_key = st.secrets["TAVILY_API_KEY"]
groq_api_key = st.secrets["GROQ_API_KEY"]

# Streamlit app
import streamlit as st

st.title("App")

uploaded_doc = st.file_uploader("Upload patient records", type=["pdf", "docx", "txt"])

if uploaded_doc:
  pages=document_loader(uploaded_doc)
  text_splitter=split_text()
  # Extract text content from pages and concatenate
  full_text = ""
  for page in pages:
    full_text += page.page_content
  raw_text=remove_extra_spaces(full_text)
  chunks=create_chunks(raw_text,text_splitter)
  embeddings, text_contents=create_embeddings(chunks)
  index=create_vector_store(embeddings)

@tool
def query_patient_records(user_query):
  """Use this tool to search through the patient’s uploaded reports, genetic test results, or medical history stored in the vector database
     For Example: “Does this patient have any genetic marker for CYP2D6 metabolism issues?"""
  try:
    context=retrieval(index,user_query,text_contents)
    return context
  except Exception as e:
    return f"Search Error: ", str(e)

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
    tools = [tavily_fact_based_search,tavily_clinical_guidelines_search,tavily_safety_data_search,query_patient_records]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)
    return agent_executor, memory

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
                response = run_query(input_messages,config=config)

                # Extracting the final response
                if response and "messages" in response:
                    response_content = response["messages"][-1].content
                else:
                    response_content = "I couldn't process your request. Please try again."

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
    try:
        # Creating new thread ID for fresh conversation
        st.session_state.thread_id = str(uuid.uuid4())
    except Exception:
        pass
    st.rerun()

