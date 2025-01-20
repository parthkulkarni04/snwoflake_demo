# Import required packages
import time
import streamlit as st
from snowflake.core import Root
from snowflake.core._common import CreateMode
from snowflake.snowpark import Session
from snowflake.cortex import Complete

# Create session with Root object to manage Snowflake objects
connection_params = {
    "account" : "HBB89474",
    "user" : "parthkulkarni04",
    "password" : "Sanjay@8520",
    "warehouse" : "rag_chatbot_wh",
    "database" : "rag_chatbot_db",
    "schema" : "rag_chatbot_db.rag_chatbot_schema",
}

session = Session.builder.configs(connection_params).create()
root = Root(session)

# Semantic search for chunks based on user query
def semantic_search(question):
    
    num_chunks = 5
    
    semantic_search_query = """
                            with results as 
                            (select url_link, VECTOR_COSINE_SIMILARITY(vector_db.chunk_vec,
                            SNOWFLAKE.CORTEX.EMBED_TEXT_768('e5-base-v2', ?)) as similarity, chunk
                            from vector_db
                            order by similarity desc
                            limit ?)
                            select chunk, similarity, url_link from results 
                            """
    df_context = session.sql(semantic_search_query, params=[question, num_chunks]).to_pandas() 

    if df_context._get_value(0, 'SIMILARITY') < 0.77:
        return "",""
        
    context_lenght = len(df_context) -1

    context_chunks = ""
    for i in range (0, context_lenght):
        context_chunks += df_context._get_value(i, 'CHUNK')
    url_link = df_context._get_value(0, 'URL_LINK')

    return context_chunks, url_link

# Get the chat history
def get_chat_history():

    slide_window = 10
    chat_history = []
    
    start_index = max(0, len(st.session_state.messages) - slide_window)
    for i in range (start_index , len(st.session_state.messages) -1):
         chat_history.append(st.session_state.messages[i])

    return chat_history

# Generate summary of the chat history for better context
def summarize_question_with_history(chat_history, question):
    summary_prompt = f"""
                      Based on the chat history below and the question, generate a query that extend the question
                      with the chat history provided. The query should be in natural language. 
                      Answer with only the query. Do not add any explanation.
                      
                      <chat_history>
                      {chat_history}
                      </chat_history>
                      <question>
                      {question}
                      </question>
                      """
    summary = Complete(root, summary_prompt) 
    return summary

# Call LLMs
def process_complete(question):

    chat_history = get_chat_history()
    if chat_history != []: 
        question_summary = summarize_question_with_history(chat_history, question)
        prompt_context, url_link =  semantic_search(question_summary)
    else:
        prompt_context, url_link = semantic_search(question) 

  
    prompt = f"""
              You are an expert assistant. 
              Extract information from the CONTEXT provided between <context> and </context> tags.
              You offer a chat experience considering the information included in the CHAT HISTORY
              provided between <chat_history> and </chat_history> tags..
              When answering the question contained between <question> and </question> tags
              be concise and do not hallucinate.
              If you do not have the information provided in the CONTEXT just say so. 
   
              KEEP THIS IN MIND: 
              Do not mention the CONTEXT used in your answer.
              Do not mention the CHAT HISTORY used in your answer.
              Do not add answer tag in your answer.
              
              <chat_history>
              {chat_history}
              </chat_history>
              <context>          
              {prompt_context}
              </context>
              <question>  
              {question}
              </question>
              Answer: 
              """
    if url_link == "":
        return "I don't have the information."
        
    response = Complete(session, prompt) + f'\n For more info visit - {url_link}'
    return response

# Stream response text
def stream_data(response):
    for word in response.split(' '):
        yield word + ' '
        time.sleep(0.02)

st.title('RAG Chatbot Built on Snowflake')

st.write("List of documents:")
docs = session.sql("ls @rag_chatbot_schema.wikipedia_mountain_assets").collect()
list_docs = []
for doc in docs:
    list_docs.append(doc["name"])
st.dataframe(list_docs, column_config={1:"Name"})

# Select Snowflake available models
st.sidebar.selectbox('Select your model:',(
                                        'snowflake-arctic',
                                        'mistral-7b',
                                        'mixtral-8x7b',
                                        'mistral-large',
                                        'llama3-8b',
                                        'llama3-70b',
                                        'llama3.1-8b',
                                        'llama3.1-70b',
                                        'llama3.1-405b',
                                        'reka-flash',
                                        'gemma-7b'), key="model")

# Clear conversation
st.sidebar.button("Clear Chat", key="clear_chat")

# Initialize chat history and clear chat option
if "messages" not in st.session_state or st.session_state.clear_chat:
        st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user query
if question := st.chat_input("Ask me"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({"role": "user", "content": question})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(stream_data(process_complete(question)))
        
    st.session_state.messages.append({"role": "assistant", "content": response})
