import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
import os
from datetime import datetime
from dotenv import load_dotenv
from merged import executor
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool


import streamlit as st
# ADDITION 0)
from google.cloud import storage
import tempfile

# ADDITION 1)
GCP_BUCKET_NAME = 'bookiebev4bucket' 
if not GCP_BUCKET_NAME:
    raise ValueError("GCP_BUCKET_NAME is not set.")

GCP_VECTOR_STORE_BLOB = 'Vector Databases/Chroma1/chroma.sqlite3'  # Example path in GCP bucket

# ADDITION 2)
def download_vector_store(bucket_name, blob_name):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Create a temporary directory to store the downloaded vector store
    temp_dir = tempfile.mkdtemp()
    chroma_dir = os.path.join(temp_dir, 'Chroma1')
    os.makedirs(chroma_dir, exist_ok=True)  # Create Chroma1 directory inside temp_dir

    # Full path to save the downloaded vector store file
    vector_store_path = os.path.join(chroma_dir, 'chroma.sqlite3')

    # Download the vector store
    blob.download_to_filename(vector_store_path)
    return chroma_dir  # Return the path to the directory containing vector store

# ADDITION 3)
# Function to upload vector store back to GCP bucket
def upload_vector_store(bucket_name, blob_name, vector_store_path):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Full path of the SQLite file to upload
    sqlite_file_path = os.path.join(vector_store_path, 'chroma.sqlite3')
    blob.upload_from_filename(sqlite_file_path)

# ADDITION 4)
# Function to get or reinitialize the vector store
def get_vector_store():
    vector_store_path = download_vector_store(GCP_BUCKET_NAME, GCP_VECTOR_STORE_BLOB)
    return Chroma(persist_directory=vector_store_path, embedding_function=embedder), vector_store_path

# ADDITION 5)
# Function to persist vector store back to GCP bucket
def persist_vector_store(vector_store, vector_store_path):
    vector_store.persist()  # Persist changes locally first
    upload_vector_store(GCP_BUCKET_NAME, GCP_VECTOR_STORE_BLOB, vector_store_path)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Maintains the conversational context

role = """ 1) You are BookieBee, an agent-based chatbot for Bookme.pk, designed to assist users with a variety of queries related to services offered by the company. Bookme.pk provides services including bus ticket booking, cinema ticketing, airline ticket reservations, and event ticket sales. Use the available tools to provide accurate, up-to-date, and helpful responses regarding these services, ensuring a seamless customer experience.
            Even if the query isn't regarding the services, try your best to answer it using "StaticTool"
            2) Try to figure out inputs/arguments for the tools you need to use from the user's given query and if they are not found there, ask them from the user.
            3) Use current date and time as a reference to identify dates/times from the user's queries. To get sure about the current date/time, use the provided tool "today_date".
            4) Always introduce yourself as BookieBee, making the conversation engaging and friendly.
            5) Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.
            6) Format your responses to make them easily readable and understandable.
            """

memory.chat_memory.add_message(SystemMessage(content=role))
load_dotenv()
# Access environment variables
LLM_api = os.getenv('GOOGLE_API_KEY')


advisor = GoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=LLM_api
)

# Set up the directory for ChromaDB
current_dir = os.path.dirname(os.path.abspath(__file__))
path_db = os.path.join(current_dir, 'Vector Databases', 'Chroma1')

# Directories for storing uploaded files and logs
uploaded_files_dir = os.path.join(current_dir, 'Uploaded Files')
log_file_path = os.path.join(current_dir, 'upload_log.txt')

# Ensure directories exist
os.makedirs(uploaded_files_dir, exist_ok=True)
os.makedirs(path_db, exist_ok=True)

# Initialize the embedder
embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=LLM_api
)

# Global cache for context
context_cache = {}

# Function to log file upload
def log_file_upload(filename):
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{filename} - {datetime.now().isoformat()}\n")

# Function to create the prompt template
def create_prompt_template(variables):
    template = ChatPromptTemplate.from_messages(
        [
            ('system', '''You are BookieBee, a helpful assistant developed by Bookme.pk to assist users with booking-related queries on the Bookme website and mobile app. Your main goals are:

Guide users professionally and efficiently with responses based solely on the knowledge and context you have access to.
Do not reveal or mention any internal mechanisms, such as a knowledge base or context being used.
If a query isn't related to Company's information and your role, don't start going into its details or guiding with it and take it as a joke.
Always introduce yourself as BookieBee, making the conversation engaging and friendly.
Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.'''),

            ('human', 'Based on the following context:{context}, Answer Query: "{query}".')
        ]
    )
    return template, variables

def create_advisor_template(variables):
    template = ChatPromptTemplate.from_messages(
        [
            ('system', '''Give zero shot answer, just give a list of what's asked without any extra information.'''),

            ('human', '''Which of the following documents from {Other_docs} complement {recent_captured}.''')
        ]
    )
    return template, variables

# Function to fill in the prompt template with values
def fill_in_template(data):
    template, inputs = data
    final_prompt = template.invoke(inputs)
    return final_prompt

# Function to chunk text and add to the vector store with a timestamp
# Function to chunk text and add to the vector store with a timestamp
def process_and_store_text(file_path):
    vector_store, vectore_store_path = get_vector_store()  # Get the latest vector store instance

    # Read and process the text file.
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # Get the current timestamp
    timestamp = datetime.now().isoformat()

    # Create a Document for each chunk, including the timestamp in metadata
    docs = [Document(page_content=chunk, metadata={"timestamp": timestamp}) for chunk in chunks]
    vector_store.add_documents(docs)
    vector_store.persist()

    # Persist and upload to cloud
    persist_vector_store(vector_store, vectore_store_path)


def clear_and_recreate_vector_database():
    # Load and clear the existing local vector store
    vector_store, vector_store_path = get_vector_store()
    vector_store.delete_collection()  # Clears all data in the local vector database
    vector_store.persist()  # Save the cleared state locally

    # Delete the existing vector database file in GCP
    client = storage.Client()
    bucket = client.get_bucket(GCP_BUCKET_NAME)
    blob = bucket.blob(GCP_VECTOR_STORE_BLOB)
    blob.delete()  # Remove the cloud version

    # Recreate a new local vector store instance
    os.makedirs(vector_store_path, exist_ok=True)
    new_vector_store = Chroma(persist_directory=vector_store_path, embedding_function=embedder)
    new_vector_store.persist()  # Persist the new empty vector store locally

    # Upload the new empty vector store to GCP
    upload_vector_store(GCP_BUCKET_NAME, GCP_VECTOR_STORE_BLOB, vector_store_path)

    # Log the deletion action
    log_deletion_action()

    st.success("Local and GCP vector databases have been cleared and recreated successfully.")

# Function to log deletion actions
def log_deletion_action():
    # Define log entry format
    log_entry = datetime.now().strftime("%I:%M %p, %B %d, %Y") + " - Vector database deleted and recreated.\n"

    # Append log entry to the local delete-logs.txt file
    log_file_path = os.path.join("delete-logs.txt")
    with open(log_file_path, 'a') as log_file:
        log_file.write(log_entry)

    # Upload the updated log file to GCP
    client = storage.Client()
    bucket = client.get_bucket(GCP_BUCKET_NAME)
    blob = bucket.blob("delete-logs.txt")
    blob.upload_from_filename(log_file_path)



#------------------------------------USER INTERFACE-------------------------------

# Tabs for Admin and User panels
tab_admin, tab_user = st.tabs(["Admin", "User"])

# Admin Panel
with tab_admin:
    st.header("Admin Panel - Upload Documents")

   

    # File upload section (added unique key)
    uploaded_file = st.file_uploader("Choose a file to upload", key="admin_file_uploader")

    if uploaded_file is not None:
        # Save the uploaded file
        uploaded_files_dir = "Uploaded Files"  # Directory where uploaded files will be saved
        file_path = os.path.join(uploaded_files_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Log the file upload
        log_file_upload(uploaded_file.name)

        # Process and store the text file content
        process_and_store_text(file_path)
        
        st.success("File uploaded and processed successfully!")

    confirm_clear = st.checkbox("Confirm to clear all data in both local and GCP vector databases (This action is irreversible)")

    if confirm_clear and st.button("Clear Local and GCP Vector Database"):
        clear_and_recreate_vector_database()
    # Download button for delete-logs.txt
    if st.button("Download Delete Logs"):
        client = storage.Client()
        bucket = client.get_bucket(GCP_BUCKET_NAME)
        blob = bucket.blob("delete-logs.txt")

        # Download the log file
        with open("delete-logs.txt", "wb") as log_file:
            blob.download_to_file(log_file)

        with open("delete-logs.txt", "r") as log_file:
            st.download_button("Click here to download delete-logs.txt", log_file, file_name="delete-logs.txt")
    
# User Panel
with tab_user:
    st.header("User Panel - Query the Database")

    # Query section
   
    def clear_input():
        st.session_state["input_text"] = ""  # Reset the input field to an empty string

    # Create the input field with a default value
    query_text = st.text_input("Enter something")

    

    if query_text:
        memory.chat_memory.add_message(HumanMessage(content=query_text))
    try:


        response = executor.invoke(
            {"input": query_text}
        )
        st.write(f"Bot: {response['output']}")
        memory.chat_memory.add_message(AIMessage(content=response['output']))
    except KeyError as e:
    # Handle a specific exception if 'output' key is not in the response
        st.write(f"KeyError: {e}")
        

    except Exception as e:
    # Catch all other exceptions
        #SEE THE ERROR 
        #st.write(e)
        
        st.write(f"😢 Iam having trouble at the moment serving your quey, try again in a while maybe?")
       
        
                
        

    