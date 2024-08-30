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

load_dotenv()
# Access environment variables
LLM_api = os.getenv('GOOGLE_API_KEY')

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

# Function to get or reinitialize the vector store
def get_vector_store():
    return Chroma(persist_directory=path_db, embedding_function=embedder)

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
            ('system', '''You are an AI assistant designed to assist customers of Bookme.Pk through their mobile app or website. Your role is to provide clear, accurate, and relevant answers to user queries. Provide helpful and precise responses based on the information available to you.

If you are unable to provide an answer, suggest that the user contact our support team for further assistance. Only do this if you truly cannot provide the necessary information.

Always aim to assist the user efficiently, while remaining courteous and professional, do not mention any internal mechanisms or show the user that you are using any information source to answer their queries and try to make the response readable using bullets, headings, and sub-headings where needed. If the user asks for your introduction, tell them that you are BookieBee developed by Bookme.pk. Also, don't make up any facts or information on your own; strictly rely only on the database.
'''),

            ('human', 'Based on the following context, answer the query: {context}\nQuery: {query}')
        ]
    )
    return template, variables

# Function to fill in the prompt template with values
def fill_in_template(data):
    template, inputs = data
    final_prompt = template.invoke(inputs)
    return final_prompt

# Function to chunk text and add to the vector store with a timestamp
def process_and_store_text(file_path):
    vector_store = get_vector_store()  # Get the latest vector store instance

    # Read and process the text file
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

# Streamlit app starts here
st.title("Document Management System")

# Tabs for Admin and User panels
tab_admin, tab_user = st.tabs(["Admin", "User"])

# Admin Panel
with tab_admin:
    st.header("Admin Panel - Upload Documents")

    # File upload section
    uploaded_file = st.file_uploader("Choose a file to upload")

    if uploaded_file is not None:
        # Save the uploaded file
        file_path = os.path.join(uploaded_files_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Log the file upload
        log_file_upload(uploaded_file.name)

        # Process and store the text file content
        process_and_store_text(file_path)
        
        st.success("File uploaded and processed successfully!")

    # File retrieval section
    st.header("Retrieve Uploaded Files and Logs")
    if st.button("Retrieve Files and Logs"):
        try:
            # Get list of uploaded files
            files = os.listdir(uploaded_files_dir)
            
            # Display the uploaded files
            st.write("Uploaded Files:", files)
            
            # Display the log file
            with open(log_file_path, 'r') as log_file:
                logs = log_file.read()
            st.text_area("Upload Log", logs)
        except Exception as e:
            st.error(f"Error: {e}")

    # File download section
    st.header("Download a Specific File")
    filename_to_download = st.text_input("Enter the filename to download")

    if filename_to_download:
        file_path = os.path.join(uploaded_files_dir, filename_to_download)
        if os.path.exists(file_path):
            with open(file_path, "rb") as file:
                btn = st.download_button(
                    label="Download File",
                    data=file,
                    file_name=filename_to_download
                )
        else:
            st.error("File not found")

# User Panel
with tab_user:
    st.header("User Panel - Query the Database")

    # Query section
    query_text = st.text_input("Enter your query")

    if query_text:
        try:
            vector_store = get_vector_store()  # Get the latest vector store instance

            # Retrieve similar documents
            retriever = vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'k': 50, "score_threshold": 0.4}
            )
            similar_docs = retriever.invoke(query_text)

            # Sort documents by timestamp to prioritize recent ones
            recent_docs = sorted(similar_docs, key=lambda doc: doc.metadata.get("timestamp", "1970-01-01T00:00:00"), reverse=True)

            # Extract the most recent document (MR)
            if recent_docs:
                MR = recent_docs[0]
                W = recent_docs[1:]

                # Use LLM to determine which embeddings from W are valid or complement MR
                model = GoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=LLM_api
                )

                # Prepare variables for the LLM
                variables = {
                    'context': f"Initial document: {MR.page_content}\nOther documents: {[doc.page_content for doc in W]}",
                    'query': "Which of the other documents complement or make sense with the initial document?"
                }

                # Generate the prompt
                final_prompt = fill_in_template(create_prompt_template(variables))

                # Invoke the model and get the result
                result = model.invoke(final_prompt)
                
                # Process LLM output
                if isinstance(result, list):
                    # Expecting a list of valid indices or documents
                    valid_embeddings = [int(idx) for idx in result if idx.isdigit()]
                else:
                    st.error("Unexpected LLM output format. Expected a list of indices.")
                    valid_embeddings = []

                # Now, compare user's query with valid embeddings
                final_docs = [MR] + [W[int(idx)] for idx in valid_embeddings]
                context = " ".join([doc.page_content for doc in final_docs])

                # Call LLM again for the final response
                variables = {
                    'context': context,
                    'query': query_text
                }
                final_prompt = fill_in_template(create_prompt_template(variables))
                final_result = model.invoke(final_prompt)

                st.write("Response:", final_result)
            else:
                st.write("No documents found.")
            
        except ValueError as ve:
            st.error(f"ValueError: {ve}")
        except Exception as e:
            st.error(f"Error: {e}")
