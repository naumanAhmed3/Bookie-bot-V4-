import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime
from dotenv import load_dotenv
from langchain.tools import tool
import streamlit as st

# ADDITION 0)
from google.cloud import storage
import tempfile

# ADDITION 1)
GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME')
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

load_dotenv()
# Access environment variables
LLM_api = os.getenv('GOOGLE_API_KEY')

teller = GoogleGenerativeAI(
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

# Function to get or reinitialize the vector store

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
If a query isn't related to Company's information and your role , don't start going into it's details or guiding with it and take it as a joke.
Always introduce yourself as BookieBee,only when asked or greeted!
Don't give your introduction for no reason with every response.
Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.'''),

            ('human', 'Based on the following context:{context} , Answer Query: "{query}".')
        ]
    )
    return template, variables

@tool
def staticTool(query):
    """
    Fallback tool for handling queries not directly related to services offered by Bookme.Pk.

    Usage Guidelines:
    - This tool should only be invoked when the query is not recognized as relevant to Bookme.Pk's services.
    - If the tool has already been used once for a particular query, it should not be called again to avoid a loop.
    - Use this tool as a one-time fallback, and use the information as it is without modifying the returned information.
    - Do not re-invoke the tool for the same query if it fails to produce a relevant result; instead, notify the user politely.

    Parameters:
    - query (str): The user's query that requires an answer.

    Returns:
    - List[str]: A list containing elements:
        1. The most relevant document's content (MR).
        2. The advised response generated by the GoogleGenerativeAI model.

    Important Considerations:
    - If the `staticTool` is used and returns an answer, the answer must be presented as-is to the user.
    - Avoid redundant calls to the tool for the same or similar queries within the same session.
    - Ensure proper error handling to gracefully handle cases where the tool is unable to provide a satisfactory response.

    Example Scenario:
    - The user enters a general question unrelated to bookings (e.g., "What's the capital of France?").
    - If no relevant response can be found using the Bookme.Pk-specific logic, `staticTool` can be invoked.
    - The response from `staticTool` should then be returned to the user directly without further processing.

    """
    advisor = GoogleGenerativeAI(
        model="gemini-1.5-pro",
        google_api_key=LLM_api
    )
    query_text = query.lower()
    if query_text == 'exit':
        return "Exiting..."

    try:
        vector_store, vector_store_path = get_vector_store()  # Get the latest vector store instance

        # Retrieve similar documents
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'k': 50, "score_threshold": 0.4}
        )
        similar_docs = retriever.invoke(query_text)

        # Sort documents by timestamp to prioritize recent ones
        recent_docs = sorted(similar_docs, key=lambda doc: doc.metadata.get("timestamp", "1970-01-01T00:00:00"), reverse=True)

        if recent_docs:
            MR = recent_docs[0].page_content
            W = recent_docs[1:]

            variables = {'Other_docs': [doc.page_content for doc in W], 'recent_captured': MR}
            final_prompt = fill_in_template(create_advisor_template(variables))
            advised = advisor.invoke(final_prompt)

            return [MR, advised]

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def create_advisor_template(variables):
    template = ChatPromptTemplate.from_messages(
        [
            ('system', '''Give zero shot answer , just give list of what's asked without any extra information. You may also refer to Bookme.Pk website to add to the list.'''),

            ('human' , '''Which of the following documents from {Other_docs} complement {recent_captured}.''')
        ]
    )
    return template, variables

# Function to fill in the prompt template with values
def fill_in_template(data):
    template, inputs = data
    final_prompt = template.invoke(inputs)
    return final_prompt

# Function to chunk text and add to the vector store with a timestamp
def process_and_store_text(uploaded_file):
    vector_store, vector_store_path = get_vector_store()  # Get the latest vector store instance

    # Since uploaded_file is a file-like object from Streamlit, get its content
    text = uploaded_file.getvalue().decode('utf-8')  # Convert binary data to string

    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # Get the current timestamp
    timestamp = datetime.now().isoformat()

    # Create a Document for each chunk, including the timestamp in metadata
    docs = [Document(page_content=chunk, metadata={"timestamp": timestamp}) for chunk in chunks]
    
    # Add documents to the vector store and persist it
    vector_store.add_documents(docs)
    vector_store.persist()

    print("File processed and embeddings created successfully.")

# Directory to save uploaded files
uploaded_files_dir = "uploaded_files"

def upload_file(uploaded_file):
    # Check if the file is uploaded
    if uploaded_file is not None:
        # Get file name from uploaded file object
        file_name = uploaded_file.name
        
        # Read the content of the uploaded file (in memory)
        file_content = uploaded_file.read()

        # Ensure the directory for saving files exists
        if not os.path.exists(uploaded_files_dir):
            os.makedirs(uploaded_files_dir)

        # Define the path to save the uploaded file
        save_path = os.path.join(uploaded_files_dir, file_name)

        # Save the uploaded file to the specified path
        with open(save_path, 'wb') as f:
            f.write(file_content)

        # Log the file upload (you can define this function)
        log_file_upload(file_name)

        # Process and store the text file content (assuming this function processes embeddings or similar)
        process_and_store_text(save_path)

        print("File uploaded and processed successfully, and embeddings created.")
        return "File uploaded successfully!"
    else:
        return "No file uploaded."

# Command line interaction starts here
def main():
    while True:
        print("\nOptions:")
        print("1. Upload a file")
        print("2. Retrieve uploaded files and logs")
        print("3. Download a specific file")
        print("4. Query the database")
        print("5. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            file_path = input("Enter the path of the file to upload: ")
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                with open(file_path, 'rb') as file:
                    file_content = file.read()

                # Save the uploaded file
                save_path = os.path.join(uploaded_files_dir, file_name)
                with open(save_path, 'wb') as f:
                    f.write(file_content)

                # Log the file upload
                log_file_upload(file_name)

                # Process and store the text file content
                process_and_store_text(save_path)

                print("File uploaded and processed successfully!")
            else:
                print("Error: File not found.")

        elif choice == '2':
            try:
                files = os.listdir(uploaded_files_dir)
                print("\nUploaded Files:", files)

                with open(log_file_path, 'r') as log_file:
                    logs = log_file.read()
                print("\nUpload Log:\n", logs)
            except Exception as e:
                print(f"Error: {e}")

        elif choice == '3':
            filename_to_download = input("Enter the filename to download: ")
            file_path = os.path.join(uploaded_files_dir, filename_to_download)
            if os.path.exists(file_path):
                with open(file_path, "rb") as file:
                    print("File content:\n")
                    print(file.read().decode('utf-8'))
            else:
                print("Error: File not found.")

        elif choice == '4':
            staticTool()
        elif choice == '5':
            print("Exiting...")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()