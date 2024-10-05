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
from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool


memory = ConversationBufferMemory(memory_key="chat_history" ,return_messages=True) #Maintains the conversational context

role =""" 1)You are BookieBee, an agent-based chatbot for Bookme.pk, designed to assist users with a variety of queries related to services offered by the company. Bookme.pk provides services including bus ticket booking, cinema ticketing, airline ticket reservations, and event ticket sales. Use the available tools to provide accurate, up-to-date, and helpful 
            responses regarding these services, ensuring a seamless customer experience.
            Even if the query isn't regrading the services , try your best to answer it using "StaticTool"
          2)Try to figure out inputs/arguments for the tools you need to use from the user's given query and if they are not found there ask them from user.
          3)Use current date and time as a reference to identify dates/times from user's queries. To get sure about current date/time use the provided tool "today_date".
          4)Always introduce yourself as BookieBee, making the conversation engaging and friendly.
          5)Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.
          6)Format the your responses to make them easily readable and understandable.
                    """

memory.chat_memory.add_message(SystemMessage(content=role))
load_dotenv()
# Access environment variables
LLM_api = os.getenv('GOOGLE_API_KEY')


teller = GoogleGenerativeAI(
                    model="gemini-1.5-flash", 
                    google_api_key=LLM_api
                )


advisor =  GoogleGenerativeAI(
                    model="gemini-1.5-flash", 
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
            ('system', '''You are BookieBee, a helpful assistant developed by Bookme.pk to assist users with booking-related queries on the Bookme website and mobile app. Your main goals are:

Guide users professionally and efficiently with responses based solely on the knowledge and context you have access to.
Do not reveal or mention any internal mechanisms, such as a knowledge base or context being used.
If a query isn't related to Company's information and your role , don't start going into it's details or guiding with it and take it as a joke.
Always introduce yourself as BookieBee, making the conversation engaging and friendly.
Use decent and professional emojis where appropriate to make interactions more appealing and light-hearted.'''),

            ('human', 'Based on the following context:{context} , Answer Query: "{query}".')
        ]
    )
    return template, variables


def create_advisor_template(variables):
    template = ChatPromptTemplate.from_messages(
        [
            ('system', '''Give zero shot answer , just give list of what's asked without any extra information.'''),

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

# Function to set background image
def set_bg_hack(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image here, replace with your image link
set_bg_hack("https://www.google.com/imgres?q=bookme&imgurl=https%3A%2F%2Flookaside.fbsbx.com%2Flookaside%2Fcrawler%2Fmedia%2F%3Fmedia_id%3D100064589565607&imgrefurl=https%3A%2F%2Fwww.facebook.com%2Fbookmepk%2F&docid=b4365xf6hDsHqM&tbnid=xfxIxjoM35BppM&vet=12ahUKEwj5-qbkhvaIAxWpBdsEHeNlI4wQM3oECGAQAA..i&w=2048&h=2048&hcb=2&ved=2ahUKEwj5-qbkhvaIAxWpBdsEHeNlI4wQM3oECGAQAA")  # Replace with a direct link to the image

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
        uploaded_files_dir = "uploads"  # Directory where uploaded files will be saved
        file_path = os.path.join(uploaded_files_dir, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        # Log the file upload
        log_file_upload(uploaded_file.name)

        # Process and store the text file content
        process_and_store_text(file_path)
        
        st.success("File uploaded and processed successfully!")

    # File retrieval section
    
# User Panel
with tab_user:
    st.header("User Panel - Query the Database")

    # Query section
    query_text = st.text_input("Enter your query")

    if query_text:
        memory.chat_memory.add_message(HumanMessage(content=query_text))
        response = executor.invoke({"input": query_text})
        memory.chat_memory.add_message(AIMessage(content=response['output']))
        st.write(f"Bot: {response['output']}")

    # Display chat history
    if "chat_history" in memory.chat_memory:
        st.subheader("Chat History")
        for message in memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                st.write(f"You: {message.content}")
            elif isinstance(message, AIMessage):
                st.write(f"Bot: {message.content}")
