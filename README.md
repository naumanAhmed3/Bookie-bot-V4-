# Bookiebot-V4

## Overview

Bookiebot is a Streamlit application designed for managing and querying documents. It uses LangChain's Chroma vector store for indexing and Google Generative AI for generating responses based on user queries. The application allows admin users to upload documents, manage logs, and provide a query interface for users to interact with the indexed data.

## Features

- **Admin Panel**:
  - Upload and process text documents.
  - View and download uploaded files.
  - Retrieve and display logs of file uploads.

- **User Panel**:
  - Submit queries to retrieve and analyze indexed documents.
  - Get responses based on document similarity and relevance.

## Prerequisites

- Docker installed on your machine. Download and install Docker from [here](https://www.docker.com/products/docker-desktop).

## Run with docker 
```
docker run --env-file .env -p 8501:8501 bookiebot-v4
```

## Setup

### Environment Variables

Create a `.env` file in the root directory of the project with the following content:

```env
GOOGLE_API_KEY=your_google_api_key
