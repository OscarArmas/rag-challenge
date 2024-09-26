# Basic RAG Implementation

This project implements a Retrieval-Augmented Generation (RAG) system that compares TF-IDF and Embeddings models to answer user queries. It also includes a basic prompt-enhanced question-answering model using Generative AI from Google.

## Features

- TF-IDF and Embeddings model comparison for question-answering
- Retrieval of relevant documents from provided text
- Generative AI model for context-aware question answering
- Intent detection for handling general inquiries (greetings, thank you, irrelevant queries, etc.)
- Streamlit-based interactive user interface

## Setup

### Prerequisites

- Python 3.8 or higher
- API key for Google Generative AI
- Streamlit (for the web interface)

### Installation

1. Clone the repository to your local machine.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt