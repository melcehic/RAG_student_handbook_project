# RAG Pipeline Demo – BSc Student Handbook

This repository presents a demo of a Retrieval-Augmented Generation (RAG) pipeline designed to answer user questions based on the content of a university student handbook. The objective is to enhance question-answering capabilities by combining document retrieval with transformer-based language models.

## Project Overview

This project demonstrates how to build and query a local semantic search database using a PDF document as the source of knowledge. It enables users to compare the performance of different pre-trained models in retrieving and generating accurate responses from the same context.

### Key Features

- PDF ingestion and preprocessing
- Semantic chunking and embedding using `sentence-transformers`
- Local storage with `ChromaDB`
- Retrieval of relevant document sections based on user queries
- Answer generation with Hugging Face QA pipelines
- Interactive interface using Gradio (for Google Colab compatibility)
- Optional Streamlit interface for local usage on machines with a GPU

## Requirements and Setup

This project supports both Conda and pip-based installations. A `requirements.txt` and `environment.yml` file are provided for flexibility.

### Using Conda (recommended)

```bash
conda env create -f environment.yml
conda activate rag_app
