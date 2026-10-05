# Intelligent-Document-Q-A-System-using-LLM-RAG-Python-and-Streamlit-
# 📚 Intelligent Document Q&A System using LLM, RAG, Python and Streamlit

An AI-powered **Document Question Answering System** that allows users to upload a PDF and ask questions about its content.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from the uploaded document and uses a **Large Language Model (LLM)** to generate accurate answers based only on the retrieved context.

---

## 🚀 Project Overview

The **Intelligent Document Q&A System** is designed to make it easier for users to interact with and understand PDF documents.

Instead of manually reading a long document, users can:

1. Upload a PDF document.
2. Process the document.
3. Ask questions in natural language.
4. Retrieve relevant information from the document.
5. Generate an AI-based answer using an LLM.

The application is built using **Python and Streamlit**, with **Sentence Transformers** for generating embeddings, **ChromaDB** for vector storage and retrieval, and **Google Gemini** for answer generation.

---

## ✨ Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDF files
- ✂️ Split documents into smaller chunks
- 🧠 Generate text embeddings using Sentence Transformers
- 🗄️ Store embeddings using ChromaDB
- 🔎 Retrieve relevant document chunks
- 🤖 Generate answers using Google Gemini
- 💬 Ask questions using natural language
- 📖 View retrieved document context
- 🚫 Avoid generating answers when information is not available
- 🖥️ Simple and interactive Streamlit interface

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Main programming language |
| **Streamlit** | Web application interface |
| **PyPDF** | Extract text from PDF documents |
| **Sentence Transformers** | Generate text embeddings |
| **ChromaDB** | Vector database for storing and retrieving embeddings |
| **Google Gemini** | Large Language Model for answer generation |
| **python-dotenv** | Manage environment variables |
| **RAG** | Retrieve relevant information before generating answers |
| **Git & GitHub** | Version control and project hosting |

---

## 🧠 What is RAG?

**RAG stands for Retrieval-Augmented Generation.**

It combines:

- 🔎 **Information Retrieval**
- 🤖 **Large Language Models**

The system first searches the uploaded document for information relevant to the user's question. The retrieved information is then provided to the LLM, which generates the final answer.

### RAG Process

```text
User Question
      ↓
Question Embedding
      ↓
Vector Search
      ↓
ChromaDB
      ↓
Relevant Document Chunks
      ↓
Google Gemini LLM
      ↓
Generated Answer