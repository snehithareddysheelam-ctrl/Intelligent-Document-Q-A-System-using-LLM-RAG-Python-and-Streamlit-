
import os

import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer
from google import genai


# ==================================================
# LOAD ENVIRONMENT VARIABLES
# ==================================================

load_dotenv()


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Intelligent Document Q&A",
    page_icon="📚",
    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title("📚 Intelligent Document Q&A System")

st.write(
    "Ask questions about your PDF using "
    "LLM + RAG + Python + Streamlit."
)


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ==================================================
# CREATE CHROMA DATABASE
# ==================================================

@st.cache_resource
def create_database():

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents"
    )

    return collection


collection = create_database()


# ==================================================
# EXTRACT TEXT FROM PDF
# ==================================================

def extract_text(pdf_file):

    reader = PdfReader(pdf_file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text


# ==================================================
# SPLIT TEXT INTO CHUNKS
# ==================================================

def split_text(
    text,
    chunk_size=800,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ==================================================
# ADD DOCUMENT TO VECTOR DATABASE
# ==================================================

def add_document(chunks):

    # Clear old documents
    try:

        collection.delete(
            where={}
        )

    except Exception:

        pass

    # Create embeddings
    embeddings = embedding_model.encode(
        chunks
    ).tolist()

    # Create unique IDs
    ids = [
        f"chunk_{i}"
        for i in range(len(chunks))
    ]

    # Store documents and embeddings
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )


# ==================================================
# RETRIEVE RELEVANT CHUNKS
# ==================================================

def retrieve_chunks(
    question,
    number_of_results=4
):

    # Create embedding for question
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=number_of_results
    )

    documents = results["documents"][0]

    return documents


# ==================================================
# GENERATE ANSWER USING GEMINI
# ==================================================

def generate_answer(
    question,
    context
):

    context_text = "\n\n".join(context)

    # Get API key
    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:

        return (
            "GEMINI_API_KEY is not configured. "
            "Please add your Gemini API key to the .env file."
        )

    # Create Gemini client
    client = genai.Client(
        api_key=api_key
    )

    # Prompt
    prompt = f"""
You are an intelligent document question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I could not find the answer in the uploaded document."

Do not make up information.

Context:
-------------------------
{context_text}
-------------------------

Question:
{question}

Answer:
"""

    # Generate response
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    return response.text


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header(
    "📄 Upload Document"
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)


# ==================================================
# PROCESS PDF
# ==================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    if st.button(
        "🔍 Process Document"
    ):

        with st.spinner(
            "Reading and processing the document..."
        ):

            # Extract text
            text = extract_text(
                uploaded_file
            )

            if not text.strip():

                st.error(
                    "Could not extract text from this PDF."
                )

            else:

                # Split into chunks
                chunks = split_text(
                    text
                )

                # Store embeddings
                add_document(
                    chunks
                )

                # Save processing status
                st.session_state[
                    "document_processed"
                ] = True

                st.success(
                    f"Document processed successfully! "
                    f"{len(chunks)} chunks created."
                )


# ==================================================
# QUESTION ANSWERING
# ==================================================

st.divider()

st.subheader(
    "💬 Ask a Question"
)

question = st.text_input(
    "Enter your question about the document:"
)


# ==================================================
# GET ANSWER
# ==================================================

if st.button(
    "🤖 Get Answer"
):

    # Check PDF
    if uploaded_file is None:

        st.warning(
            "Please upload a PDF first."
        )

    # Check question
    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    # Check processing
    elif not st.session_state.get(
        "document_processed",
        False
    ):

        st.warning(
            "Please click 'Process Document' first."
        )

    else:

        # Retrieve relevant chunks
        with st.spinner(
            "Searching the document..."
        ):

            relevant_chunks = retrieve_chunks(
                question
            )

        # Generate answer
        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                relevant_chunks
            )

        # Display answer
        st.subheader(
            "💡 Answer"
        )

        st.write(
            answer
        )

        # Display retrieved context
        with st.expander(
            "📖 View Retrieved Context"
        ):

            for i, chunk in enumerate(
                relevant_chunks
            ):

                st.markdown(
                    f"**Relevant Chunk {i + 1}**"
                )

                st.write(
                    chunk
                )

                st.divider()