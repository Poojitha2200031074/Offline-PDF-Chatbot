from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import fitz  # PyMuPDF


# STEP 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        pages.append({
            "page": page_number + 1,
            "content": text
        })

    document.close()

    return pages


# STEP 2: Split PDF text into chunks
def split_text_into_chunks(pages):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []

    for page in pages:

        documents.append(
            Document(
                page_content=page["content"],
                metadata={"page": page["page"]}
            )
        )

    chunks = splitter.split_documents(documents)

    return chunks


# STEP 3: Generate embeddings
def generate_embeddings(chunks):

    # Initialize the local embedding model
    embedding_model = OllamaEmbeddings(
        model="all-minilm"
    )

    # Extract text from all chunks
    texts = [chunk.page_content for chunk in chunks]

    # Convert text into numerical vectors
    vectors = embedding_model.embed_documents(texts)

    return embedding_model, vectors


# STEP 4: Store embeddings in FAISS
def store_embeddings(chunks, embedding_model, vectors):

    # Extract text from chunks
    texts = [chunk.page_content for chunk in chunks]

    # Combine each text with its embedding
    text_embeddings = list(zip(texts, vectors))

    # Create FAISS vector database
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedding_model,
        metadatas=[chunk.metadata for chunk in chunks]
    )

    # Save database locally
    vectorstore.save_local("vectorstore")

    return vectorstore


# STEP 5: Main program
if __name__ == "__main__":

    pdf_path = "data/sample.pdf"

    print("Reading PDF...")

    pages = extract_text_from_pdf(pdf_path)

    # Check extracted PDF text
    print("\n--- Extracted PDF Content ---")

    for page in pages:
        print("\nPage:", page["page"])
        print(page["content"])

    print("PDF successfully loaded!")
    print("Total pages:", len(pages))

    print("\nSplitting PDF into chunks...")

    chunks = split_text_into_chunks(pages)

    print("Total chunks:", len(chunks))

    if not chunks:
        print("No text found in PDF!")
        exit()

    print("\nGenerating embeddings...")

    embedding_model, vectors = generate_embeddings(chunks)
    print("\nStoring embeddings in FAISS...")

    vectorstore = store_embeddings(
        chunks,
        embedding_model,
        vectors
    )

    print("FAISS database created successfully!")

    print("\nEmbeddings generated successfully!")

    print("Total embeddings:", len(vectors))

    print("Vector dimensions:", len(vectors[0]))

    print("First embedding:", vectors[0][:5])

    # STEP 6: Test similarity search
    query = "What is Artificial Intelligence?"

    results = vectorstore.similarity_search(query, k=2)

    print("\nRelevant PDF content:")

    for result in results:
        print("\nPage:", result.metadata["page"])
        print(result.page_content)
