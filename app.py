
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM


# STEP 1: Load the embedding model
print("Loading embedding model...")

embedding_model = OllamaEmbeddings(
    model="all-minilm"
)


# STEP 2: Load the FAISS vector database
print("Loading PDF database...")

vectorstore = FAISS.load_local(
    "vectorstore",
    embedding_model,
    allow_dangerous_deserialization=True
)

print("PDF database loaded successfully!")


# STEP 3: Initialize the LLM
print("Loading Llama model...")

llm = OllamaLLM(
    model="llama3.2:1b"
)


# STEP 4: Create the question-answering function
def answer_question(question):

    # Retrieve relevant chunks from FAISS
    results = vectorstore.similarity_search(
        question,
        k=8
    )

    # DEBUG: Display retrieved PDF content
    print("\n--- Retrieved PDF Content ---")

    for i, doc in enumerate(results):

        print("\nChunk:", i + 1)
        print("Page:", doc.metadata["page"])
        print("Content:", doc.page_content)

    # Combine retrieved content
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    # Create prompt
    prompt = f"""
    You are a helpful PDF assistant.

    Answer the question using only the information
    provided in the context.

    If the answer is not available in the context,
    say "The answer is not available in the PDF."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # Generate response using Llama
    response = llm.invoke(prompt)

    return response, results


# STEP 5: Chat with the PDF
if __name__ == "__main__":

    print("\nWelcome to the Offline PDF Chatbot!")

    while True:

        question = input(
            "\nAsk your question (type exit to stop): "
        )

        if question.lower() == "exit":
            print("Thank you for using the chatbot!")
            break

        if not question.strip():
            print("Please enter a question.")
            continue

        print("\nSearching your PDF...")

        response, sources = answer_question(question)

        print("\nChatbot Answer:")
        print(response)

        print("\nSource Pages:")

        for doc in sources:
            print("Page:", doc.metadata["page"])
