import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile")

vectorstore = PineconeVectorStore(
    index_name=os.environ["INDEX_NAME"], embedding=embeddings
)

retriver = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:
    
    {context}
    
    Question: {question}
    
    Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def retrieval_chain_without_lcel(query: str):
    """
    Simple retireval chain without langchian expression language(LCEL).
    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    #Step 1: Retrieve relevant document
    docs = retriver.invoke(query)

    #Step 2: Format documents into context string
    context = format_docs(docs)

    #Step 3: Format the prompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    #Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    #Step 5: Return the content
    return response.content


if __name__ == "__main__":
    print("Retrieving...")

    #Query
    query = "what is Pinecone in Machine Learning?"

    #===============================================================   
    # Option 0: Raw invocation without RAG 
    #===============================================================
    print('\n'+"=" * 60)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 60)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer")
    print(result_raw.content)

    #===============================================================   
    # Option 1: Use implementation WITHOUT LCEL 
    #===============================================================
    print('\n'+"=" * 60)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 60)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer")
    print(result_without_lcel)