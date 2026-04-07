import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader(
        file_path=r"C:\Users\saipr\Langchain-POC\mediumblog1.txt",
        # autodetect_encoding=True.
        encoding="utf-8"
    )
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    '''
    chunk_size is 1000 characters, 
    chunk_overlap is useful when we don't want to use context between chunks.
    '''
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print('ingesting...')
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])

    print('finish.')

    