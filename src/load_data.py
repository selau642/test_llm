from langchain_community.document_loaders import AsyncChromiumLoader
from langchain.document_transformers.html2text import \
    Html2TextTransformer

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS 
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

def load_from_url(url_list):
    loader = AsyncChromiumLoader(url_list)
    html_doc_list = loader.load()
    # Error message handling
    # sample error message page_content="Error: Timeout 30000ms exceeded"

    html_to_text = Html2TextTransformer()
    text_doc_list = html_to_text.transform_documents(html_doc_list)
    text_splitter = RecursiveCharacterTextSplitter()
    chunk_text_doc_list = text_splitter.split_documents(text_doc_list)

    return chunk_text_doc_list

def load_faiss_retriever(chunk_text_doc_list):
    all_mpnet_base_v2_embeddings= HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    faiss_store = FAISS.from_documents(
        documents=chunk_text_doc_list, 
        embedding=all_mpnet_base_v2_embeddings
        )

    return faiss_store.as_retriever()

if __name__ == "__main__":
    url_list = [
        'https://cnaluxury.channelnewsasia.com/experiences/pizza-pop-singapore-diego-vitagliano-la-bottega-enoteca-antonio-miscellaneo-243026'
    ]

    chunk_text_doc_list = load_from_url(url_list)
    faiss_retriever = load_faiss_retriever(chunk_text_doc_list)