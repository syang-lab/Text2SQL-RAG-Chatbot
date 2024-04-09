from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os

def get_files(dir_path):
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            if filename.endswith(".md"):
                file_list.append(os.path.join(filepath, filename))
            elif filename.endswith(".txt"):
                file_list.append(os.path.join(filepath, filename))
    return file_list


def get_text(dir_path):
    file_lst = get_files(dir_path)
    docs = []
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
        else:
            continue
        docs.extend(loader.load())
    return docs

tar_dir = [
    "/root/data/data_source",
]

docs = []
for dir_path in tar_dir:
    docs.extend(get_text(dir_path))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=20, chunk_overlap=10)
split_docs = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

persist_directory = 'data_base/vector_db/chroma'
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  
)
vectordb.persist()
