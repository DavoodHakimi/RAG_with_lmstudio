from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import chromadb
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import FakeEmbeddings

docs_path="./Docs/"

file_names = [f for f in listdir(docs_path) if isfile(join(docs_path, f))]

chroma_client = chromadb.PersistentClient(path="./DB/")
collection=chroma_client.get_or_create_collection(name="AI_llm")
existing_ids = set(collection.get(ids=None)["ids"]) # reading the doc ids to prevent from writing duplicate docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
    length_function=len,
    is_separator_regex=False,
)
# text_embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
text_embedding=HuggingFaceEmbeddings(model_name="BAAI/bge-large-zh")

for filename in file_names[1:]:
    print("Trying to Read ",filename)
    doc=pypdf.PdfReader(docs_path + filename)
    print("\n number of pages:",len(doc.pages))
    doc_text=""
    for page_num in tqdm(range(len(doc.pages)), desc="Adding Pages"):
        doc_text += doc.pages[page_num].extract_text()
    print("\n",filename, " Loaded Succesfully!!")
    
    print("\n Chunking Started")
    chunks=text_splitter.split_text(doc_text)
    print("\n Chunking Done!")


        
    for i, chunk in tqdm(enumerate(chunks),total=len(chunks),desc="Writing to Database"):
        doc_id=f"{filename}_{i}"

        if doc_id in existing_ids:
            # print(f"ID {doc_id} already exists, skipping...")
            continue

        # Chroma can't add big sized bachtes to Vector DB, so i did this to handle adding big files to DB
        documents_list = []
        embeddings_list = []
        ids_list = []
        vector = text_embedding.embed_query(chunk)
        
        documents_list.append(chunk)
        embeddings_list.append(vector)
        ids_list.append(doc_id)

        collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list
        )
