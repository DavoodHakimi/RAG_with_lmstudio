from os import listdir
from os.path import isfile, join
from tqdm import tqdm
import chromadb
import pypdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


TEXT_EMBEDDING_MODEL = "BAAI/bge-m3" # Or any other Embedding model from HuggingFace
VECTOR_DB_NAME = "YOUR_DB_NAME"
VECTOR_DB_PATH = "PATH_TO_YOUR_VECTOR_DB"
CHUNK_SIZE = 1000
OVERLAP = 10

docs_path="YOUR_PDF_DOCUMENTS_DIR"
file_names = [f for f in listdir(docs_path) if isfile(join(docs_path, f))]

chroma_client = chromadb.PersistentClient(path = VECTOR_DB_PATH)
collection=chroma_client.get_or_create_collection(name = VECTOR_DB_NAME)
existing_ids = set(collection.get(ids=None)["ids"]) # reading the doc ids to prevent from writing duplicate docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = CHUNK_SIZE,
    chunk_overlap = OVERLAP,
    length_function = len,
    is_separator_regex = False,
)

text_embedder=HuggingFaceEmbeddings(
    model_name = TEXT_EMBEDDING_MODEL
)

def write_2_DB(filename):
    print("Trying to Read ",filename)
    doc=pypdf.PdfReader(docs_path + filename)

    print("\n number of pages:",len(doc.pages))
    doc_text=""
    for page_num in tqdm(range(len(doc.pages)), desc="Adding Pages"):
        doc_text += doc.pages[page_num].extract_text()
    print("\n",filename, " Loaded Succesfully!!")
    
    print("\n Chunking Started ...")
    chunks=text_splitter.split_text(doc_text)
    print("\n Chunking Done!")


        
    for i, chunk in tqdm(enumerate(chunks),total=len(chunks),desc="Writing to Database"):
        doc_id=f"{filename}_{i}"

        if doc_id in existing_ids:
            continue

        # Chroma can't add big size bachtes to Vector DB, so i did this to handle adding large files to DB
        documents_list = []
        embeddings_list = []
        ids_list = []
        vector = text_embedder.embed_query(chunk)
        
        documents_list.append(chunk)
        embeddings_list.append(vector)
        ids_list.append(doc_id)

        collection.add(
            embeddings=embeddings_list,
            documents=documents_list,
            ids=ids_list
        )


for filename in file_names:
    write_2_DB(filename)