from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CHEOMA_PATH = "./DB/"
MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct-GGUF"
TEXT_EMBEDDING_MODEL = "BAAI/bge-m3"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}"""

client = OpenAI(base_url="http://localhost:8998/v1", api_key="lm-studio")
text_embedder=HuggingFaceEmbeddings(
    model_name = TEXT_EMBEDDING_MODEL
)

history = [
    {"role": "system", "content": "You are an intelligent assistant. You always provide well-reasoned answers that are both correct and helpful."},
    {"role": "user", "content": "Just introduce yourself"},
]

print("-------------------")
def query_rag(text):
    vector_db=Chroma(persist_directory = CHEOMA_PATH, embedding_function= text_embedder,collection_name="AI_llm")
    retrieved_data=vector_db.similarity_search_with_score(text,k=5)

    retrieved_context="".join([doc.page_content for doc, _score in retrieved_data])
    prompt = PROMPT_TEMPLATE.format(context=retrieved_context,question = text)
    return prompt

while True:
    completion = client.chat.completions.create(
        model= MODEL_NAME,
        messages=history,
        temperature=0,
        stream=True,
    )

    new_message = {"role": "assistant", "content": ""}
    
    for chunk in completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
            new_message["content"] += chunk.choices[0].delta.content

    history.append(new_message)
    

    print()
    history.append({"role": "user", "content": query_rag(input("> "))})