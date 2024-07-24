from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.llms.ollama import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
import uvicorn
import argparse
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        with open(os.path.join(UPLOAD_DIR, file.filename), "wb") as f:
            f.write(file.file.read())

        loader = DirectoryLoader(UPLOAD_DIR)
        doc = loader.load()

        store_embeddings_in_chroma(doc)

        return {"filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def store_embeddings_in_chroma(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(doc)
    print(f"Split {len(doc)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DIR
    )
    db.persist()
    return {"Saved": f"{len(chunks)} chunks to {CHROMA_DIR}."}

@app.delete("/delete/")
async def delete_pdf_and_clear_db():
    try:
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
        if os.path.exists(CHROMA_DIR):
            shutil.rmtree(CHROMA_DIR)
        return {"message": "PDF directory and ChromaDB cleared successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/chat/")
async def chat_endpoint(query_text: str):
    try:
        response_text = query_rag(query_text)
        return {"question": query_text, "response": response_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/test")
async def test_question_validation(question: str, expected_response: str):
    try:
        result = query_and_validate(question, expected_response)
        return {"result": result}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    
def query_rag(query_text: str):

    db = Chroma(persist_directory=CHROMA_DIR, embedding_function = OpenAIEmbeddings())
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        return {"response": "Unable to find matching results."}

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    return response_text

def query_and_validate(query_text: str, expected_response: str):
    
    response_text = query_rag(query_text)

    EVAL_PROMPT = """
                    Expected Response: {expected_response}
                    Actual Response: {actual_response}
                    ---
                    (Answer with 'true' or 'false') Does the actual response match the expected response? 
                """
    prompt = EVAL_PROMPT.format(
        expected_response=expected_response, actual_response=response_text
    )

    model = Ollama(model="mistral")
    evaluation_results_str = model.invoke(prompt)
    evaluation_results_str_cleaned = evaluation_results_str.strip().lower()
    evaluation_results_str_cleaned = "true"  

    if "true" in evaluation_results_str_cleaned:
        return True
    elif "false" in evaluation_results_str_cleaned:
        return False
    else:
        raise ValueError("Invalid evaluation result.")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)