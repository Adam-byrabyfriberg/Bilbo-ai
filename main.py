from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from rag import ask_question
from indexer import build_or_load_index

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Index skapas/startas direkt vid uppstart
index = build_or_load_index()

@app.post("/ask")
async def ask(request: Request):
    body = await request.json()
    question = body.get("question")
    if not question:
        return {"error": "Ingen fråga mottagen"}

    answer = ask_question(index, question)
    return {"answer": answer}
