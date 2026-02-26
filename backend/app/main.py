from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .data_loader import load_titanic_df
from .agent import run_agent

app = FastAPI(title="Titanic Chat Agent")

@app.get("/")
def root():
    return {"message":"Titanic Chat Agent API. Use /docs, /health, /chat"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

df = load_titanic_df()

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str
    image_b64: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    out = run_agent(req.question, df)
    return ChatResponse(**out)
