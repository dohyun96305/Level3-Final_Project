from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from domain.get_data import api_data
from domain.question_data import question_data

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello():
    return '안녕하세요?'

app.include_router(api_data.router)
app.include_router(question_data.router)