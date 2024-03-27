from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from domain.get_data import get_data
from domain.login_data import login_data
from domain.chat_data import chat_data

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://175.45.201.130:3000"
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

app.include_router(get_data.router)
app.include_router(login_data.router)
app.include_router(chat_data.router)