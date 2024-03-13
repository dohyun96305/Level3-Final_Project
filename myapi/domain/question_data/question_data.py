from fastapi import APIRouter, HTTPException

from database import SessionLocal
from models import PaperInfo
from domain.chatbot import get_requests

from sqlalchemy import func
from pydantic import BaseModel


router = APIRouter(
    prefix="/api/question",
)

class User_question(BaseModel) :
    user_question : str

@router.post("/upload")
async def get_question(user_question : User_question):
    _user_question = user_question

    return _user_question

