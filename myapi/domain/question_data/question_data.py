from fastapi import APIRouter, Depends, HTTPException

from database import SessionLocal
from models import PaperInfo, Users 

from sqlalchemy import func
from pydantic import BaseModel

from typing import Annotated
from domain.login_data.login_data import get_current_user

router = APIRouter(
    prefix="/api/question",
)

user_dependency = Annotated[dict, Depends(get_current_user)]

class User_question(BaseModel) :
    user_question : str

@router.post("/upload")
async def get_question(user: user_dependency, user_question : User_question):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    _user_question = user_question
    _user_id = user.get('id')
    
    print(_user_question)
    print(_user_id)
    return _user_question