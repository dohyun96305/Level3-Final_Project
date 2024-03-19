from fastapi import APIRouter, Depends, HTTPException, Path

from database import SessionLocal
from models import PaperInfo

import sys
sys.path.append("/home/dohyun/Final_P/myapi")

import requests

from typing import Annotated
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette import status
from models import Keyword, PaperInfo
from domain.login_data.login_data import get_current_user

from pydantic import BaseModel


from sqlalchemy import func

router = APIRouter(
    prefix="/api/data",
)
user_dependency = Annotated[dict, Depends(get_current_user)]

@router.get("/get_data/{user_question}", status_code=status.HTTP_200_OK)
async def get_data(user: user_dependency, user_question : str):
    db = SessionLocal()

    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    _data_list = db.query(PaperInfo).order_by(func.rand()).limit(10).all() # 랜덤 10개 추출

    db.close()
    
    print(user_question)
    return _data_list

@router.get("/get_data_id", status_code=status.HTTP_200_OK)
def get_data_id(user: user_dependency, get_id : str):
    db = SessionLocal()
    _data_list = db.query(PaperInfo).filter(PaperInfo.id == get_id).first()

    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')

    db.close()

    return _data_list


@router.get('/chatbot/{paper_id}/{query}')
async def get_chatbot(paper_id: str, query: str):
    # return {'answer': paper_id, 'refernece': query}
    data ={'paper_id': paper_id, 'query': query}
    data = requests.post("http://223.130.162.53:8000/predict", json=data)
    
    data = data.json()
    print(data)
    
    _answer = data['prediction'].split('\n\n')[1]
    _Reference = data['prediction'].split('\n\n')[3:]

    chatbot = {'answer' : _answer, 'Reference' : _Reference}

    return chatbot