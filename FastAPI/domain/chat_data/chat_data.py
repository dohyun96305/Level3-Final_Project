from typing import Annotated
from pydantic import BaseModel, Field
from datetime import datetime
from sqlalchemy.orm import Session
from fastapi import APIRouter, Depends, HTTPException
from starlette import status
from models import Chat, Message, PaperInfo
from database import SessionLocal
from domain.login_data.login_data import get_current_user
from sqlalchemy import asc, desc;

router = APIRouter(
    prefix='/chat',
    tags=['chat']
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

class ChatRequest(BaseModel):
    paper_id : str

class MessageRequest(BaseModel):
    content : str
    paper_id : str
    time : datetime = Field(default_factory=datetime.now)
    user_com : bool # 사용자면 0 / chatgpt면 1

@router.get("/room", status_code=status.HTTP_200_OK)
async def read_all_chat(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    return db.query(Chat).filter(Chat.user_id == user.get('id')).all()

@router.get("/", status_code=status.HTTP_200_OK)
async def determine_chat(user: user_dependency, db: db_dependency, paper_id: str):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    Chat_model = db.query(Chat).filter(Chat.user_id == user.get('id')).filter(Chat.paper_id == paper_id).first()

    if Chat_model:
        print(Chat_model.chat_id)
        print(db.query(Message).filter(Message.chat_id == Chat_model.chat_id).order_by(asc(Message.time)).all())
        return db.query(Message).filter(Message.chat_id == Chat_model.chat_id).order_by(asc(Message.time)).all()
    else:
        return False


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_chat(user: user_dependency, db: db_dependency,
                      chat_request: ChatRequest):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    Chat_model = db.query(Chat).filter(Chat.user_id == user.get('id')).filter(Chat.paper_id == chat_request.paper_id).first()
    
    if Chat_model is not None:
        raise HTTPException(status_code=409, detail="Chatroom already exists")
    else:
        paper_title_model = db.query(PaperInfo).filter(PaperInfo.id == chat_request.paper_id)

        chat_model = Chat(paper_id = chat_request.paper_id, user_id=user.get('id'), paper_title = paper_title_model[0].title)
    
        db.add(chat_model)
        db.commit()

# text, paper_id, user_com(bool) 0이면 / 사용자 1이면 챗봇
@router.post("/message", status_code=status.HTTP_201_CREATED)
async def create_message(user: user_dependency, db: db_dependency,
                      message_request: MessageRequest):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')

    chat_model = db.query(Chat).filter(Chat.paper_id == message_request.paper_id).first()
    message_model = Message(content = message_request.content, chat_id = chat_model.chat_id, user_com = message_request.user_com)

    db.add(message_model)
    db.commit()

@router.delete("/", status_code=status.HTTP_200_OK)
async def delete_chat(user: user_dependency, db: db_dependency,
                      paper_id: str):
    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')
    
    Chat_model = db.query(Chat).filter(Chat.user_id == user.get('id')).filter(Chat.paper_id == paper_id).first()
    
    if Chat_model is not None:
        db.query(Message).filter(Message.chat_id == Chat_model.chat_id).delete()
        db.query(Chat).filter(Chat.chat_id == Chat_model.chat_id).delete()

        db.delete(Chat_model)
        db.commit()
    else:
       raise HTTPException(status_code=204, detail="NO_CONTENT")


