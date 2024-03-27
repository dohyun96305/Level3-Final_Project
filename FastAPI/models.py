from sqlalchemy import Column, Integer, String, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.sql import func

from database import Base

class PaperInfo(Base): # 논문
    __tablename__ = "PaperInfo"        
    id = Column(String(255), nullable = False, primary_key=True)
    title = Column(Text, nullable=False)
    published_year = Column(Integer, nullable=False)
    updated_year = Column(Integer, nullable=False)
    abstract = Column(Text, nullable=False)
    categories = Column(Text, nullable=False)
    author1 = Column(Text, nullable=False)
    citation_count = Column(Integer, nullable=False)
    reference_count = Column(Integer, nullable=False)
    citation_graph = Column(JSON)
    
class Users(Base): # 사용자 login 관련
    __tablename__ = 'users'
    user_id = Column(Integer, primary_key=True, index=True, unique=True)
    email = Column(String(255), unique=True, nullable=False)
    username = Column(String(255),nullable=False)
    hashed_password = Column(String(255),nullable=False)

class Keyword(Base): # keyword 저장
    __tablename__ = 'keyword'
    keyword_id = Column(Integer, primary_key=True, index=True, unique=True)
    content = Column(String(255),nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id"),nullable=False)

class Chat(Base): # 채팅 저장
    __tablename__ = 'chat' 
    chat_id = Column(Integer, primary_key=True, index=True, unique=True)
    paper_id = Column(String(255), ForeignKey("PaperInfo.id"),nullable=False)
    user_id = Column(Integer, ForeignKey("users.user_id"),nullable=False)
    paper_title = Column(Text, nullable=False)

class Message(Base): # 채팅방에서 했던 대화 내용 저장 
    __tablename__ = 'message'
    message_id = Column(Integer, primary_key=True, index=True, unique=True)
    content = Column(Text,nullable=False)
    chat_id = Column(Integer, ForeignKey("chat.chat_id"),nullable=False)
    time = Column(DateTime(timezone=True), server_default=func.now())
    user_com = Column(Boolean, default=False) # 사용자면 0 / chatgpt면 1