from sqlalchemy import Column, Integer, String, Text, DateTime, JSON

from database import Base
from pydantic import BaseModel


class PaperInfo(Base):

    __tablename__ = "PaperInfo"
        
    id = Column(String(255), nullable = False, primary_key=True)
    title = Column(Text, nullable=False)
    year = Column(Integer, nullable=False)
#    abstract = Column(Text, nullable=False)
    categories = Column(Text, nullable=False)
    journals = Column(Text)
    author1 = Column(Text, nullable=False)
    keyword = Column(JSON, nullable=False)

class PaperInfo_mini(Base):

    __tablename__ = "PaperInfo_mini"
        
    id = Column(String(255), nullable = False, primary_key=True)
    title = Column(Text, nullable=False)
    year = Column(Integer, nullable=False)
#    abstract = Column(Text, nullable=False)
    categories = Column(Text, nullable=False)
    journals = Column(Text)
    author1 = Column(Text, nullable=False)
    keyword = Column(Text, nullable=False)    
    citation_count = Column(Integer, nullable=False)
    reference_count = Column(Integer, nullable=False)
    

