from fastapi import APIRouter, Depends, HTTPException

from database import SessionLocal
from models import PaperInfo

import sys
sys.path.append("/home/dohyun/Final_P/")

import torch
import torch.nn as nn

import json
from tqdm import tqdm

from utils.embeddings_similarity import parse_csv, compute_similarity, get_top_similarity_results

from typing import Annotated
from starlette import status

from models import PaperInfo, Keyword
from database import SessionLocal
from sqlalchemy.orm import Session

from domain.login_data.login_data import get_current_user

from sqlalchemy import or_, select

from paper_models_get import get_models_id

########################################################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):    # Define the RNN model

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)  # Move initial hidden state to device
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

model_rnn = RNN(input_size=1, hidden_size=128, num_layers=1, output_size=1).to(device)  # Move model to device

model_rnn.load_state_dict(torch.load('/home/dohyun/Final_P/paper_models/model_rnn_state_dict.pth', map_location=device))

embedding_vectors = parse_csv('/home/dohyun/Final_P/csv_files/embedding.csv')

router = APIRouter(
    prefix="/api/data",
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[dict, Depends(get_current_user)]

@router.get("/get_data/{user_question}", status_code=status.HTTP_200_OK)
async def get_data(user: user_dependency, db: db_dependency, user_question: str): 

    if user is None:
        raise HTTPException(status_code=401, detail='Authentication Failed')

    user_question = user_question.strip()

    _data_list = db.query(PaperInfo).filter(
        or_(
            PaperInfo.title.like(f"%{user_question}%"),
            PaperInfo.categories.like(f"%{user_question}%"),
        )
    ).all()
    
    sorted_ids = get_models_id(model_rnn, _data_list)
    sorted_ids = sorted_ids.values.tolist()

    print(sorted_ids) # 10개 

    matched_papers = []

    for id in sorted_ids : 
        query_1 = select([PaperInfo.title, 
                                PaperInfo.id,
                                PaperInfo.author1, 
                                PaperInfo.citation_count, 
                                PaperInfo.published_year]).where(PaperInfo.id == id)
        
        result = db.execute(query_1).fetchall()
        matched_papers += result

    keyword_model = Keyword(content = user_question, user_id=user.get('id'))

    db.add(keyword_model)
    db.commit()

    top_5_results = {}
    for base_id in sorted_ids:
        similarity_results = compute_similarity([base_id], embedding_vectors)
        top_5_results[base_id] = get_top_similarity_results(similarity_results)

    targets_list = []

    for target in top_5_results : 
        for list1 in top_5_results[target] :
            query_2 = select([PaperInfo.title, 
                            PaperInfo.id,
                            PaperInfo.author1, 
                            PaperInfo.citation_count, 
                            PaperInfo.published_year]).where(PaperInfo.id == list1.get('source'))
            result = db.execute(query_2).fetchall()
            if result : 
                targets_list += result

  
    return {'matched_papers' : matched_papers, 
            'top_5_results' : top_5_results, 
            'targets_list' : targets_list}