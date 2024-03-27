from fastapi import APIRouter, Depends, HTTPException

from database import SessionLocal
from models import PaperInfo

import sys
sys.path.append("/home/dohyun/Final_P/myapi")

import torch
import torch.nn as nn

from typing import Annotated
from starlette import status
from models import PaperInfo
from domain.login_data.login_data import get_current_user

from sqlalchemy import or_

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

router = APIRouter(
    prefix="/api/data",
)
user_dependency = Annotated[dict, Depends(get_current_user)]

@router.get("/get_data/{user_question}", status_code=status.HTTP_200_OK)
async def get_data(user: user_dependency, user_question: str): 
    db = SessionLocal()

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
    sorted_ids =  sorted_ids.values.tolist()

    print(sorted_ids) # 10개 
    matched_papers = db.query(PaperInfo).filter(PaperInfo.id.in_(sorted_ids)).all()

    return matched_papers
    