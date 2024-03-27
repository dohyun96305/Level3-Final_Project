import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import pymysql

import multiprocessing as mp

from pytz import timezone
from datetime import datetime

from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

device = "cuda"
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)

def target_date_convert(str1) : 
    return str1[:10]

def show_time(str1) : 
    now1 = datetime.now(timezone('Asia/Seoul'))
    now2 = str(now1.strftime('%Y.%m.%d - %H:%M:%S'))
    print('------------ ' + str1 + '_' + now2 + ' ------------')
    
def citation_list_interval(list1, int1) :
    quote_dates = list1

    quarterly_quotes = defaultdict(int)
    max_quotes = 0

    for date_str in quote_dates:
        date = datetime.strptime(date_str, '%Y-%m-%d')

        year_quarter = (date.year, (date.month - 1) // int1 + 1)
        quarterly_quotes[year_quarter] += int(list1.get(date_str)) - max_quotes
        max_quotes = int(list1.get(date_str))
    
    accumulated_quotes = defaultdict(int)
    total_quotes = 0

    start_year = min(quarterly_quotes.keys())[0]
    start_quarter = min(quarterly_quotes.keys(), key=lambda x: (x[0], x[1]))[1]
    end_year = max(quarterly_quotes.keys())[0]
    end_quarter = max(quarterly_quotes.keys(), key=lambda x: (x[0], x[1]))[1]

    current_year = start_year
    current_quarter = start_quarter

    while (current_year, current_quarter) <= (end_year, end_quarter):
        total_quotes += quarterly_quotes[(current_year, current_quarter)]
        if len(str(current_quarter)) == 1 : 
            keys_name = f'{current_year}-0{current_quarter}'
        else : 
            keys_name = f'{current_year}-{current_quarter}'
        accumulated_quotes[keys_name] = total_quotes

        current_quarter += 1
        if current_quarter > (12//int1):
            current_year += 1
            current_quarter = 1
            
    return accumulated_quotes
   
def interval_change(str1) : 
    str1_dict = json.loads(str1)

    return list(str1_dict.values())

def create_sequences(data, sequence_length) :
    sequences = []
    targets = []

    for sequence in data:
        input = sequence[:sequence_length]
        label = sequence[sequence_length]
        sequences.append(input)
        targets.append(label)

    return sequences, targets

def model_train_save(target_end, sql_user, sql_password, sql_port) : 

    target_start = '2010-01-01'
    target_end = target_date_convert(target_end)
    interval = 1 

    print(target_end)

    db = pymysql.connect(
        host = 'localhost',  # DATABASE_HOST
        port = int(sql_port),
        user = sql_user,  # DATABASE_USERNAME
        passwd = sql_password,  # DATABASE_PASSWORD
        db = 'final_project',  # DATABASE_NAME
        charset = 'utf8'
    )

    citation_graph_sql = f'SELECT id, citation_graph FROM PaperInfo;'
    citation_graph_list = pd.read_sql(citation_graph_sql, db)

    db.close()

    show_time(f'Wait!! {interval}_interval citation convert Start!!')
    
    print('target_start', target_start)
    print('target_end', target_end)
    print('interval', interval)

    for i, data in enumerate(citation_graph_list['citation_graph']) : 
        data = json.loads(data)
        
        last_key = max(data.keys())
        last_value = data[last_key]

        if target_end not in data.keys() : 
            data[target_end] = last_value

        data[target_start] = 0
        data = dict(sorted(data.items()))

        if min(data.keys()) != target_start : 

            previous_key = None
            for key in list(data.keys()):
                if key == target_start : 
                    data[key] = data[previous_key]
                    break
                previous_key = key

            to_remove_start = [key for key in data if key < target_start]
            for key in to_remove_start:
                del data[key] 

        if max(data.keys()) != target_end : 

            previous_key = None 
            for key in list(data.keys())[::-1] : 
                if key == target_end : 
                    data[key] = data[previous_key]
                    break
                previous_key = key

            to_remove_end = [key for key in data if key > target_end]
            for key in to_remove_end:
                del data[key] 

        data = citation_list_interval(data, interval)
        citation_graph_list.loc[i, f'citation_graph_interval'] = json.dumps(data) 

        if i % 100000 == 0 : 
                show_time(f'Wait!! {i}th data_{interval}_interval citation convert Done!!')

    show_time(f'Wait!! {interval}_interval citation convert Done!!')

    data = citation_graph_list['citation_graph_interval'].apply(interval_change)

    n = len(data[0])

    sequence_length = n-1
    sequences, targets = create_sequences(data, sequence_length)

    sequences_tensor = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)  
    targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    sequences_tensor = sequences_tensor.to(device)
    targets_tensor = targets_tensor.to(device)

    print("sequences_tensor size:", sequences_tensor.size())
    print("targets_tensor size:", targets_tensor.size())

    dataset = TensorDataset(sequences_tensor, targets_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    class RNN(nn.Module) :  

        def __init__(self, input_size, hidden_size, num_layers, output_size) :
            super(RNN, self).__init__()
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x) :
            h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)  
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    rnn = RNN(input_size=1, hidden_size=128, num_layers=1, output_size=1).to(device)  
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn.parameters(), lr=0.001)

    num_epochs = 10
    for epoch in range(num_epochs) :
        rnn.train()
        train_loss = 0.0
        for seqs, targets in train_dataloader :
            optimizer.zero_grad()
            outputs = rnn(seqs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)
        train_loss /= len(train_dataloader.dataset)

        rnn.eval()
        val_loss = 0.0
        with torch.no_grad() :
            for seqs, targets in val_dataloader :
                outputs = rnn(seqs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * seqs.size(0)
        val_loss /= len(val_dataloader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    path = '/home/dohyun/Final_P/paper_models/'
    torch.save(rnn.state_dict(), path + 'model_rnn_state_dict.pth')

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    model_train_save(target_end, sql_user, sql_password, sql_port)
