import warnings
warnings.filterwarnings('ignore')

import json
import pandas as pd

import torch

from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
now = datetime.now()

target_start = '2010-01-01'
target_end = now.strftime('%Y-%m-%d')
interval = 1

def get_model_score(model, test_data) : 
    texst_sequences = torch.tensor(test_data, dtype=torch.float32).unsqueeze(-1) 
    texst_sequences1 = texst_sequences.unsqueeze(0).to(device)

    return model(texst_sequences1)

def interval_change(str1) : 
    str1_dict = json.loads(str1)
    return list(str1_dict.values())

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

def get_models_id(model, _data_list) : 
    id = []
    citation_graph = []

    for a in _data_list : 
        id.append(a.id)
        citation_graph.append(json.dumps(a.citation_graph))
    
    citation_graph_list = pd.DataFrame({
        'id': id,
        'citation_graph': citation_graph
    })    

    for i, data in enumerate(tqdm(citation_graph_list['citation_graph'])) : 
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
        citation_graph_list.loc[i, f'last_value'] = last_value
        citation_graph_list.loc[i, f'data_value'] = json.dumps(list(data.values()))

    for i in range(len(citation_graph_list)) : 
        values = json.loads(citation_graph_list.loc[i, 'data_value'])
        values = get_model_score(model, values)

        citation_graph_list.loc[i, 'model_value'] = values.item()
    
    sorted_df = citation_graph_list.sort_values(by='model_value', ascending=False)
    sorted_ids = sorted_df['id']

    return sorted_ids[:10]   

