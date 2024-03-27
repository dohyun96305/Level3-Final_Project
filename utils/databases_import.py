import warnings
warnings.filterwarnings('ignore')

import os
import json
import random
import pandas as pd
import requests

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pymysql

import re

from pytz import timezone
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm

from keybert import KeyBERT
from sqlalchemy import create_engine

FILE = '/home/dohyun/Final_P/arxiv/arxiv-metadata-oai-snapshot.json'

def target_date_convert(str1) : 
    return str1[:7]

def show_time(str1) :
    now1 = datetime.now(timezone('Asia/Seoul'))
    now2 = str(now1.strftime('%Y.%m.%d - %H:%M:%S'))
    print('------------ ' + str1 + '_' + now2 + ' ------------')

def get_data():
    with open(FILE) as f:
        for line in f:
            yield line

category_ch = {
    'cs.AI' : 'Artificial Intelligence',
    'cmp-lg' : 'Computation and Language',
    'cs.AR' : 'Hardware Architecture',
    'cs.CC' : 'Computational Complexity', 
    'cs.CE' : 'Computational Engineering, Finance and Science', 
    'cs.CG' : 'Computational Geometry', 
    'cs.CL' : 'Computational and Language',
    'cs.CR' : 'Cryptography and Security', 
    'cs.CV' : 'Computer Vision and Pattern Recognition', 
    'cs.CY' : 'Computers and Society', 
    'cs.DB' : 'Databases',
    'cs.DC' : 'Distributed, Parallel and Cluster Computing', 
    'cs.DL' : 'Digital Libraries', 
    'cs.DM' : 'Discrete Mathematics',
    'cs.DS' : 'Data Structures and Algorithms', 
    'cs.ET' : 'Emerging Technologies', 
    'cs.FL' : 'Formal Languages and Automata Theory', 
    'cs.GL' : 'General Literature', 
    'cs.GR' : 'Graphics', 
    'cs.GT' : 'Computer Science and Game Theory', 
    'cs.HC' : 'Human-Computer Interaction', 
    'cs.IR' : 'Information Retrieval', 
    'cs.IT' : 'Information Theory', 
    'cs.LG' : 'Machine Learning', 
    'cs.LO' : 'Logic in Computer Science', 
    'cs.MA' : 'Multiagent Systems', 
    'cs.MM' : 'Multimedia', 
    'cs.MS' : 'Mathematical Software', 
    'cs.NA' : 'Numerical Analysis', 
    'cs.NE' : 'Neural and Evolutionary Computing', 
    'cs.NI' : 'Networking and Internet Architecture', 
    'cs.OH' : 'Other Computer Science', 
    'cs.OS' : 'Operating Systems', 
    'cs.PF' : 'Performance', 
    'cs.PL' : 'Programming Language', 
    'cs.RO' : 'RObotics', 
    'cs.SC' : 'Symbolic Computation',     
    'cs.SD' : 'Sound', 
    'cs.SE' : 'Software Engineering', 
    'cs.SI' : 'Social and Information Networks', 
    'cs.SY' : 'Systems and Control', 
    'econ.EM' : 'Econometrics', 
    'econ.GN' : 'General Economics', 
    'econ.TH' : 'Theoretical Economics', 
    'eess.AS' : 'Audio and Speech Processing', 
    'eess.IV' : 'Image and Video Processing', 
    'eess.SP' : 'Signal Processing', 
    'eess.SY' : 'Systems and Control', 
    'math.AC' : 'Commutative Algebra', 
    'math.AG' : 'Algebraic Geometry', 
    'alg-geom' : 'Algebraic Geometry', 
    'math.AP' : 'Analysis of PDEs', 
    'math.AT' : 'Algebraic Topology', 
    'math.CA' : 'Classical Analysis and ODEs', 
    'math.CO' : 'Combinatorics',
    'math.CT' : 'Category Theory', 
    'math.CV' : 'Complex Variables', 
    'math.DG' : 'Differential Geometry', 
    'dg-ga' : 'Differential Geometry',
    'math.DS' : 'Dynamical Systems', 
    'math.FA' : 'Functional Analysis', 
    'funct-an' : 'Functional Analysis', 
    'math.GM' : 'General Mathematics', 
    'math.GN' : 'General Topology', 
    'math.GR' : 'Group Theory', 
    'math.GT' : 'Geometric Topology',
    'math.HO' : 'History and Overview', 
    'math.IT' : 'Information Theory', 
    'math.KT' : 'K-Theory and Homology', 
    'math.LO' : 'Logic', 
    'math.MG' : 'Metirc Geometry', 
    'math.MP' : 'Mathematical Physics', 
    'math.NA' : 'Numerical Analysis', 
    'math.NT' : 'Number Theory', 
    'math.OA' : 'Operator Algebras', 
    'math.OC' : 'Optimization and Control', 
    'math.PR' : 'Probability', 
    'math.QA' : 'Quantum ALgebra', 
    'q-alg' : 'Quantum Algebra and Topology', 
    'math.RA' : 'Rings and Algebras', 
    'math.RT' : 'Representation Theory', 
    'math.SG' : 'Symplectic Geometry', 
    'math.SP' : 'Spectral Theory', 
    'math.ST' : 'Statistics Theory', 
    'astro-ph' : 'Astrophysics', 
    'astro-ph.CO' : 'Cosmology and Nongalactic Astrophysics', 
    'astro-ph.EP' : 'Earth and Planetary Astrophysics', 
    'astro-ph.GA' : 'Astrophysics of Galaxies', 
    'astro-ph.HE' : 'High Energy Astrophysical Phenomena', 
    'astro-ph.IM' : 'Instrumentation and Methods for Astrophysics', 
    'astro-ph.SR' : 'Solar and Stellar Astrophysics', 
    'cond-mat' : 'Condensed Matter',
    'cond-mat.dis-nn' : 'Disordered Systems and Neural Networks', 
    'cond-mat.mes-hall' : 'Mesoscale and Nanoscale Physics', 
    'cond-mat.mtrl-sci' : 'Materials Science', 
    'mtrl-th' : 'Materials Theory',
    'cond-mat.other' : 'Other Condensed Matter',  
    'cond-mat.quant-gas' : 'Quantum Gases', 
    'cond-mat.soft' : 'Soft Condensed Matter', 
    'cond-mat.stat-mech' : 'Statistical Mechanics', 
    'cond-mat.str-el' : 'Strongly Correlated Electrons', 
    'cond-mat.supr-con' : 'Superconductivity', 
    'supr-con' : 'Superconductivity', 
    'gr-qc' : 'General Relativity and Quantum Cosmology', 
    'hep-ex' : 'High Energy Physics - Experiment', 
    'hep-lat' : 'High Energy Physics - Lattice', 
    'hep-ph' : 'High Energy Physics - Phenomenology', 
    'hep-th' : 'High Energy Physics - Theory', 
    'math-ph' : 'Mathematical Physics', 
    'nlin.AO' : 'Adaptation and Self-Organizing Systems', 
    'adap-org' : 'Adaptation and Self-Organizing Systems', 
    'nlin.CD' : 'Chaotic Dynamics',
    'chao-dyn' : 'Chaotic Dynamics',
    'nlin.CG' : 'Cellular Automata and Lattice Gases', 
    'comp-gas' : 'Cellular Automata and Lattice Gases',
    'nlin.PS' : 'Pattern Formation and Solitons', 
    'patt-sol': 'Pattern Formation and Solitons', 
    'nlin.SI' : 'Exactly Solvable and Integrable Systems', 
    'solv-int' : 'Exactly Solvable and Integrable Systems',
    'nucl-ex' : 'Nuclear Experiment', 
    'nucl-th' : 'Nuclear Theory', 
    'physics.acc-ph' : 'Accelerator Physics',
    'acc-phys' :  'Accelerator Physics',
    'physics.ao-ph' : 'Atmospheric and Oceanic Physics', 
    'physics.app-ph' : 'Applied Physics', 
    'physics.atm-clus' : 'Atomic and Molecular Clusters', 
    'physics.atom-ph' : 'Atomic Physics', 
    'atom-ph' : 'Atomic Physics',
    'physics.bio-ph' : 'Biological Physics', 
    'physics.chem-ph' : 'Chemical Physics', 
    'chem-ph' : 'Chemical Physics', 
    'physics.class-ph' : 'Classical Physics', 
    'physics.comp-ph' : 'Computational Physics', 
    'physics.data-an' : 'Data Analysis, Statistics and Probability', 
    'physics.ed-ph' : 'Physics Education', 
    'physics.flu-dyn' : 'Fluid Dynamics', 
    'physics.gen-ph' : 'General Physics', 
    'physics.geo-ph' : 'Geophysics', 
    'physics.hist-ph' : 'History and Philosophy of Physics', 
    'physics.ins-det' : 'Instrumentation and Detectors', 
    'physics.med-ph' : 'Medical Physics', 
    'physics.optics' : 'Optics', 
    'physics.plasm-ph' : 'Plasma Physics', 
    'plasm-ph' : 'Plasma Physics',
    'physics.pop-ph' : 'Popular Physics', 
    'ao-sci' : 'Atmospheric-Oceanic Sciences',
    'bayes-an' : 'Bayesian Analysis',
    'physics.soc-ph' : 'Physics and Society', 
    'physics.space-ph' : 'Space Physics', 
    'quant-ph' : 'Quantum Physics', 
    'q-bio.BM' : 'Biomolecules', 
    'q-bio.CB' : 'Cell Behavior', 
    'q-bio.GN' : 'Genomics', 
    'q-bio.MN' : 'Molecular Networks', 
    'q-bio.NC' : 'Neurons and Cognition', 
    'q-bio.OT' : 'Other Quantitative Biology', 
    'q-bio.PE' : 'Populations and Evolution', 
    'q-bio.QM' : 'Quantitative Methods', 
    'q-bio.SC' : 'Subcellular Process', 
    'q-bio.TO' : 'Tissues and Organs', 
    'q-fin.CP' : 'Computational Finance', 
    'q-fin.EC' : 'Economics', 
    'q-fin.GN' : 'General Finance', 
    'q-fin.MF' : 'Mathematical Finance', 
    'q-fin.PM' : 'Portfolio Management', 
    'q-fin.PR' : 'Pricing of Securities', 
    'q-fin.RM' : 'Risk Management', 
    'q-fin.ST' : 'Statistical Finance', 
    'q-fin.TR' : 'Trading and Market Microstructure',    
    'stat.AP' : 'Applications', 
    'stat.CO' : 'Computation', 
    'stat.ME' : 'Methodology', 
    'stat.ML' : 'Machine Learning', 
    'stat.OT' : 'Other Statistics', 
    'stat.TH' : 'Statistics Theory'   
    }

def category_change(str1) : 
    return category_ch[str1]

def id_list(str1) : 
    return f'ARXIV:{str1}'

def get_citation(list1, api_key) : 

    r = requests.post(
        'https://api.semanticscholar.org/graph/v1/paper/batch',
        params={'fields':'citationCount,referenceCount,citations.publicationDate,year'},
        headers = {'x-api-key':api_key},
        json={"ids": list1}
    )

    return r.json()

def get_citation_graph(data) : 
    temp = {}

    if data['citations'] :    

        for citation in data['citations'] : 

            if citation['publicationDate'] : 
                citation_year_month = citation['publicationDate'][:7]
                citation_week = str((int(citation['publicationDate'][8:])//7)+1)
                citation_time = citation_year_month + '-' + citation_week

                if citation_time not in temp : 
                    temp[citation_time] = 1
                else : 
                    temp[citation_time] += 1
    
        temp = dict(sorted(temp.items()))

        time_values = list(temp.values())
        time_values_cumsum = np.cumsum(time_values)
        
        for a, b in zip(temp.keys(), time_values_cumsum) : 
            temp[a] = str(b)

    result = json.dumps(temp)

    return result

def citation_list_interval(list1, int1) : 

    quote_dates = json.loads(list1)

    quarterly_quotes = defaultdict(int)
    max_quotes = 0

    for date_str in quote_dates:
        date = datetime.strptime(date_str, '%Y-%m-%d')

        year_quarter = (date.year, (date.month - 1) // int1 + 1)
        quarterly_quotes[year_quarter] += int(quote_dates.get(date_str)) - max_quotes
        max_quotes = int(quote_dates.get(date_str))
    
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

def str_change(str1) : 
    return f'"{str1}"'

################################
def get_CSV(target_date1) : 

    show_time('CSV START!!!')

    target_date = target_date_convert(target_date1)

    dataframe = {
        'id' : [], 
        'title': [],
        'updated_year': [],
        'abstract': [],
        'categories' : [],
        'author1' : [],
    }

    data = get_data()

    for i, paper in enumerate(data):
        paper = json.loads(paper)
        try:
            date = paper['update_date'][:7]
            title = paper['title'].replace('\n ', '')
            category = paper['categories'].split(' ')[0]
            author = ' '.join(paper['authors_parsed'][0][-2::-1]).strip()

            if date == target_date :
                dataframe['id'].append(paper['id'])
                dataframe['title'].append(title) 
                dataframe['updated_year'].append(int(date[:4])) 
                dataframe['abstract'].append(paper['abstract'])
                dataframe['categories'].append(category_change(category)) 
                dataframe['author1'].append(author) 
                    
        except: pass

    df = pd.DataFrame(dataframe)
    print(df.columns)
    print(len(df))

    del dataframe
    
    df_name = f'/home/dohyun/Final_P/csv_files/databases_{target_date}.csv'
    df.to_csv(df_name, header = True, index = False )
    print(df_name)

    del df

    show_time('CSV DONE!!!')
################################
  
################################
def get_citation_reference(target_date1, semantic_api_key) : 

    show_time('Citation & Reference START!!!')

    target_date = target_date_convert(target_date1)
    df_name = f'/home/dohyun/Final_P/csv_files/databases_{target_date}.csv'

    index_list = []

    df = pd.read_csv(df_name)
    print(df_name)

    for i in range(0, len(df), 500) :
        df1 = df[i: i+500]

        df1_id_list = list(df1['id'].apply(id_list)) # 
        df1_id_result = get_citation(df1_id_list, semantic_api_key)

        for j, data in enumerate(df1_id_result) : 

            if data : 
                result = get_citation_graph(data)

                df.loc[i+j, 'citation_count'] = int(data['citationCount'])
                df.loc[i+j, 'reference_count'] = int(data['referenceCount'])
                df.loc[i+j, 'published_year'] = int(data['year'])
                df.loc[i+j, 'citation_graph'] = result

            else : 
                index_list.append(i+j)

        print(f'Wait! {i+500}th data Done!')
        os.system('sleep 3')

    df = df.drop(index_list)
    df = df.dropna(subset=['citation_count'], how='any', axis=0)
    df = df.dropna(subset=['reference_count'], how='any', axis=0)
    df = df.dropna(subset=['published_year'], how='any', axis=0)

    print(f'Before Length of DF : {len(df)}')

    condition = df['citation_graph'] == '{}'

    print(f'No citation data : {condition.sum()}')

    df.drop(df[condition].index, inplace=True)

    print(f'After Length of DF : {len(df)}')

    print(df.columns)

    df.to_csv(df_name, header = True, index = False )

    del df

    show_time('Citation & Reference Done!!!')
################################

#################################
def import_dataset(target_date1, sql_user, sql_password, sql_port) : 
    
    show_time('IMPORT START!!!')

    target_date = target_date_convert(target_date1)

    engine = create_engine(f'mysql+pymysql://{sql_user}:{sql_password}@localhost:{sql_port}/final_project')
    conn = engine.connect()

    db = pymysql.connect(
        host = 'localhost',  # DATABASE_HOST
        port = int(sql_port),
        user = sql_user,  # DATABASE_USERNAME
        passwd = sql_password,  # DATABASE_PASSWORD
        db = 'final_project',  # DATABASE_NAME
        charset = 'utf8'
    )

    table_name = 'PaperInfo'

    table_sql = f'SELECT id FROM {table_name};'
    table_id = pd.read_sql(table_sql, db)
    
    cursor = db.cursor()

    file_path = f'/home/dohyun/Final_P/csv_files/databases_{target_date}.csv'
    data = pd.read_csv(file_path)
    print(file_path)
    
    data.drop_duplicates(inplace = True)

    data_id = pd.DataFrame(data['id'])

    temp_id = pd.concat([table_id, data_id])

    if temp_id.duplicated().any():
        show_time('Duplicated Database Start!!!')
        print(temp_id[temp_id.duplicated()])

        duplicated_id_df = pd.DataFrame(temp_id[temp_id.duplicated()])
        duplicated_id_df['id'] = duplicated_id_df['id'].apply(str_change)
        duplicated_id_list = ','.join(duplicated_id_df['id'])

        duplicated_sql = f'Delete FROM {table_name} where id in ({duplicated_id_list})'
        cursor.execute(duplicated_sql)

        db.commit()
        
        show_time('Duplicated Database Done!!!')

    print(len(data))

    data.to_sql(table_name, con = conn, if_exists='append', index=False)
    
    del data
    
    db.commit()

    conn.close()
    db.close()

    os.remove(file_path)

    show_time('IMPORT DONE!!!')
################################