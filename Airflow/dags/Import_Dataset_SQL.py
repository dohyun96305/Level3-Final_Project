from datetime import datetime, timedelta

from airflow import DAG
from airflow.models.variable import Variable

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import os
import sys
sys.path.append(os.path.abspath('/home/dohyun/Final_P/'))

from utils.databases_import import get_CSV, import_dataset, get_citation_reference
from utils.paper_models_train import model_train_save

default_args = {
    'depends_on_past' : True, 
    'owner' : 'dohyun', 
    'retries' : 3, 
    'retry_delay' : timedelta(minutes=5)
}

kaggle_api_key = Variable.get("kaggle_api_key")
kaggle_user = Variable.get("kaggle_username")

api_key = Variable.get("semantic_api_key") # Airflow - site - variables 추가
sql_user = Variable.get("sql_user")
sql_password = Variable.get("sql_password")
sql_port = Variable.get("sql_port")

with DAG(
    dag_id = 'Import_Dataset_SQL',
    description = 'Import SQL with Target_date',
    start_date = datetime(2020, 1, 1),
    schedule_interval = '@monthly',
    default_args = default_args,
    tags = ['my_dags'],
) as dag :
    
    t1 = BashOperator(
        task_id = 'Get_Json',
        bash_command = f"""
        export KAGGLE_USERNAME={kaggle_user}
        export KAGGLE_KEY={kaggle_api_key}
        kaggle datasets download Cornell-University/arxiv \-p /home/dohyun/Final_P/arxiv/ --unzip
        """,

    )

    t2 = PythonOperator(
        task_id = 'Get_Csv',
        python_callable = get_CSV,
        op_args = ['{{execution_date}}'],

    )

    t3 = PythonOperator(
        task_id = 'Get_Citation_Reference',
        python_callable = get_citation_reference,
        op_args = ['{{execution_date}}', api_key],

    )

    t4 = PythonOperator(
        task_id = 'Import_MySQL',
        python_callable = import_dataset,
        op_args = ['{{execution_date}}', sql_user, sql_password, sql_port],

    )

    t5 = PythonOperator(
        task_id = 'Model_retraining',
        python_callable = model_train_save,
        op_args = ['{{execution_date}}', sql_user, sql_password, sql_port],

    )
    
    t1 >> t2 >> t3 >> t4 >> t5
