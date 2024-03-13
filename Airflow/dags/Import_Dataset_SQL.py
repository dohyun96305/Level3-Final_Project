from datetime import datetime, timedelta

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.models.variable import Variable

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

import sys
sys.path.append("/home/dohyun/Final_P")

from databases_import import get_CSV, get_keyword, import_dataset, get_citation_reference

default_args = {
    'depends_on_past' : True, 
    'owner' : 'dohyun', 
    'retries' : 3, 
    'retry_delay' : timedelta(minutes=5)
}

api_key = Variable.get("semantic_api_key") # Airflow - site - variables 추가
sql_user = Variable.get("sql_user")
sql_password = Variable.get("sql_password")
sql_port = Variable.get("sql_port")

with DAG(
    dag_id = 'Import_Dataset_SQL',
    description = 'Import SQL with Target_date',
    start_date = datetime(2024, 2, 1),
    schedule_interval = '@monthly',
    default_args = default_args,
    tags = ['my_dags'],
) as dag :
    
    t1 = BashOperator(
        task_id = 'Get_Json',
        bash_command = """
        export KAGGLE_USERNAME=dohyunyoon
        export KAGGLE_KEY=86c3f0d0573c48c2b52cf44fc3b6c6a3
        kaggle datasets download Cornell-University/arxiv \-p /home/dohyun/Final_P/arxiv/ --unzip
        """,

    )

    t2 = PythonOperator(
        task_id = 'Get_Csv',
        python_callable = get_CSV,
        op_args = ['{{execution_date}}'],

    )
    
    t3 = PythonOperator(
        task_id = 'Get_Keyword',
        python_callable = get_keyword,
        op_args = ['{{execution_date}}'],

    )

    t4 = PythonOperator(
        task_id = 'Get_Citation_Reference',
        python_callable = get_citation_reference,
        op_args = ['{{execution_date}}', api_key],

    )

    t5 = PythonOperator(
        task_id = 'Import_MySQL',
        python_callable = import_dataset,
        op_args = ['{{execution_date}}', sql_user, sql_password, sql_port],

    )

    t1 >> t2 >> t3 >> t4 >> t5

'''    

    t6 = BashOperator(
        task_id = 'Delelte_Json',
        bash_command = """
        rm /home/dohyun/Final_P/arxiv/arxiv-metadata-oai-snapshot.json
        """,

    )

'''
