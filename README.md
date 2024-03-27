## Level3-Final_Project (Backend) - PRQAS 
* "Paper Recommend and QA System" 
* AI 기술의 발전 및 국내 및 해외의 논문 수가 비약적으로 증가하는 상태 
    * 미국에서의 arXiv 논문 사이트 이용자 수가 1주일에 16만명
    * 국내 공학 분야 기준 2015년 부터 2만편씩 논문이 꾸준히 발간되어가고 있음

* 사용자가 keyword 검색을 통해 관심있는 논문에 대해 추천 및 Chatbot을 활용하여  
논문에 대한 QA 시스템 적용
    * keyowrd를 통한 Filtering 및 논문의 최근 피 인용수 변화를 예측하여 논문 추천

* 사용자의 논문을 찾고 이용하는 것에 대한 어려움을 해결 + 정보를 제공하는 서비스


ㄴ Data (Kaggle arXiv Dataset + SemanticScholar API)
-

* [Kaggle arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv/data)
    * [Creative Commons CC0 1.0 Universal Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)
    * 각 Paper의 저자, published_year, id 등 Metadata 수집 
    * CORNELL UNIVERSITY 에서 매주 업데이트 진행 
* [SemanticScholar API](https://www.semanticscholar.org/product/api)
    * 각 Paper의 피인용 횟수 및 증가 시점 수집
        * 각 논문을 피인용한 Paper의 Published_date를 확인해 Paper의 피인용 시점 확인

ㄴ DB (MySQL)
-

* MySQL DB 및 Table를 활용
    - 사용자의 Login Data, Paper Metadata, 사용자의 QA 대화 내용 Data 저장
* Alembic 활용
    * Database 및 Table 관련 Schema 업데이트 및 관리  


ㄴ Backend Server (FastAPI)
-

* Frontend (Reaact) 와 연동
    * Paper Metadata 및 모델 예측을 통한 TOP-10 Paper 전달
    * 사용자 Login 관련 Frontend Server와 연동 및 Authorization Token 전달
    * PRQAS 시스템 이용간 검색 및 QA 시스템 내역 연동, Database 저장

ㄴ Workflow (Airflow)
-

* Airflow DAG 활용 (@Monthly)
    * Kaggle arXiv Dataset + SemanticScholar API 릍 통해 데이터 Update,    
      데이터 전처리 및 DB Import 진행
    * DB에 추가된 데이터를 포함 모델 재학습 및 저장
* Airflow Task 실행간 필요한 API_key, Password 등 민감 정보에 대해 Variable 처리
