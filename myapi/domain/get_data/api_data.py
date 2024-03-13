from fastapi import APIRouter, HTTPException

from database import SessionLocal
from models import PaperInfo
# from domain.chatbot import get_requests
# domain/chatbot.pymove to Ignore_to_Push dir

from pydantic import BaseModel


from sqlalchemy import func


router = APIRouter(
    prefix="/api/data",
)

@router.get("/get_data")
def get_data():
    db = SessionLocal()
    _data_list = db.query(PaperInfo).order_by(func.rand()).limit(10).all() # 랜덤 10개 추출
    
    db.close()
    return _data_list

@router.get("/get_data_id")
def get_data_id(get_id : str):
    db = SessionLocal()
    _data_list = db.query(PaperInfo).filter(PaperInfo.id == get_id).first()

    if _data_list is None : 
        raise HTTPException(status_code = 404, detail = "Not Found")

    db.close()
    return _data_list

'''
# -----------------------------------------------------------------------------------------------------------------------------
# question

@router.get("/get_question/{query}")
def get_question(query : str):


    data = get_requests(paper_id = paper_id, query = query) # 이게 된다는 가정 하

    _answer = data['prediction'].split('\n\n')[1]
    _Reference = data['prediction'].split('\n\n')[3:]

    chatbot = {'answer' : _answer, 'Reference' : _Reference}

    return chatbot

# ----------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------------------------------------
# chatbot

@router.post("/get_paper/{paperId}")
def get_chatbot(paperId : str):

    print(paperId, 'chatbot1')

    result ={'answer' : 'The text discusses the comparison between the original Collins-Soper-Sterman (CSS) resummation scheme and the CFG scheme in the context of prompt diphoton production cross sections at Tevatron and LHC energies. It highlights differences in the B and C functions, the inclusion of universal B and C depending on the type of ident, and the numerical smallness of differences between the schemes in production. The NLO expansion of the spin-flip resummed cross section generates terms proportional to Σg(θ∗, ϕ∗) ∝ cos 2ϕ. Resummation is crucial for predicting distributions in production and estimating the effects of physical QT experimental acceptance (Bal0704 pages 9-10, Bal0704 pages 5-6, Bal0704 pages 3-4).',
                'Reference' : '(Bal0704 pages 9-10): Balázs, C., et al. "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies." ANL-HEP-PR-07-12, arXiv:0704.0001, 3 May 2007. 2. (Bal0704 pages 5-6): Balázs, C., et al. "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies." ANL-HEP-PR-07-12, arXiv:0704.0001, 3 May 2007. 3. (Bal0704 pages 3-4): Balázs, C., et al. "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies." ANL-HEP-PR-07-12, arXiv:0704.0001, 3 May 2007.'}

# ----------------------------------------------------------------------------------------------------------------------------

@router.get("/get_chatbot")
def get_chatbot():


    result = {'answer' : 'The text discusses the comparison between the original Collins-Soper-Sterman (CSS) resummation scheme and the CFG scheme in the context of prompt diphoton production cross sections at Tevatron and LHC energies. It highlights differences in the B and C functions, the inclusion of universal B and C depending on the type of ident, and the numerical smallness of differences between the schemes in production. The NLO expansion of the spin-flip resummed cross section generates terms proportional to Σg(θ∗, ϕ∗) ∝ cos 2ϕ. Resummation is crucial for predicting distributions in production and estimating the effects of physical QT experimental acceptance (Bal0704 pages 9-10, Bal0704 pages 5-6, Bal0704 pages 3-4).',
                'Reference' : '(Bal0704 pages 9-10): Balázs, C., et al. "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies." ANL-HEP-PR-07-12, arXiv:0704.0001, 3 May 2007. 2. (Bal0704 pages 5-6): Balázs, C., et al. "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies." ANL-HEP-PR-07-12, arXiv:0704.0001, 3 May 2007. 3. (Bal0704 pages 3-4): Balázs, C., et al. "Calculation of prompt diphoton production cross sections at Tevatron and LHC energies." ANL-HEP-PR-07-12, arXiv:0704.0001, 3 May 2007.'}

    return result

# ----------------------------------------------------------------------------------------------------------------------------

'''