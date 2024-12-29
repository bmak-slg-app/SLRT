from fastapi import FastAPI
from pydantic import BaseModel
from service import T2GService

app = FastAPI()

t2g_service = T2GService()

class T2GRequest(BaseModel):
    text: str

class T2GResponse(BaseModel):
    gloss: str

@app.post("/t2g")
def t2g(request: T2GRequest) -> T2GResponse:
    return T2GResponse(gloss=t2g_service.translate(request.text))
