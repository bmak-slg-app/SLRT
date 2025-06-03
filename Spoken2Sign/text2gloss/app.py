from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from service import T2GService

app = FastAPI()


class LanguageModel(BaseModel):
    id: str
    name: str
    language: str


available_languages = [
    # LanguageModel(id="phoenix",
    #               name="RWTH-PHOENIX-Weather 2014 T", language="German Sign Language 德國手語"),
    LanguageModel(id="csl", name="Chinese Sign Language Corpus",
                  language="Chinese Sign Language 中国手语"),
    LanguageModel(id="tvb", name="TVB-HKSL-News",
                  language="Hong Kong Sign Language 香港手語"),
]

t2g_service = {language.id: T2GService(
    config=f"./configs/T2G_{language.id}.yaml") for language in available_languages}


class T2GRequest(BaseModel):
    model: str
    text: str


class T2GResponse(BaseModel):
    gloss: str


@app.post("/")
def t2g(request: T2GRequest) -> T2GResponse:
    if request.model not in [language.id for language in available_languages]:
        raise HTTPException(
            status_code=400,
            detail="Model not found"
        )
    return T2GResponse(gloss=t2g_service[request.model].translate(request.text))


@app.get("/languages")
def list_languages() -> list[LanguageModel]:
    return available_languages
