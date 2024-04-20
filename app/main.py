from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from app.model.model import learn_predict
from app.model.model import __version__ as model_version

app = FastAPI()

class TextIn(BaseModel):
    text: str


class PredictionOut(BaseModel):
    language: str


@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    pred = learn_predict(await file.read())
    return pred

#if __name__ == "__main__":
#    uvicorn.run(app, host="localhost", port=8003)