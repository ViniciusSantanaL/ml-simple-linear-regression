from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

app = FastAPI()

class RequestBody(BaseModel):
    horas_estudo: float

modelo_regre = joblib.load('./modelo_regressao.pkl')

@app.post('/predics')
def predict(data: RequestBody):
    input_feature = [[data.horas_estudo]]

    y_pred = modelo_regre.predict(input_feature)[0].astype(int)

    return { 'pontuacao_teste': int(y_pred)}