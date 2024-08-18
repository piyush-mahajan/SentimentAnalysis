from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

app = FastAPI()

class RequestModel(BaseModel):
    input: str

@app.post("/sentiment")
def get_response(request: RequestModel):
    prompt = request.input

    response = pipe(prompt)

    label = response[0]["label"]
    score = response[0]["score"]
    return f"The '{prompt}' input is {label} with a score of {score}"