from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from app.models.model import model
from app.core.logging import logger

class ModelInput(BaseModel):
    news_id: int
    text: str

class ModelOutput(ModelInput):
    summary: str

router = APIRouter()

@router.post("/predict/", response_model=List[ModelOutput])
async def predict(input_data: List[ModelInput]):
    
    input = []
    for item in input_data:
        input.append(item.text)
    logger.debug(f"INPUT: {input}")
    
    prediction = model.predict(input)
    logger.debug(f"OUTPUT: {prediction}")
    
    output = []
    for i, text in enumerate(prediction):
        output.append(ModelOutput(news_id=input_data[i].news_id, text=input_data[i].text, summary=text))
    
    
    return output