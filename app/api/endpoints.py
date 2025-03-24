from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List
# from app.models.model import model
from app.models.model import get_model
from app.core.logging import logger

class ModelInput(BaseModel):
    news_id: int
    text: str

class ModelOutput(ModelInput):
    summary: str

router = APIRouter()

@router.post("/predict/", response_model=List[ModelOutput])
async def predict(input_data: List[ModelInput], model = Depends(get_model)):
    
    input_list = []
    for i, item in enumerate(input_data):
        input_list.append(item.text)
        logger.debug(f"INPUT_{i}: {item.text}")
    
    prediction = model.predict(input_list)
    
    
    output = []
    for i, text in enumerate(prediction):
        logger.debug(f"OUTPUT_{i}: {text}")
        output.append(ModelOutput(news_id=input_data[i].news_id, text=input_data[i].text, summary=text))
    
    
    return output