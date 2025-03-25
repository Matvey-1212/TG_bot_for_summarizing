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
    sum_class: str

router = APIRouter()

@router.post("/predict/", response_model=List[ModelOutput])
async def predict(input_data: List[ModelInput], model = Depends(get_model)):
    
    input_list = []
    for i, item in enumerate(input_data):
        input_list.append(item.text)
        logger.debug(f"INPUT_{i}: {item.text}")
    
    prediction, class_prediction = model.predict(input_list)
    
    
    output = []
    for i, (text, text_class) in enumerate(zip(prediction, class_prediction)):
        logger.debug(f"OUTPUT_{i}: {text}")
        logger.debug(f"OUTPUT_{i}: {text_class}")
        output.append(ModelOutput(news_id=input_data[i].news_id, text=input_data[i].text, summary=text, sum_class=text_class))
    
    
    return output