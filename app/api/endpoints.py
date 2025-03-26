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
    fake_news_prob: str

router = APIRouter()

@router.post("/predict/", response_model=List[ModelOutput])
async def predict(input_data: List[ModelInput], model = Depends(get_model)):
    
    input_list = []
    for i, item in enumerate(input_data):
        input_list.append(item.text)
        logger.debug(f"INPUT_{i}: {item.text}")
    
    prediction, class_prediction, fake_news_prediction = model.predict(input_list)
    
    
    output = []
    for i, (text, text_class, fake_news_prob) in enumerate(zip(prediction, class_prediction, fake_news_prediction)):
        logger.debug(f"OUTPUT_{i}_sum: {text}")
        logger.debug(f"OUTPUT_{i}_class: {text_class}")
        logger.debug(f"OUTPUT_{i}_prob: {fake_news_prob}")

        output.append(ModelOutput(news_id=input_data[i].news_id, 
                                  text=input_data[i].text, 
                                  summary=text, 
                                  sum_class=text_class,
                                  fake_news_prob=str(f'{fake_news_prob;.2f}')
                                  ))
    
    
    return output