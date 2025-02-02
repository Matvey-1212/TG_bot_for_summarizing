from fastapi import APIRouter
from app.schemas.request import ModelInput
from app.schemas.response import ModelOutput
from app.models.model import model
from app.core.logging import logger

router = APIRouter()

@router.post("/predict/", response_model=ModelOutput)
async def predict(input_data: ModelInput):
    logger.debug(f"Получен текст: {input_data.texts}")
    prediction = model.predict(input_data.texts)
    logger.debug(f"Получен предит: {prediction}")
    return {"prediction": prediction}