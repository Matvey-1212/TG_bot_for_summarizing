from pydantic import BaseModel
from typing import Dict

class ModelOutput(BaseModel):
    prediction: Dict[str, str]