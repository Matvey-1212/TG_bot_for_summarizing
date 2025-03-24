from pydantic import BaseModel
from typing import Dict

class ModelInput(BaseModel):
    texts: Dict[str, str]