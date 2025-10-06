from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict, Union,Literal
from enum import Enum

class GlobalInputState(BaseModel):
    id:str
    context: str
    prompt: str
    response: str

class GlobalOutputState(BaseModel):
    label: Literal["no", "intrinsic", "extrinsic"]
