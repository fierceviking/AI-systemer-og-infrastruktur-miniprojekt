from pydantic import BaseModel

class Variables(BaseModel):
    hour: float
    day: float
    month: float