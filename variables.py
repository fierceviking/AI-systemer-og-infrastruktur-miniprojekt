from pydantic import BaseModel

class Variables(BaseModel):
    hour: int
    day: int
    month: int