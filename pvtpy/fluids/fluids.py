from pydantic import BaseModel, Field
from typing import List, Optional


class InitialConditions(BaseModel):
    pressure: float = Field(...,gt=0)
    temperature: float = Field(...,gt=0)
    pressure_unit: PressureUnits = PressureUnits.psi
    temperature_unit: PressureUnits = TemperatureUnits.farenheit


class Oil(BaseModel):
    api: float = Field(...,gt=0)
    sulphur: Optional[float] = Field(None, gt=0)
    bubble_pressure: Optional[float] = Field(None, gt=0)
    rsb: Optional[float] = Field(None, gt=0)
    initial_conditions: InitialConditions = Field(...)

