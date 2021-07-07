
from pydantic import BaseModel, Field
from ..pvt import PVT
from ..units import TemperatureUnits,PressureUnits, Pressure, Temperature
from ..compositional import Chromatography

class InitialConditions(BaseModel):
    pressure: Pressure = Field(...)
    temperature: Temperature = Field(...)

class FluidBase(BaseModel):
    initial_conditions: InitialConditions = Field(...)
    pvt: PVT = Field(None)
    chromatography: Chromatography = Field(None)

    class Config:
        validate_assignment = True