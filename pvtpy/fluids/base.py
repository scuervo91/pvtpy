
from pydantic import BaseModel, Field
from ..pvt import PVT, TemperatureUnits,PressureUnits, Chromatography

class InitialConditions(BaseModel):
    pressure: float = Field(...,gt=0)
    temperature: float = Field(...,gt=0)
    pressure_unit: PressureUnits = PressureUnits.psi
    temperature_unit: PressureUnits = TemperatureUnits.farenheit



class FluidBase(BaseModel):
    initial_conditions: InitialConditions = Field(...)
    pvt: PVT = Field(None)
    chromatography: Chromatography = Field(None)

    class Config:
        validate_assignment = True