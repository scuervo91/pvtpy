from enum import Enum
from pydantic import BaseModel, Field
from typing import Union, List


class TemperatureUnits(str, Enum):
    farenheit = 'farenheit'
    celcius = 'celcius'
    kelvin = 'kelvin'
    
class PressureUnits(str, Enum):
    psi = 'psi'
    kpa = 'kpa'
    bar = 'bar'
    
    
    
class Pressure(BaseModel):
    value: Union[float, List[float]]
    unit: PressureUnits = Field(PressureUnits.psi)
    
class Temperature(BaseModel):
    value: Union[float, List[float]]
    unit: PressureUnits = Field(TemperatureUnits.psi)