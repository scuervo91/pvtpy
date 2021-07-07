from enum import Enum
from pydantic import BaseModel, Field
from typing import Union, List
import pandas as pd

class TemperatureUnits(str, Enum):
    farenheit = 'farenheit'
    celcius = 'celcius'
    kelvin = 'kelvin'
    rankine = 'rankine'
    
temperature_converter_dict = {
    'from':['farenheit','farenheit','farenheit','farenheit', 'celcius','celcius','celcius','celcius', 'kelvin','kelvin','kelvin','kelvin','rankine','rankine','rankine','rankine'],
    'to': ['farenheit', 'celcius', 'kelvin', 'rankine','farenheit', 'celcius', 'kelvin', 'rankine','farenheit', 'celcius', 'kelvin', 'rankine','farenheit', 'celcius', 'kelvin', 'rankine'],
    'value':[
        lambda x: x, lambda x: (x-32)*5/9, lambda x: (x-32)*5/9 + 273.15, lambda x: x + 459.67,
        lambda x: (x * 9/5) + 32, lambda x: x, lambda x: x + 273.15, lambda x: (x * 9/5) + 32 +459.67,
        lambda x: ((x-273.15) * 9/5) + 32, lambda x: x - 273.15, lambda x: x, lambda x: ((x-273.15) * 9/5) + 32 +459.67,
        lambda x: x - 459.67, lambda x: ((x-459.67)-32)*5/9, lambda x: ((x - 459.67) - 32) *(5/9) +273.15, lambda x: x
             ] 
}

temperature_converter_matrix = pd.DataFrame(temperature_converter_dict).pivot(index='from',columns='to',values='value')

def temperature_converter(value:float, From:str,To:str)->float:
    return temperature_converter_matrix.loc[From,To](value)

    
class PressureUnits(str, Enum):
    psi = 'psi'
    kpa = 'kpa'
    bar = 'bar'
    atm = 'atm'
    
pressure_converter_dict = {
    'from': ['psi','psi','psi','psi', 'kpa','kpa','kpa','kpa','bar','bar','bar','bar', 'atm','atm','atm','atm'],
    'to': ['psi','kpa', 'bar','atm','psi','kpa', 'bar','atm','psi','kpa', 'bar','atm','psi','kpa', 'bar','atm',],
    'value': [1.,6.89476, 0.0689476,0.068046, 0.145038,1,0.01,0.00986923,14.5038,100,1,0.986923,14.6959,101.325,1.01325,1]
}

pressure_converter_matrix = pd.DataFrame(pressure_converter_dict).pivot(index='from',columns='to',values='value')

def pressure_converter(value:float, From:str,To:str)->float:
    return pressure_converter_matrix.loc[From,To]*value
    
class Pressure(BaseModel):
    value: Union[float, List[float]]
    unit: PressureUnits = Field(PressureUnits.psi)
    
class Temperature(BaseModel):
    value: Union[float, List[float]]
    unit: PressureUnits = Field(TemperatureUnits.farenheit)