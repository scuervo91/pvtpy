from logging import critical
import numpy as np
import pandas as pd
from pydantic import BaseModel, constr, Field
import os 

#local imports
from ..units import Pressure, Temperature

#upload table property list
file_dir = os.path.dirname(__file__)
components_path = os.path.join(file_dir,'components_properties1.csv')
properties_df = pd.read_csv(components_path)


class Antoine(BaseModel):
    a: float
    b: float
    c: float
    
    def vapor_pressure(self, t: Temperature, pressure_unit='psi') -> Pressure:
        a = self.a
        b = self.b
        c = self.c
        
        temp_kelvin = t.to('kelvin')
        
        right_part = a - (b / (temp_kelvin.value + c))
        pressure_value = np.power(10,right_part)
        pressure_obj = Pressure(value = pressure_value, unit = 'bar')
        return pressure_obj.convert_to(pressure_unit)
        

class Component(BaseModel):
    name: str = Field(..., description='Component name')
    formula: str = Field(None, description='Component formula')
    iupac_key: constr(regex = r'^([0-9A-Z\-]+)$') = Field(None, description='Component IUPAC key')
    cas: constr(regex = r'\b[1-9]{1}[0-9]{1,6}-\d{2}-\d\b') = Field(None, description='Component CAS')
    molecular_weight: float = Field(...,gt=0,description='Component molecular weight')
    critical_pressure: Pressure = Field(None, description='Component critical pressure')
    critical_temperature: Temperature = Field(None, description='Component critical temperature')
    antoine_coefficients: Antoine = Field(None, description='Component Antoine coefficients')
    
    def __init__(self,**kwargs):
        super().__init__(
            name = kwargs.get('name'),
            formula = kwargs.get('formula',None),
            iupac_key = kwargs.get('iupac_key',None),
            cas = kwargs.get('cas',None),
            molecular_weight = kwargs.get('molecular_weight',None),
            critical_temperature = Temperature(value=kwargs.get('critical_temperature'),unit = kwargs.get('critical_temperature_unit')) if 'critical_temperature' in kwargs else None,
            critical_pressure = Pressure(value=kwargs.get('critical_pressure'),unit = kwargs.get('critical_pressure_unit')) if 'critical_pressure' in kwargs else None,
            antoine_coefficients = Antoine(a = kwargs.get('antoine_a'),b = kwargs.get('antoine_b'),c = kwargs.get('antoine_c')) if 'antoine_a' in kwargs else None
        )

    
