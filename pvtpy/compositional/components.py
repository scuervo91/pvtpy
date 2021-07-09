from logging import critical
import numpy as np
import pandas as pd
from pydantic import BaseModel, constr, Field, parse_obj_as
from typing import List, Dict, Union
import os 

#local imports
from ..units import Pressure, Temperature

#upload table property list
file_dir = os.path.dirname(__file__)
components_path = os.path.join(file_dir,'components_properties1.csv')
properties_df = pd.read_csv(components_path,index_col='name')

class Antoine(BaseModel):
    a: float
    b: float
    c: float
    
    def vapor_pressure(self, t: Temperature, pressure_unit='psi') -> Pressure:
        a = self.a
        b = self.b
        c = self.c
        
        temp_kelvin = t.convert_to('kelvin')
        
        right_part = a - (b / (temp_kelvin.value + c))
        pressure_value = np.power(10,right_part)
        pressure_obj = Pressure(value = pressure_value, unit = 'bar')
        return pressure_obj.convert_to(pressure_unit)
        

class Component(BaseModel):
    name: str = Field(..., description='Component name')
    formula: str = Field(None, description='Component formula')
    iupac_key: constr(regex = r'^([0-9A-Z\-]+)$') = Field(None, description='Component IUPAC key')
    iupac: str = Field(None, description='Component IUPAC name')
    cas: constr(regex = r'\b[1-9]{1}[0-9]{1,6}-\d{2}-\d\b') = Field(None, description='Component CAS')
    molecular_weight: float = Field(...,gt=0,description='Component molecular weight')
    critical_pressure: Pressure = Field(None, description='Component critical pressure')
    critical_temperature: Temperature = Field(None, description='Component critical temperature')
    antoine_coefficients: Antoine = Field(None, description='Component Antoine coefficients')
    mole_fraction: float = Field(None, ge=0, le=1)
    params: Dict[str, Union[int,float,str,Pressure, Temperature]] = Field(None, description='Component parameters')
    def __init__(self,**kwargs):
        super().__init__(
            name = kwargs.pop('name',None),
            formula = kwargs.pop('formula',None),
            iupac_key = kwargs.pop('iupac_key',None),
            iupac = kwargs.pop('iupac',None),
            cas = kwargs.pop('cas',None),
            molecular_weight = kwargs.pop('molecular_weight',None),
            critical_temperature = Temperature(value=kwargs.pop('critical_temperature'),unit = kwargs.pop('critical_temperature_unit')) if 'critical_temperature' in kwargs else None,
            critical_pressure = Pressure(value=kwargs.pop('critical_pressure'),unit = kwargs.pop('critical_pressure_unit')) if 'critical_pressure' in kwargs else None,
            antoine_coefficients = Antoine(a = kwargs.pop('antoine_a'),b = kwargs.pop('antoine_b'),c = kwargs.pop('antoine_c')) if 'antoine_a' in kwargs else None,
            mole_fraction = kwargs.pop('mole_fraction',None),
            params = kwargs if bool(kwargs) else None
        )
        
    def df(self):
        d = self.dict(
            exclude={'critical_pressure', 'critical_temperature', 'antoine_coefficients','params','name'},
            exclude_none=True
        )
        if self.params is not None:
            for i in self.params:
                if isinstance(self.params[i],(Pressure,Temperature)):
                    d.update({i:self.params[i].value})
                else:
                    d.update({i:self.params[i]})

        return pd.DataFrame(d, index=[self.name])
    
    
    def vapor_pressure(self, temperature: Temperature, pressure_unit='psi'):
        
        vp_value = self.antoine_coefficients.vapor_pressure(
            temperature, 
            pressure_unit=pressure_unit
        )
        
        dict_vapor_pressure  = {'vapor_pressure':vp_value,'vapor_temperature':temperature}
        if self.params is None:
            self.params = dict_vapor_pressure
        else:
            self.params.update(dict_vapor_pressure)

        
        return vp_value
    
def component_from_name(name:str):
    if name not in properties_df['name'].tolist():
        raise ValueError(f'{name} not found in database')
    
    comp_df = properties_df.loc[name,:]
    
    return parse_obj_as(Component,comp_df.dict(orient='records')[0])
        
        

        