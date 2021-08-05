from logging import critical
from pvtpy.eos import peng_robinson, redlich_kwong, soave_redlich_kwong
from pvtpy.units.units import CriticalProperties
import numpy as np
import pandas as pd
from pydantic import BaseModel, constr, Field, parse_obj_as
from typing import List, Dict, Union
import os 

#local imports
from ..units import Pressure, Temperature
from ..eos import VanDerWalls, RedlichKwong, SoaveRedlichKwong, PengRobinson

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
    molecular_weight: float = Field(None,gt=0,description='Component molecular weight')
    specific_gravity: float = Field(None,gt=0,description='Component specific_gravity')
    critical_pressure: Pressure = Field(None, description='Component critical pressure')
    critical_temperature: Temperature = Field(None, description='Component critical temperature')
    antoine_coefficients: Antoine = Field(None, description='Component Antoine coefficients')
    van_der_walls: VanDerWalls = Field(VanDerWalls())
    redlich_kwong: RedlichKwong = Field(RedlichKwong())
    soave_redlich_kwong: SoaveRedlichKwong = Field(SoaveRedlichKwong())
    peng_robinson: PengRobinson = Field(PengRobinson())
    mole_fraction: float = Field(None, ge=0, le=1)
    params: Dict[str, Union[float,int,str,Pressure, Temperature]] = Field(None, description='Component parameters')
    
    def __init__(self,**kwargs):
        super().__init__(
            name = kwargs.pop('name',None),
            formula = kwargs.pop('formula',None),
            iupac_key = kwargs.pop('iupac_key',None),
            iupac = kwargs.pop('iupac',None),
            cas = kwargs.pop('cas',None),
            specific_gravity = kwargs.pop('specific_gravity',None),
            molecular_weight = kwargs.pop('molecular_weight',None),
            critical_temperature = Temperature(value=kwargs.pop('critical_temperature'),unit = kwargs.pop('critical_temperature_unit')) if 'critical_temperature' in kwargs else None,
            critical_pressure = Pressure(value=kwargs.pop('critical_pressure'),unit = kwargs.pop('critical_pressure_unit')) if 'critical_pressure' in kwargs else None,
            antoine_coefficients = Antoine(a = kwargs.pop('antoine_a'),b = kwargs.pop('antoine_b'),c = kwargs.pop('antoine_c')) if 'antoine_a' in kwargs else None,
            mole_fraction = kwargs.pop('mole_fraction',None),
            params = kwargs.pop('params',kwargs) if bool(kwargs) else None
        )
        
    def df(self, pressure_unit='psi', temperature_unit='farenheit'):
        d = self.dict(
            exclude={'critical_pressure', 'critical_temperature', 'antoine_coefficients','params','name'},
            exclude_none=True
        )
                
        if self.critical_pressure is not None:
            d.update({'critical_pressure': self.critical_pressure.convert_to(pressure_unit).value})

        if self.critical_temperature is not None:
            d.update({'critical_temperature': self.critical_temperature.convert_to(temperature_unit).value})

        if self.params is not None:
            for i in self.params:
                if isinstance(self.params[i],(Pressure,Temperature)):
                    d.update({i:self.params[i].value})
                else:
                    d.update({i:self.params[i]})
        
        if self.redlich_kwong.a is not None and self.redlich_kwong.b is not None:
            d.update({
                'rk_a':self.redlich_kwong.a,
                'rk_b':self.redlich_kwong.b,
            })

        if self.van_der_walls.a is not None and self.van_der_walls.b is not None:
            d.update({
                'vdw_a':self.vdw.a,
                'vdw_b':self.vdw.b,
            })
            
        if self.soave_redlich_kwong.a is not None and self.soave_redlich_kwong.b is not None:
            d.update({
                'srk_a':self.soave_redlich_kwong.a,
                'srk_b':self.soave_redlich_kwong.b,
                'srk_alpha':self.soave_redlich_kwong.alpha,
                
            })
            
        if self.peng_robinson.a is not None and self.peng_robinson.b is not None:
            d.update({
                'pr_a':self.peng_robinson.a,
                'pr_b':self.peng_robinson.b,
                'pr_alpha':self.peng_robinson.alpha,
                
            })

        return pd.Series(d, name = self.name)
    
    def critical_properties(self):
        return  CriticalProperties(
            critical_pressure = self.critical_pressure,
            critical_temperature = self.critical_temperature)

    def vapor_pressure(self, temperature: Temperature, pressure_unit='psi'):
        
        vp_value = self.antoine_coefficients.vapor_pressure(
            temperature, 
            pressure_unit=pressure_unit
        )
        
        # dict_vapor_pressure  = {'vapor_pressure':vp_value,'vapor_temperature':temperature}
        # if self.params is None:
        #     self.params = dict_vapor_pressure
        # else:
        #     self.params.update(dict_vapor_pressure)

        return vp_value
        
def component_from_name(name:str):
    if name not in properties_df.index.tolist():
        raise ValueError(f'{name} not found in database')
    
    comp_df = properties_df.loc[[name],:].reset_index().rename(columns={'index':'name'})
    
    return parse_obj_as(Component,comp_df.to_dict(orient='records')[0])
        
        

        