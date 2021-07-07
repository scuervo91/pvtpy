from pvtpy.black_oil.correlations import critical_properties, critical_properties_correction, z_factor, rhog
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum
import pandas as pd 
import numpy as np
from scipy.interpolate import interp1d

#local imports

class TemperatureUnits(str, Enum):
    farenheit = 'farenheit'
    celcius = 'celcius'
    kelvin = 'kelvin'
    
class PressureUnits(str, Enum):
    psi = 'psi'
    kpa = 'kpa'
    bar = 'bar'
    
class PVT(BaseModel):
    pressure: List[float] = Field(...)
    fields: Dict[str,List[float]] = Field(...)
    pressure_unit: PressureUnits = Field(PressureUnits.psi)
      
    @validator('pressure')
    def check_array_pressures_order(cls, v,values):
        diff = np.diff(np.array(v))
        if not any([np.all(diff>0),np.all(diff<0)]):
            raise ValueError('Pressure must be ordered')
        return v
    
    @validator('fields')
    def check_length_fields(cls,v,values):
        for field in v:
            assert len(v[field]) == len(values['pressure']), f'{field} has not the same length than pressure'
        return v
    
    class Config:
        extra = 'forbid'
        validate_assignment = True
        
    def df(self):
        d = self.dict()
        _df = pd.DataFrame(d['fields'], index=d['pressure'])
        _df.index.name = 'pressure'
        
        return _df
    
    def interpolate(self, value, cols=None):
        p = np.atleast_1d(value)
        
        int_dict={}
        
        int_cols = list(self.fields.keys()) if cols is None else cols
        
        for i in int_cols:
            int_dict[i] = interp1d(self.pressure,self.fields[i],bounds_error=False,fill_value='extrapolate')(p)

        int_df = pd.DataFrame(int_dict, index=p)
        int_df.index.name = 'pressure'
        return int_df 