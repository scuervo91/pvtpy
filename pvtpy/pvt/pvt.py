from pydantic import BaseModel, Field, NoneIsAllowedError, validator
from typing import List, Optional, Dict
from enum import Enum
import pandas as pd 
import numpy as np
from scipy.interpolate import interp1d

class TemperatureUnits(str, Enum):
    farenheit = 'farenheit'
    celcius = 'celcius'
    
class PressureUnits(str, Enum):
    psi = 'psi'
    kpa = 'kpa'
    
class PVT(BaseModel):
    pressure: List[float] = Field(...)
    fields: Dict[str,List[float]] = None
    pressure_unit: PressureUnits = PressureUnits.psi
    
    @validator('pressure', each_item=True, pre=True)
    def check_array_pressure(cls, v):
        assert v > 0
        return v
    
    @validator('pressure')
    def check_array_pressures(cls, v):
        assert np.all(np.diff(v)>0) or np.all(np.diff(v)<0)
        return v
    
    @validator('fields', each_item=True)
    def check_length_fields(cls,v,values):
        assert len(v) == len(values['pressure'])
        return v
        
    
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
