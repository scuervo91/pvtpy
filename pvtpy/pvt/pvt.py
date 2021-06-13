from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from enum import Enum
import pandas as pd 
import numpy as np
from scipy.interpolate import interp1d

#local imports
from .components import properties_df

class TemperatureUnits(str, Enum):
    farenheit = 'farenheit'
    celcius = 'celcius'
    
class PressureUnits(str, Enum):
    psi = 'psi'
    kpa = 'kpa'
    
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
    
    
class JoinItem(str, Enum):
    id = 'id'
    name = 'name'
    
class Chromatography(BaseModel):
    join: JoinItem = Field(JoinItem.name)
    components: List[str] = Field(...)
    mole_fraction: List[float] = Field(...)
    
    @validator('components', pre=True)
    def check_components_merge(cls, v, values):
        if values['join'] == JoinItem.id:
            for id in v:
                assert id>0 and id<= properties_df.shape[0]+1
            return v
        if values['join'] == JoinItem.name:
            for name in v:
                assert name in properties_df['name'].tolist()
            return v
        
    @validator('mole_fraction', each_item=True)
    def check_values_mole_fraction(cls,v):
        assert v >= 0 and v <= 1, f'{v} is either less than 0 or greater than 1'
        return v

    @validator('mole_fraction')
    def check_length_fields(cls,v,values):
        assert len(v) == len(values['components'])
        return v
        

    class Config:
        extra = 'forbid'
        validate_assignment = True
        
    def df(self, normalize=True):
        if normalize:
            mf = np.array(self.mole_fraction)
            mfn = mf / mf.sum()
            _df = pd.DataFrame({'mole_fraction':mfn}, index=self.components)
        else:
            _df = pd.DataFrame({'mole_fraction':self.mole_fraction}, index=self.components)
        
        _merged = _df.merge(properties_df, how='inner',left_index=True,right_on=self.join)
        
        return _merged[['id','name','formula','mole_fraction','mw','ppc','tpc']]
    
    def mwa(self, normalize=True):
        df = self.df(normalize=normalize)
        return np.dot(df['mole_fraction'].values, df['mw'].values)
    
    def gas_sg(self, normalize=True):
        mwa = self.mwa(normalize=normalize)
        return mwa / 28.96
        
