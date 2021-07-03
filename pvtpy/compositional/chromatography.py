from pydantic import BaseModel, Field, validator
from typing import List
from enum import Enum
import numpy as np
import pandas as pd
from .components import properties_df
from ..black_oil import correlations as cor


class JoinItem(str, Enum):
    id = 'id'
    name = 'name'

class CriticalProperties(BaseModel):
    ppc: float
    tpc: float


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
    
    def get_pseudo_critical_properties(
        self,
        correct=True, 
        correct_method = 'wichert_aziz',
        normalize=True
    ):
        df = self.df(normalize=normalize)
        _ppc = np.dot(df['mole_fraction'].values, df['ppc'].values)
        _tpc = np.dot(df['mole_fraction'].values, df['tpc'].values)
        
        if correct:
            _co2 = df.loc[df['name']=='carbon-dioxide', 'mole_fraction'].values[0] if 'carbon-dioxide' in df['name'].tolist() else 0
            _n2 = df.loc[df['name']=='nitrogen', 'mole_fraction'].values[0] if 'nitrogen' in df['name'].tolist() else 0
            _h2s = df.loc[df['name']=='hydrogen-sulfide', 'mole_fraction'].values[0] if 'hydrogen-sulfide' in df['name'].tolist() else 0
            cp_correction = cor.critical_properties_correction(ppc=_ppc, tpc=_tpc, co2=_co2, n2=_n2, h2s=_h2s, method=correct_method)
        else:
            cp_correction = {'ppc':_ppc,'tpc':_tpc}
            
        return CriticalProperties(**cp_correction)
    
    def get_z(self,pressure=14.7,temperature=60, z_method='papay', cp_correction_method='wichert_aziz', normalize=True):
        cp = self.get_pseudo_critical_properties(correct=True, correct_method=cp_correction_method,normalize=normalize)
        return cor.z_factor(p=pressure,t=temperature, ppc = cp.ppc, tpc = cp.tpc, method=z_method)

    def get_rhog(self,pressure=14.7,temperature=60, z_method='papay',rhog_method='real_gas',normalize=True):
        _ma = self.mwa(normalize=normalize)
        if rhog_method == 'ideal_gas':
            _rhog = cor.rhog(p=pressure,ma=_ma,t=temperature)
        elif rhog_method == 'real_gas':
            _z = self.get_z(pressure=pressure,temperature=temperature,z_method = z_method, normalize=normalize)
            _rhog = cor.rhog(p=pressure,ma=_ma,z=_z.values.reshape(-1), t=temperature, method=rhog_method)
        return _rhog
    
    def get_sv(self,pressure=14.7,temperature=60, z_method='papay',rhog_method='real_gas',normalize=True):
        rhog = self.get_rhog(pressure=pressure,temperature=temperature, z_method=z_method,rhog_method=rhog_method,normalize=normalize)
        rhog['sv'] = 1 / rhog['rhog']
        return rhog['sv']