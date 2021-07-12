from logging import critical
from pydantic import BaseModel, Field, validator, parse_obj_as
from typing import List
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
#Local imports
from .components import properties_df, Component
from ..black_oil import correlations as cor
from ..units import Pressure, Temperature
from .equations import cost_flash, cost_flash_prime, cost_dew_point,cost_bubble_point

class JoinItem(str, Enum):
    id = 'id'
    name = 'name'

class CriticalProperties(BaseModel):
    critical_pressure: float
    critical_temperature: float
    

class Chromatography(BaseModel):
    components: List[Component] = Field(None)

    class Config:
        validate_assignment = True
        extra = 'forbid'
        
    def from_df(self, df: pd.DataFrame, name:str = None, mole_fraction:str = None):
        if name is None:
            if 'name' not in df.columns:
                raise ValueError('Column name not defined')
            name = 'name'
        df = df.set_index(name)
        
        if mole_fraction is None:
            if 'mole_fraction' not in df.columns:
                raise ValueError('Column mole_fraction not defined')
        else:
            df = df.rename(columns={mole_fraction: 'mole_fraction'})
               
        _merged = df.merge(properties_df, how='inner',left_index=True,right_index=True).reset_index().rename(columns={'index':'name'})
        

        self.components = parse_obj_as(List[Component], _merged.to_dict(orient='records'))

    def df(self, normalize=True):
        df = pd.DataFrame()
        
        for i in self.components:
            df = df.append(i.df())
            
        if normalize:
            mf = np.array(df['mole_fraction'])
            mfn = mf / mf.sum()
            df['mole_fraction'] = mfn
            
        return df
    
    def apparent_molecular_weight(self, normalize=True):
        df = self.df(normalize=normalize)
        return np.dot(df['mole_fraction'].values, df['molecular_weight'].values)

    def gas_sg(self, normalize=True):
        mwa = self.apparent_molecular_weight(normalize=normalize)
        return mwa / 28.96
    
    def get_pseudo_critical_properties(
        self,
        correct=True, 
        correct_method = 'wichert_aziz',
        normalize=True
    ):
        df = self.df(normalize=normalize)
        _ppc = np.dot(df['mole_fraction'].values, df['critical_pressure'].values)
        _tpc = np.dot(df['mole_fraction'].values, df['critical_temperature'].values)
        
        if correct:
            _co2 = df.loc[df['name']=='carbon-dioxide', 'mole_fraction'].values[0] if 'carbon-dioxide' in df['name'].tolist() else 0
            _n2 = df.loc[df['name']=='nitrogen', 'mole_fraction'].values[0] if 'nitrogen' in df['name'].tolist() else 0
            _h2s = df.loc[df['name']=='hydrogen-sulfide', 'mole_fraction'].values[0] if 'hydrogen-sulfide' in df['name'].tolist() else 0
            cp_correction = cor.critical_properties_correction(ppc=_ppc, tpc=_tpc, co2=_co2, n2=_n2, h2s=_h2s, method=correct_method)
        else:
            cp_correction = {'critical_pressure':_ppc,'critical_temperature':_tpc}
            
        return CriticalProperties(**cp_correction)
    
    def get_z(self,pressure=14.7,temperature=60, z_method='papay', cp_correction_method='wichert_aziz', normalize=True):
        cp = self.get_pseudo_critical_properties(correct=True, correct_method=cp_correction_method,normalize=normalize)
        return cor.z_factor(p=pressure,t=temperature, ppc = cp.ppc, tpc = cp.tpc, method=z_method)

    def get_rhog(self,pressure=14.7,temperature=60, z_method='papay',rhog_method='real_gas',normalize=True):
        _ma = self.apparent_molecular_weight(normalize=normalize)
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
    
    def vapor_pressure(self, t:Temperature=None, temperature:float = None, temperature_unit=None, pressure_unit='psi'):
        
        if t is None:
            t = Temperature(value=temperature, unit=temperature_unit)
        
        for i in self.components:
            _ = i.vapor_pressure(t, pressure_unit=pressure_unit)
            
        return self.df()
    
    def ideal_equilibrium_ratios(self, p:Pressure, t:Temperature, pressure_unit='psi'):
        p = p.convert_to(pressure_unit)
        
        if 'vapor_pressure' not in self.df().columns:
            df = self.vapor_pressure(t = t, pressure_unit=pressure_unit)
        else:
            df = self.df()
            
        #Estimate Equilibrium Ratios. Equation 5-4. Tarek Ahmed Equation of State and Pvt Analysis
        df['k'] = df['vapor_pressure'] / p.value
        
        return df
    
    def ideal_flash_calculations(self, p:Pressure, t:Temperature, pressure_unit='psi', method='newton'):
              
        #Estimate Equilibrium ratios. Assuming Ideal Solutions
        df = self.ideal_equilibrium_ratios(p,t,pressure_unit=pressure_unit)
               
        #Intial Guess. page 336. Tarek Ahmed, Equation of State and Pvt Analysis
        A = np.sum(df['mole_fraction'].values*(df['k'].values-1))
        B = np.sum(df['mole_fraction'].values*((1/df['k'].values)-1))
        
        guess = A/(A+B)
    
        sol = root_scalar(
            cost_flash, 
            args=(df['mole_fraction'].values,df['k'].values),
            x0=guess, 
            method=method,
            fprime = cost_flash_prime            
        )
        
        #nv = total number of moles in the vapor (gas) phase
        nv = sol.root 
        
        #nL = total number of moles in the liquid phase 
        nl = 1- nv 
        
        #yi = mole fraction of component i in the gas phase
        #xi = mole fraction of component i in the liquid phase
        df['xi'] = df['mole_fraction'] / (nl + nv*df['k'])
        df['yi'] = df['xi'] * df['k'] 
        
        
        return df[['mole_fraction','xi','yi','k','vapor_pressure']]
    
    def dew_point(self, t:Temperature, pressure_unit='psi', method='ideal'):
        
        df = self.vapor_pressure(t, pressure_unit='psi')
        
        guess_pd = 1 / (np.sum(df['mole_fraction'].values/df['vapor_pressure'].values))

        if method=='ideal':
            k_func = self.ideal_equilibrium_ratios
        else:
            raise ValueError(f'Method {method} not allowed')
        
        sol = root_scalar(
            cost_dew_point,
            args=(t,df['mole_fraction'].values, k_func, pressure_unit),
            x0=guess_pd,
            method='brentq',
            bracket=[0,10000]
        )
              
        return sol.root
    
    def bubble_point(self, t:Temperature, pressure_unit='psi', method='ideal'):
        
        df = self.vapor_pressure(t, pressure_unit='psi')
        
        guess_pb = np.sum(df['mole_fraction'].values*df['vapor_pressure'].values)

        if method=='ideal':
            k_func = self.ideal_equilibrium_ratios
        else:
            raise ValueError(f'Method {method} not allowed')
        
        sol = root_scalar(
            cost_bubble_point,
            args=(t,df['mole_fraction'].values, k_func, pressure_unit),
            x0=guess_pb,
            method='brentq',
            bracket=[0,10000]
        )
              
        return sol.root
        