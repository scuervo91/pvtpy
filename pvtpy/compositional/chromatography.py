from logging import critical
from pydantic import BaseModel, Field, validator, parse_obj_as
from typing import List, Union
from enum import Enum
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar
#Local imports
from .components import properties_df, Component
from ..black_oil import correlations as cor
from ..units import Pressure, Temperature
from .equations import cost_flash, cost_flash_prime
from .correlations import equilibrium, acentric_factor

class JoinItem(str, Enum):
    id = 'id'
    name = 'name'

class CriticalProperties(BaseModel):
    critical_pressure: float
    critical_temperature: float
    

class Chromatography(BaseModel):
    components: List[Component] = Field(None)
    plus_fraction: Component = Field(None, description='Add component to the chromatography')
    class Config:
        validate_assignment = True
        extra = 'forbid'
        
    def from_df(self, df: pd.DataFrame, name:Union[str,List] = None, mole_fraction:Union[str,List] = None):
        #Assume names are in the index if name is None
        if name is not None:
            if isinstance(name, str):
                df = df.set_index(name)
            else:
                df.index = name
            
        if mole_fraction is None:
            if 'mole_fraction' not in df.columns:
                raise ValueError('Column mole_fraction not defined')
        else:
            df = df.rename(columns={mole_fraction: 'mole_fraction'})
               
        _merged = df.merge(properties_df, how='inner',left_index=True,right_index=True).reset_index().rename(columns={'index':'name'})
        

        self.components = parse_obj_as(List[Component], _merged.to_dict(orient='records'))

    def df(self, pressure_unit='psi',temperature_unit='farenheit',normalize=True, plus_fraction=True):
        df = pd.DataFrame()
        
        for i in self.components:
            df = df.append(i.df(pressure_unit=pressure_unit, temperature_unit=temperature_unit))
        
        if plus_fraction and self.plus_fraction is not None:
            df = df.append(self.plus_fraction.df(pressure_unit=pressure_unit, temperature_unit=temperature_unit))
            
        if normalize and plus_fraction:
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
            _co2 = df.loc['carbon-dioxide', 'mole_fraction'] if 'carbon-dioxide' in df.index.tolist() else 0
            _n2 = df.loc['nitrogen', 'mole_fraction'] if 'nitrogen' in df.index.tolist() else 0
            _h2s = df.loc['hydrogen-sulfide', 'mole_fraction'] if 'hydrogen-sulfide' in df.index.tolist() else 0
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
    
    def vapor_pressure(self, t:Temperature, pressure_unit='psi'):
        vp_df = pd.DataFrame()
        for i in self.components:
            vp = i.vapor_pressure(t, pressure_unit=pressure_unit)
            vp_df = vp_df.append(pd.DataFrame({'vapor_pressure':vp.value,'vapor_pressure_unit':vp.unit.value}, index=[i.name]))

        return vp_df
    
    def acentric_factor(self,pressure_unit='psi'):
        vp_df = pd.DataFrame()
        for i in self.components:
            temp_frac = i.critical_temperature.value*0.7
            vp = i.vapor_pressure(Temperature(value = temp_frac, unit=i.critical_temperature.unit), pressure_unit=pressure_unit)
            vp_df = vp_df.append(
                pd.DataFrame(
                    {
                        'vapor_pressure':vp.value,
                        'vapor_pressure_unit':vp.unit.value,
                        'critical_pressure':i.critical_pressure.convert_to(pressure_unit).value,
                        'temperature': temp_frac,
                        'critical_temperature':i.critical_temperature.convert_to('farenheit').value
                     }, 
                    index=[i.name]
                )
            )

        vp_df['acentric_factor'] = acentric_factor(
            vapor_pressure = vp_df['vapor_pressure'].values, 
            critical_pressure = vp_df['critical_pressure'].values, 
            plus_fraction=False
        )
        
        return vp_df
    
    def convergence_pressure(self, t:Temperature, method='standing'):
        if self.plus_fraction is None:
            raise ValueError('No plus_fraction defined')
        if method == 'standing':
            return 60*self.plus_fraction.molecular_weight - 4200
        if method == 'rzasa':
            a = np.array([6124.3049,-2753.2538,415.42049])
            mw_gamma = self.plus_fraction.molecular_weight * self.plus_fraction.specific_gravity
            c = mw_gamma/(t.convert_to('rankine').value - 460)
            c1 = np.power(c,[1,2,3])
            return -2381.8542 + 46.31487 * mw_gamma + np.dot(a,c1)
                
    def equilibrium_ratios(self, p:Pressure, t:Temperature, pressure_unit='psi', method='wilson', pk_method='rzasa'):
        
        if method=='ideal':
            p = p.convert_to(pressure_unit)
            df = self.vapor_pressure(t = t, pressure_unit=pressure_unit)

            df['k'] = equilibrium(pv=df['vapor_pressure'],p=p.value, method=method)
                           
            return df['k']
        
        if method == 'wilson':
            
            df = self.df(temperature_unit='rankine')
            df['k'] = equilibrium(
                pc = df['critical_pressure'].values,
                tc = df['critical_temperature'].values,
                p = p.convert_to('psi').value,
                t = t.convert_to('rankine').value,
                acentric_factor = df['acentric_factor'].values,
                method = method
            )
            return df['k']
        
        if method == 'whitson':

            df = self.df(temperature_unit='rankine', pressure_unit='psi')
            pk = self.convergence_pressure(t,method=pk_method)    

            df['k'] = equilibrium(
                pc = df['critical_pressure'].values,
                tc = df['critical_temperature'].values,
                p = p.convert_to('psi').value,
                t = t.convert_to('rankine').value,
                acentric_factor = df['acentric_factor'].values,
                method = method,
                pk = pk
            )
            return df['k']        
    
    def phase_moles(self, p:Pressure, t:Temperature, pressure_unit='psi', method='newton', k_method='wilson'):
        #Estimate Equilibrium ratios.
        k = self.equilibrium_ratios(p=p,t=t,pressure_unit=pressure_unit, method=k_method)
        
        df = self.df()
        #Intial Guess. page 336. Tarek Ahmed, Equation of State and Pvt Analysis
        A = np.sum(df['mole_fraction'].values*(k.values-1))
        B = np.sum(df['mole_fraction'].values*((1/k.values)-1))
        
        guess = A/(A+B)
    
        sol = root_scalar(
            cost_flash, 
            args=(df['mole_fraction'].values,k.values),
            x0=guess, 
            method=method,
            fprime = cost_flash_prime            
        )
        
        #nv = total number of moles in the vapor (gas) phase
        nv = sol.root 
        
        #nL = total number of moles in the liquid phase 
        nl = 1- nv 
        
        return {'liquid_moles':nl,'gas_moles':nv}
    
    def flash_calculations(self, p:Pressure, t:Temperature, pressure_unit='psi', method='newton', k_method='wilson'):
                     
        phase_moles = self.phase_moles(p, t, pressure_unit=pressure_unit, method=method, k_method=k_method)
        
        nl = phase_moles['liquid_moles']
        nv = phase_moles['gas_moles']
        
        k = self.equilibrium_ratios(p=p,t=t,pressure_unit=pressure_unit, method=k_method)
        df = self.df()
        #yi = mole fraction of component i in the gas phase
        #xi = mole fraction of component i in the liquid phase
        df['xi'] = df['mole_fraction'] / (nl + nv*k.values)
        df['yi'] = df['xi'] * k.values
        df['k'] = k
        
        return df[['mole_fraction','xi','yi','k']], phase_moles
    
    def dew_point(self, t:Temperature, pressure_unit='psi', k_method='wilson', pk_method='rzasa'):
        
        df = self.df(
            pressure_unit=pressure_unit,
            temperature_unit='rankine',
            normalize=True,
            plus_fraction=True
        )
         
        df['vapor_pressure'] = self.vapor_pressure(t, pressure_unit=pressure_unit)['vapor_pressure']
        
        guess_pd = 1 / (np.sum(df['mole_fraction'].values/df['vapor_pressure'].values))

        def cost_dew_point(p,t,z):
            
            k = self.equilibrium_ratios(
                Pressure(value=p,unit=pressure_unit),
                t,
                pressure_unit=pressure_unit,
                method=k_method,
                pk_method=pk_method
            ).values
            return np.sum(z/k)-1

        sol = root_scalar(
            cost_dew_point,
            args=(t,df['mole_fraction'].values),
            x0=guess_pd,
            method='brentq',
            bracket=[10,10000]
        )
              
        return sol.root

    def bubble_point(self, t:Temperature, pressure_unit='psi', k_method='wilson', pk_method='rzasa',method=None,**kwargs):
        
        df = self.df(
            pressure_unit=pressure_unit,
            temperature_unit='rankine',
            normalize=False,
            plus_fraction=False
        )
         
        df['vapor_pressure'] = self.vapor_pressure(t, pressure_unit=pressure_unit)['vapor_pressure']
        
        guess_pb = np.sum(df['mole_fraction'].values*df['vapor_pressure'].values)
        
        print(f'guesss {guess_pb}')
        def cost_bubble_point(p,t,z):
            print(p)
            k = self.equilibrium_ratios(
                Pressure(value=p,unit=pressure_unit),
                t,
                pressure_unit=pressure_unit,
                method=k_method,
                pk_method=pk_method
            ).values
            
            return np.sum(z*k)-1

        sol = root_scalar(
            cost_bubble_point,
            args=(t,self.df()['mole_fraction'].values),
            x0=guess_pb,
            method=method,
            **kwargs
        )
              
        return sol.root    