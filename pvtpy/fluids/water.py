from pydantic import Field
import numpy as np
import pandas as pd
from ..pvt import PVT
from .base import FluidBase
from ..black_oil import correlations as cor 
from ..units import Pressure


class Water(FluidBase):
    salinity: float = Field(..., gt=0)
    pb: float = Field(None,gt=0)
    
    class Config:
        extra = 'ignore'
        validate_assignment = True
        
    def pvt_from_correlation(
        self,
        start_pressure=20,
        end_pressure=5000,
        n=20,
        correlations = cor.SetWaterCorrelations()
    ):

        p_range=np.linspace(start_pressure,end_pressure,n)


        rsw_cor = cor.rsw(p=p_range, t=self.initial_conditions.temperature.value, s=self.salinity, method=correlations.rsw)
        cw_cor = cor.cw(p=p_range, t=self.initial_conditions.temperature.value, rsw=rsw_cor['rsw'].values, s=self.salinity, method=correlations.cw)
        bw_cor = cor.bw(p=p_range, t=self.initial_conditions.temperature.value, pb=self.pb, cw=cw_cor['cw'].values, s=self.salinity, method=correlations.bw)
        rhow_cor = cor.rhow(p=p_range,s=self.salinity, bw=bw_cor['bw'].values, method = correlations.rhow)
        muw_cor = cor.muw(p=p_range, t=self.initial_conditions.temperature.value, s = self.salinity,  method = correlations.muw)

        _pvt = pd.concat([rsw_cor,cw_cor,bw_cor,muw_cor,rhow_cor],axis=1)

        _pvt = PVT(
            pressure= Pressure(value=p_range.tolist()),
            fields={
                'rs':rsw_cor['rsw'].values.tolist(),
                'cw':cw_cor['cw'].values.tolist(),
                'bw':bw_cor['bw'].values.tolist(),
                'muw':muw_cor['muw'].values.tolist(),
                'rhow':rhow_cor['rhow'].values.tolist()
            }
        )

        self.pvt = _pvt
        #print(_pvt.df())
        return _pvt
    
    def to_ecl(
        self,
        pressure=None,
        n = 10,
        min_pressure = None,
        max_pressure = None,
        properties = ['bw','cw','muw'],
        float_format = '{:.3f}'.format
    ):
    
        if self.pvt is None:
            raise ValueError('No pvt data has been set')
        
        string = ""
        string += "-- WATER PVT TABLE\n"
        string += 'PVTW\n'
        string += "-- pres   bw       cw     visc  visc  \n"
        string += "-- psi    RB/STB   1/PSIA cP    GRAD \n"
        string += "-- ----   ----     ---- \n"
        
        if pressure is None:
            if min_pressure is None:
                min_pressure = np.min(self.pvt.pressure)
            
            if max_pressure is None:
                max_pressure = np.max(self.pvt.pressure)
                
            pressure = np.linspace(min_pressure,max_pressure,n)
            
        pvt_df = self.pvt.interpolate(pressure, cols=properties).reset_index()
        
        
        # Write the string
        string += pvt_df[['pressure','bg','mug']].to_string(header=False, index=False,float_format=float_format) +'/\n'
                   
        return string 
