from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from typing import List, Optional
from ..pvt import PVT, TemperatureUnits,PressureUnits, Chromatography
from  ..black_oil import correlations as cor 
#(
#    n2_correction, co2_correction, h2s_correction, pb, rs,
#    bo, rho_oil, co, muod, muo,rsw, bw, cw, muw, rhow, rhog, z_factor, bg, eg, critical_properties,
#    critical_properties_correction, cg, SetGasCorrelations, SetOilCorrelations, SetWaterCorrelations
#)

class InitialConditions(BaseModel):
    pressure: float = Field(...,gt=0)
    temperature: float = Field(...,gt=0)
    pressure_unit: PressureUnits = PressureUnits.psi
    temperature_unit: PressureUnits = TemperatureUnits.farenheit


class FluidBase(BaseModel):
    initial_conditions: InitialConditions = Field(...)
    pvt: PVT = Field(None)
    chromatography: Chromatography = Field(None)

    class Config:
        validate_assignment = True

class Oil(FluidBase):
    api: float = Field(...,gt=0)
    sulphur: float = Field(None, gt=0)
    pb: float = Field(None, gt=0)
    rsb: float = Field(None, gt=0)
    sg_gas: float = Field(None, gt=0)
    
    class Config:
        #extra = 'forbid'
        validate_assignment = True
    
    def pvt_from_correlation(
        self,
        start_pressure=20,
        end_pressure=5000,
        n=20,
        correlations = cor.SetOilCorrelations()
    ):
    
        p_range=np.linspace(start_pressure,end_pressure,n)
        
        if (self.pb is None) & (self.rsb is None):
            raise ValueError('Either Bubble point or Gas Oil Ratio must be defined')
        elif self.pb is None:
            pb = cor.pb(
                rs=self.rsb,
                temp=self.initial_conditions.temperature,
                sg_gas=self.sg_gas,
                api=self.api,
                methods=correlations.pb.value, correction=True)['pb'].values
            self.pb = pb
            
            rs_cor = cor.rs(
                p = p_range,
                pb = pb,
                temp = self.initial_conditions.temperature,
                sg_gas=self.sg_gas,
                api=self.api,
                methods = 'valarde'
            )
        else:
            rs_cor = cor.rs(
                p = p_range,
                pb = self.pb,
                temp = self.initial_conditions.temperature,
                sg_gas=self.sg_gas,
                api=self.api,
                methods = correlations.rs.value
            )
            
        bo_cor = cor.bo(
            p = p_range,
            rs = rs_cor['rs'].values,
            pb = self.pb,
            temp = self.initial_conditions.temperature,
            sg_gas=self.sg_gas,
            api=self.api,
            methods = correlations.bo.value
        )
        
        co_cor = cor.co(
            p = p_range,
            rs = rs_cor['rs'].values,
            pb = self.pb,
            temp = self.initial_conditions.temperature,
            sg_gas=self.sg_gas,
            api=self.api,
            bo=bo_cor['bo'].values,
            method_above_pb = correlations.co_above.value,
            method_below_pb = correlations.co_below.value             
        )
        
        muo_cor = cor.muo(
            p = p_range,
            rs = rs_cor['rs'].values,
            pb = self.pb,
            temp = self.initial_conditions.temperature,
            api=self.api,
            method_above_pb = correlations.muo_above.value,
            method_below_pb = correlations.muo_below.value,
            method_dead = correlations.muod.value
        )
        
        rho_cor = cor.rho_oil(
            p=p_range,
            co=co_cor['co'].values,
            bo=bo_cor['bo'].values,
            rs=rs_cor['rs'].values,
            api=self.api,
            pb=self.pb,
            methods = correlations.muod.value
        )
        
        #_pvt = pd.concat([rs_cor,bo_cor,co_cor,muo_cor,rho_cor],axis=1)
        _pvt = PVT(
            pressure= p_range.tolist(),
            fields={
                'rs':rs_cor['rs'].values.tolist(),
                'bo':bo_cor['bo'].values.tolist(),
                'co':co_cor['co'].values.tolist(),
                'muo':muo_cor['muo'].values.tolist(),
                'rho':rho_cor['rhoo'].values.tolist()
            }
        )
        self.pvt = _pvt
        #print(_pvt.df())
        return _pvt
        
        
        