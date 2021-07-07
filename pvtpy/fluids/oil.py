from pydantic import Field
import numpy as np
import pandas as pd
from ..pvt import PVT
from .base import FluidBase
from ..black_oil import correlations as cor 
from ..units import Pressure
#(
#    n2_correction, co2_correction, h2s_correction, pb, rs,
#    bo, rho_oil, co, muod, muo,rsw, bw, cw, muw, rhow, rhog, z_factor, bg, eg, critical_properties,
#    critical_properties_correction, cg, SetGasCorrelations, SetOilCorrelations, SetWaterCorrelations
#)

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
                temp=self.initial_conditions.temperature.value,
                sg_gas=self.sg_gas,
                api=self.api,
                methods=correlations.pb.value, correction=True)['pb'].values
            self.pb = pb
            
            rs_cor = cor.rs(
                p = p_range,
                pb = pb,
                temp = self.initial_conditions.temperature.value,
                sg_gas=self.sg_gas,
                api=self.api,
                methods = 'valarde'
            )
        else:
            rs_cor = cor.rs(
                p = p_range,
                pb = self.pb,
                temp = self.initial_conditions.temperature.value,
                sg_gas=self.sg_gas,
                api=self.api,
                methods = correlations.rs.value
            )
            
        bo_cor = cor.bo(
            p = p_range,
            rs = rs_cor['rs'].values,
            pb = self.pb,
            temp = self.initial_conditions.temperature.value,
            sg_gas=self.sg_gas,
            api=self.api,
            methods = correlations.bo.value
        )
        
        co_cor = cor.co(
            p = p_range,
            rs = rs_cor['rs'].values,
            pb = self.pb,
            temp = self.initial_conditions.temperature.value,
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
            temp = self.initial_conditions.temperature.value,
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
            pressure= Pressure(value=p_range.tolist()),
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
        
    def to_ecl(
        self,
        pressure=None,
        n_sat = 10,
        n_unsat = 5,
        min_pressure = None,
        max_pressure = None,
        properties = ['rs','bo','muo'],
        float_format = '{:.3f}'.format,
    ):
        if any([self.pvt is None, self.pb is None, self.rsb is None]):
            raise ValueError('PVT, pb and rsb must be defined')

        string = ""
        string += "-- OIL PVT TABLE FOR LIVE OIL\n"
        string += 'PVTO\n'
        string += "-- rs      pres  bo      visc\n"
        string += "-- Mscf/rb psi   RB/STB  cP  \n"
        string += "-- ------- ----  ----    ---- \n"
        
        if pressure  is None:
            if min_pressure is None:
                min_pressure = np.min(self.pvt.pressure.value)
            if max_pressure is None:
                max_pressure = np.max(self.pvt.pressure.value)
            if min_pressure >= self.pb:
                pressure = np.linspace(min_pressure,max_pressure,n_sat)
                flag = 'unsat'
            elif max_pressure <= self.pb:
                pressure = np.linspace(min_pressure,max_pressure,n_sat)
                flag = 'sat'
            else:
                sat_pressure = np.linspace(min_pressure,self.pb,n_sat)
                unsat_pressure = np.linspace(self.pb,max_pressure, n_unsat+1)
                pressure = np.concatenate((sat_pressure,unsat_pressure[1:]))
                flag = 'mixed'
                
        pvt_df = self.pvt.interpolate(pressure, cols=properties).reset_index()

        #convert rs units from scf/bbl to Mscf/bbl
        pvt_df['rs'] = pvt_df['rs'] * 1e-3
        
        if flag == 'unsat':
            string += pvt_df[['rs','pressure','bo','muo']].to_string(header=False, index=False,float_format=float_format) +'\n /\n'
        elif flag == 'sat':
            
            for i,r in pvt_df.iterrows():
                string += pvt_df.loc[[i],['rs','pressure','bo','muo']].to_string(index=False, header=False,float_format=float_format) + '/\n'
                
            string += '/\n'
            
        else:
            
            #Print Saturated data
            for i,r in pvt_df[pvt_df['pressure']<self.pb].iterrows():
                string += pvt_df.loc[[i],['rs','pressure','bo','muo']].to_string(index=False, header=False,float_format=float_format) + '/\n'  
                    
            #Print data at bubble point
            string += pvt_df.loc[pvt_df['pressure']==self.pb,['rs','pressure','bo','muo']].to_string(index=False, header=False,float_format=float_format) + '\n'  
            
            string += '-- Unsaturated Data\n'
            string += pvt_df.loc[pvt_df['pressure']>self.pb,['pressure','bo','muo']].to_string(index=False, header=False,float_format=float_format)
            string += '/\n/'
        
        return string