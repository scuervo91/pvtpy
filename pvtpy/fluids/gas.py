from pydantic import Field 
import numpy as np
from enum import Enum
import pandas as pd
from ..pvt import PVT
#Local imports
from .base import FluidBase
from  ..black_oil import correlations as cor

class GasType(str,Enum):
    natural_gas = 'natural_gas'
    condensate = 'condensate'


class Gas(FluidBase):
    sg: float = Field(None, gt=0)
    gas_type: GasType = Field(GasType.natural_gas)

    def pvt_from_correlations(
        self,
        start_pressure=20,
        end_pressure=5000,
        n=20, 
        correlations = cor.SetGasCorrelations(),
        normalize = True
    ):
        
        p_range=np.linspace(start_pressure,end_pressure,n)
        
        # Define Pseudo critical properties
        if self.chromatography is not None:
            _cp = self.chromatography.get_pseudo_critical_properties(
                correct = correlations.correct_critical_properties,
                correct_method = correlations.critical_properties_correction,
                normalize = normalize
            )
        elif self.sg is not None:
            _cp = cor.critical_properties(
                sg = self.sg, 
                gas_type = self.gas_type.value, 
                method = correlations.critical_properties
            )
        else:
            raise ValueError('Neither chromatography nor sg gas been set')
        
        # Compressibility factor z
        z_cor = cor.z_factor(
            p=p_range, 
            t=self.initial_conditions.temperature, 
            ppc=_cp.ppc, 
            tpc=_cp.tpc, 
            method=correlations.z)
        
        # Density 
        if self.chromatography is not None:
            _ma = self.chromatography.mwa(normalize=normalize)
        else:
            _ma = self.sg * 28.96
        # Density     
        rhog_cor = cor.rhog(
            p=p_range, 
            ma=_ma, 
            z=z_cor['z'].values, 
            r=10.73, 
            t=self.initial_conditions.temperature, 
            method=correlations.rhog
        )
        #Gas volumetric factor
        bg_cor = cor.bg(
            p=p_range,
            t=self.initial_conditions.temperature, 
            z=z_cor['z'].values, 
            method=correlations.bg
        )
        #Gas viscosity
        mug_cor = cor.mug(
            p=p_range,
            t=self.initial_conditions.temperature, 
            rhog=rhog_cor['rhog'].values, 
            ma=_ma, 
            method=correlations.mug
        )
        
        #Gas compressibility 
        cg_cor = cor.cg(
            p=p_range, 
            z=z_cor['z'].values, 
            method=correlations.cg
        )
        
        _pvt = pd.concat([z_cor,rhog_cor,bg_cor,mug_cor,cg_cor],axis=1)
        
        _pvt = PVT(
            pressure= p_range.tolist(),
            fields={
                'z':z_cor['z'].values.tolist(),
                'rhog':rhog_cor['bo'].values.tolist(),
                'bg':bg_cor['bg'].values.tolist(),
                'mug':mug_cor['mug'].values.tolist(),
                'cg':cg_cor['cg'].values.tolist()
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
        properties = ['bg','mug'],
        float_format = '{:.3f}'.format
    ):
    
        if self.pvt is None:
            raise ValueError('No pvt data has been set')
        
        string = ""
        string += "-- GAS PVT TABLE FOR LIVE OIL\n"
        string += 'PVDG\n'
        string += "-- pres   bg       vic  \n"
        string += "-- psi    Rb/Mscf  cP  \n"
        string += "-- ----   ----     ---- \n"
        
        if pressure is None:
            if min_pressure is None:
                min_pressure = np.min(self.pvt.pressure)
            
            if max_pressure is None:
                max_pressure = np.max(self.pvt.pressure)
                
            pressure = np.linspace(min_pressure,max_pressure,n)
            
        pvt_df = self.pvt.interpolate(pressure, cols=properties).reset_index()
        
        #convert bo units from rb/scf to rb/Mscf
        pvt_df['bg'] = pvt_df['bg'] * 1e3
        
        # Write the string
        string += pvt_df[['pressure','bg','mug']].to_string(header=False, index=False,float_format=float_format) +'/\n'
                   
        return string 
        
        
      

        
   
        
        
            