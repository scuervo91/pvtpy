from pydantic import BaseModel, Field
from typing import List
#Local Imports
from .chromatography import Chromatography
from .components import Component
from ..units import Pressure, Temperature


class PhaseMoles(BaseModel):
    liquid_moles: float = Field(..., ge=0, description="The moles of the liquid phase.")
    gas_moles: float = Field(..., ge=0, description="The moles of the gas phase.")
    
class Stage(BaseModel):
    pressure: Pressure
    temperature: Temperature
    chromatography: Chromatography = Field(None)
    phase_moles: PhaseMoles = Field(None)
    
    class Config:
        validate_assignment = True
        extra = 'ignore'
        arbitrary_types_allowed = True
        

class SeparatorTest(BaseModel):
    initial_chromatography: Chromatography
    stages: List[Stage]
    
    class Config:
        validate_assignment = True
        extra = 'ignore'
        arbitrary_types_allowed = True
        
    def solve(self, **kwargs):
        plus_fraction = False if self.initial_chromatography.plus_fraction is None else True
        
        for i,stage in enumerate(self.stages):
            print(f"Stage {i}")
            p_stage = stage.pressure
            t_stage = stage.temperature
            
            if i == 0:
                flash_df, phase_moles = self.initial_chromatography.flash_calculations(p_stage,t_stage, **kwargs)
            else:
                flash_df, phase_moles = self.stages[i-1].chromatography.flash_calculations(p_stage,t_stage, **kwargs)
            
            chr_ = Chromatography()
            chr_.from_df(flash_df, mole_fraction='xi')
            
            if plus_fraction:
                plus_component = self.initial_chromatography.plus_fraction.copy()
                plus_component.mole_fraction = flash_df['mole_fraction'].iloc[-1]
                chr_.plus_fraction = plus_component
            
            stage.phase_moles = phase_moles
            stage.chromatography = chr_
            
        return True
        
        
    
    
    
