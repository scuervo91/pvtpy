from pydantic import BaseModel, Field
from typing import List, Optional
from ..pvt import PVT, TemperatureUnits,PressureUnits
from ..black_oil.correlations import n2_correction, co2_correction, h2s_correction, pb, rs, \
    bo, rho_oil, co, muod, muo,rsw, bw, cw, muw, rhow, rhog, z_factor, bg, eg, critical_properties,\
        critical_properties_correction, cg

class InitialConditions(BaseModel):
    pressure: float = Field(...,gt=0)
    temperature: float = Field(...,gt=0)
    pressure_unit: PressureUnits = PressureUnits.psi
    temperature_unit: PressureUnits = TemperatureUnits.farenheit


class Oil(BaseModel):
    api: float = Field(...,gt=0)
    sulphur: float = Field(None, gt=0)
    bubble_pressure: float = Field(None, gt=0)
    rsb: float = Field(None, gt=0)
    initial_conditions: InitialConditions = Field(...)
    pvt: PVT = Field(None)

