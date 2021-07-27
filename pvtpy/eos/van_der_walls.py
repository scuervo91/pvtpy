from pydantic import BaseModel, Field
import numpy as np
from ..units import Pressure, Temperature, CriticalProperties


class VanDerWalls(BaseModel):
    a: float = Field(None)
    b: float = Field(None)
    
    def coef_ab(self,critical_properties:CriticalProperties, R = 10.73):
        pc = critical_properties.critical_pressure.convert_to('psi').value
        tc = critical_properties.critical_temperature.convert_to('rankine').value
        a = 0.421875 * ((np.square(R) * np.square(tc))/pc)
        b = 0.125 * (R * tc) / pc
        
        self.a = a
        self.b = b
        
        return a, b
    
    def coef_AB(self, p:Pressure, t:Temperature, R=10.73):
        pressure = p.convert_to('psi').value
        temperature = t.convert_to('rankine').value
        a = self.a
        b = self.b
        A = (a * pressure) / (np.square(R) * np.square(temperature))
        B = (b * pressure) / (R * temperature)
        
        return A, B
    
    def cubic_poly(self,p:Pressure, t:Temperature, R=10.73):
        A, B = self.coef_AB(p, t, R=R)
        
        coef = [-A*B, A, -(1+B),1]
        return np.polynomial.Polynomial(coef)
    
    def estimate_densities(self, p:Pressure, t:Temperature, molecular_weight:float, R=10.73):
        poly = self.cubic_poly(p, t, R=R)
        
        roots = poly.roots()
        gas_root = roots.max()
        liquid_root = roots.min()
        
        pressure = p.convert_to('psi').value
        temperature = t.convert_to('rankine').value
        
        rho_gas = (pressure*molecular_weight)/(gas_root*R*temperature)
        rho_liquid = (pressure*molecular_weight)/(liquid_root*R*temperature)
        
        return {'rho_gas':rho_gas, 'rho_liquid':rho_liquid}
        
    
    