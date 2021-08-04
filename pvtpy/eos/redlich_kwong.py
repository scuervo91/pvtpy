from pydantic import BaseModel, Field
import numpy as np
from ..units import Pressure, Temperature, CriticalProperties


class RedlichKwong(BaseModel):
    a: float = Field(None)
    b: float = Field(None)
    
    def coef_ab(self,critical_properties:CriticalProperties, R = 10.73):
        pc = critical_properties.critical_pressure.convert_to('psi').value
        tc = critical_properties.critical_temperature.convert_to('rankine').value
        a = 0.42747 * ((np.square(R) * np.power(tc,2.5))/pc)
        b = 0.08664 * (R * tc) / pc
        
        self.a = a
        self.b = b
        
        return a, b
    
    def mixture_coef_ab(self, mole_fraction, a, b):
        # Redlich-Kwong coefficients from Hydrocarbon mixtures
        am = np.square(np.dot(mole_fraction, np.sqrt(a)))
        bm = np.dot(mole_fraction, b)
        
        self.a = am
        self.b = bm
        
        return am, bm
    
    def coef_AB(self, p:Pressure, t:Temperature, R=10.73):
        pressure = p.convert_to('psi').value
        temperature = t.convert_to('rankine').value
        a = self.a
        b = self.b
        A = (a * pressure) / (np.square(R) * np.power(temperature,2.5))
        B = (b * pressure) / (R * temperature)
        
        return A, B
    
    def cubic_poly(self,p:Pressure, t:Temperature, R=10.73):
        A, B = self.coef_AB(p, t, R=R)
        
        coef = [-A*B, A-B-np.square(B), -1,1]
        return np.polynomial.Polynomial(coef)
    
    def estimate_densities(self, p:Pressure, t:Temperature, molecular_weight:float, R=10.73):
        poly = self.cubic_poly(p, t, R=R)
        
        pressure = p.convert_to('psi').value
        temperature = t.convert_to('rankine').value
        
        roots = poly.roots()
        real_roots = np.isreal(roots)
        
        if real_roots.sum() == 1:
            root_z = roots[real_roots].real
            rho = (pressure*molecular_weight)/(root_z*R*temperature)
            
            return {'rho':rho}
                    
        gas_root = roots.max()
        liquid_root = roots.min()
        
        rho_gas = (pressure*molecular_weight)/(gas_root*R*temperature)
        rho_liquid = (pressure*molecular_weight)/(liquid_root*R*temperature)
        
        return {'rho_gas':rho_gas, 'rho_liquid':rho_liquid}
        