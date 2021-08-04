from pydantic import BaseModel, Field
import numpy as np
from ..units import Pressure, Temperature, CriticalProperties


class SoaveRedlichKwong(BaseModel):
    a: float = Field(None)
    b: float = Field(None)
    alpha: float = Field(None)
    a_alpha: float = Field(None)
    
    def coef_ab(self,critical_properties:CriticalProperties, R = 10.73):
        pc = critical_properties.critical_pressure.convert_to('psi').value
        tc = critical_properties.critical_temperature.convert_to('rankine').value
        a = 0.42747 * ((np.square(R) * np.square(tc))/pc)
        b = 0.08664 * (R * tc) / pc
        
        self.a = a
        self.b = b
        
        return a, b
    
    def coef_m(self, acentric_factor:float):
        return 0.480 + 1.574*acentric_factor - 0.176*np.power(acentric_factor,2)
    
    def coef_alpha(self,t:Temperature,critical_properties:CriticalProperties,acentric_factor:float):
        tc = critical_properties.critical_temperature.convert_to('rankine').value
        t = t.convert_to('rankine').value
        
        #Reduced Temperature
        tr = t/tc
        
        #m coef
        m = self.coef_m(acentric_factor)
        
        #Alpha coef
        alpha = np.square(1 + m*(1-np.sqrt(tr)))
        
        self.alpha = alpha
        
        return alpha

    def coef_AB(self, p:Pressure, t:Temperature, R=10.73):
        pressure = p.convert_to('psi').value
        temperature = t.convert_to('rankine').value
        a = self.a
        b = self.b
        alpha= self.alpha
        a_alpha = self.a * self.alpha if self.a_alpha is None else self.a_alpha
        A = (a_alpha * pressure) / (np.square(R) * np.square(temperature))
        B = (b * pressure) / (R * temperature)
        
        return A, B

    def cubic_poly(self,p:Pressure, t:Temperature, R=10.73):
        A, B = self.coef_AB(p, t, R=R)
        
        coef = [-A*B, A-B-np.square(B), -1,1]
        return np.polynomial.Polynomial(coef)
    
    def mixture_coef_ab(self, mole_fraction, a, b, alpha, k=None):
        # Redlich-Kwong coefficients from Hydrocarbon mixtures
        xx = np.matmul(mole_fraction.reshape(-1,1), mole_fraction.reshape(1,-1))
        aa = np.matmul(a.reshape(-1,1), a.reshape(1,-1))
        hh = np.matmul(alpha.reshape(-1,1), alpha.reshape(1,-1))

        if k is None:
            k = 0.
        elif k.shape != xx.shape:
            raise ValueError(f'k must be a scalar or have the same shape as xx {xx.shape}')
            
        product = xx * np.sqrt(aa * hh)
        a_alpha = product.sum().sum()
        bm = np.dot(mole_fraction,b)
              
        self.b = bm
        self.a_alpha = a_alpha
        
        return a_alpha, bm
    
    
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
        
        positive_roots = roots[roots > 0]
        gas_root = positive_roots.max()
        liquid_root = positive_roots.min()
        
        rho_gas = (pressure*molecular_weight)/(gas_root*R*temperature)
        rho_liquid = (pressure*molecular_weight)/(liquid_root*R*temperature)
        
        return {'rho_gas':rho_gas, 'rho_liquid':rho_liquid}