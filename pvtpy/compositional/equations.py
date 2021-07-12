import numpy as np
from ..units import Pressure, Temperature
#Estimate Equilibrium Ratios. Equation 5-4. Tarek Ahmed Equation of State and Pvt Analysis

def cost_flash(x,z,k):
    return np.sum((z*(k - 1))/(x*(k - 1)+1))

# Derivative of cost function
def cost_flash_prime(x,z,k):
    return -np.sum((z*np.square(k - 1))/np.square(x*(k - 1)+1))
###############################################################


#Estimate dew point. Equation 5-32. Tarek Ahmed Equation of State and Pvt Analysis
def cost_dew_point(p,t,z,func_k,pressure_unit):
    df = func_k(Pressure(value=p, unit=pressure_unit),t, pressure_unit=pressure_unit)
    k = df['k'].values
    
    return np.sum(z/k)-1

#Estimate bubble point. Equation 5-32. Tarek Ahmed Equation of State and Pvt Analysis
def cost_bubble_point(p,t,z,func_k,pressure_unit):
    df = func_k(Pressure(value=p, unit=pressure_unit),t, pressure_unit=pressure_unit)
    k = df['k'].values
    
    return np.sum(z*k)-1
    
    