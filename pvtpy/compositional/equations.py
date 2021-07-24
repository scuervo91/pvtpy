import numpy as np
from ..units import Pressure, Temperature
#Estimate Equilibrium Ratios. Equation 5-4. Tarek Ahmed Equation of State and Pvt Analysis

def cost_flash(x,z,k):
    return np.sum((z*(k - 1))/(x*(k - 1)+1))

# Derivative of cost function
def cost_flash_prime(x,z,k):
    return -np.sum((z*np.square(k - 1))/np.square(x*(k - 1)+1))
###############################################################    