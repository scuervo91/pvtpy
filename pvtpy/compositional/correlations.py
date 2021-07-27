import numpy as np
from enum import Enum

#Accentric Factor
def acentric_factor(
    critical_pressure=None,
    critical_temperature=None,
    boiling_temperature=None,
    vapor_pressure=None, 
    plus_fraction=False):
    """Accentric Factor calculated by Edmister’s Correlations

    Args:
        pc ([type]): Critical Pressure [psi]
        tc ([type]): Critical Temperature [°R]
        tb ([type]): Normal Boiling Point [°R]
    """
    if plus_fraction:
        upper = 3 * np.log(critical_pressure/14.7)
        lower = 7 * ((critical_temperature/boiling_temperature) - 1)
        
        return (upper/lower)-1 
    else:
        return -(np.log10(vapor_pressure/critical_pressure) - 1)


# Equilibrium rations Correlations

class k_correlations(str,Enum):
    wilson = 'wilson'
    whitson = 'whitson'
    ideal = 'ideal'

def equilibrium(pc=None,tc=None,p=None,t=None,acentric_factor=None, pk=None,pv=None, method='wilson'):
    """Calculate Equilibrium rations Correlations

    Args:
        pc (float): Critical Pressure [psi]
        tc (float): Critical Temperature [°R]
        p (float): System Pressure [psi]
        t (float): System Temperature [°R]
        acentric_factor (float): Accentric Factor [-]
        pk (float): Convergence Pressure (only used on Whitson correlation) [psi]
        pv (float): Vapor Pressure (only used on Ideal solution) [psi]
    Returns:
        [float]: Equibrium rations [-]
    """
    if method == 'wilson':
        return (pc/p)*np.exp(5.37*(1+acentric_factor)*(1-(tc/t)))
    if method == 'whitson':
        if pk is None:
            raise ValueError('pk must be specified for Whitson correlation')
        A = 1 - np.power(p/pk,0.7)
        return np.power(pc/pk,A-1)*(pc/p)*np.exp(5.37*A*(1+acentric_factor)*(1-(tc/t)))
    
    if method == 'ideal':
        return pv/p

def equilibrium_h2s(p, t, pk):
    """equilibrium_h2s Lohrenze, Clark, and Francis (1963)

    Args:
        p (float): System Pressure [psi]
        t (float): System Temperature [°R]
        pk (float): Convergence Pressure [psi]

    Returns:
        [type]: [description]
    """
    a = np.power(1 - (p/pk), 0.8)
    b = 6.3992127 + 1399.2204*np.power(t,-1)
    c = np.log(p)*(0.76885112+18.215052*np.power(t,-1))
    d = 1112446.2*np.power(t,-2)
    
    return a * (b - c - d)

def equilibrium_n2(p,t,pk):
    """equilibrium_n2 Lohrenze, Clark, and Francis (1963)

    Args:
        p (float): System Pressure [psi]
        t (float): System Temperature [°R]
        pk (float): Convergence Pressure [psi]

    Returns:
        [type]: [description]
    """
    a = np.power(1 - (p/pk), 0.6)
    b = 11.294748 - 1184.2409*np.power(t,-1) - 0.90459907*np.log(p)
    
    return a * b

def equilibrium_co2(p,t,pk):
    """equilibrium_n2 Lohrenze, Clark, and Francis (1963)

    Args:
        p (float): System Pressure [psi]
        t (float): System Temperature [°R]
        pk (float): Convergence Pressure [psi]

    Returns:
        [type]: [description]
    """
    a = np.power(1 - (p/pk), 0.6)
    b = 7.0201913 - 152.7291*np.power(t,-1)
    c = np.log(p)*(1.8896974 - 1719.2956*np.power(t,-1) + 644740.69*np.power(t,-2))
    
    return a * (b - c)