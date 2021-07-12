import numpy as np
from enum import Enum
#Accentric Factor

def accentric_factor(pc,tc,tb):
    """Accentric Factor calculated by Edmister’s Correlations

    Args:
        pc ([type]): Critical Pressure [psi]
        tc ([type]): Critical Temperature [°R]
        tb ([type]): Normal Boiling Point [°R]
    """
    
    upper = 3 * np.log(pc/14.7)
    lower = 7 * ((tc/tb) - 1)
    
    return (upper/lower)-1 


# Equilibrium rations Correlations

class k_correlations(str,Enum):
    wilson = 'wilson'
    whitson = 'whitson'

def k_wilson(pc,tc,p,t,omega, pk=None, method='wilson'):
    """Calculate Equilibrium rations Correlations

    Args:
        pc (float): Critical Pressure [psi]
        tc (float): Critical Temperature [°R]
        p (float): System Pressure [psi]
        t (float): System Temperature [°R]
        omega (float): Accentric Factor [-]
        pk (float): Convergence Pressure (only used on Whitson correlation) [psi]

    Returns:
        [float]: Equibrium rations [-]
    """
    if method == 'wilson':
        return (pc/p)*np.exp(5.37*(1+omega)*(1-(tc/t)))
    if method == 'whitson':
        if pk is None:
            raise ValueError('pk must be specified for Whitson correlation')
        A = 1 - np.power(p/pk,0.7)
        return np.power(pc/pk,A-1)*(pc/p)*np.exp(5.37*A*(1+omega)*(1-(tc/t)))

