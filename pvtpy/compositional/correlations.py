import numpy as np

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

def k_wilson(pc,tc,p,t,omega):
    """Calculate Equilibrium rations from Wilson Correlation

    Args:
        pc (float): Critical Pressure [psi]
        tc (float): Critical Temperature [°R]
        p (float): System Pressure [psi]
        t (float): System Temperature [°R]
        omega (float): Accentric Factor [-]

    Returns:
        [float]: Equibrium rations [-]
    """
    
    return (pc/p)*np.exp(5.37*(1+omega)*(1-(tc/t)))

