import pandas as pd
from pydantic import BaseModel, constr
import os 

#local imports
from ..units import Pressure, Temperature

#upload table property list
file_dir = os.path.dirname(__file__)
components_path = os.path.join(file_dir,'components_properties1.csv')
properties_df = pd.read_csv(components_path)


class Antoine(BaseModel):
    a: float
    b: float
    c: float
    
    #def vapor_pressure(self, T: Temperature) -> Pressure:
        
    


class Component(BaseModel):
    name: str
    formula: str
    iupac_key: constr(regex = r'^([0-9A-Z\-]+)$')
    cas: constr(regex = r'\b[1-9]{1}[0-9]{1,6}-\d{2}-\d\b')
    molecular_weight: float
    critical_pressure: Pressure
    critical_temperature: Temperature
    
