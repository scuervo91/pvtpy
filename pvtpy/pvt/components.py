import pandas as pd
import os 

#upload table property list
file_dir = os.path.dirname(__file__)
components_path = os.path.join(file_dir,'components_properties.csv')
properties_df = pd.read_csv(components_path)
