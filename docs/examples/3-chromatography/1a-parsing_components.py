from typing import List 
import pvtpy.compositional as comp
from pydantic import parse_obj_as


comps_dict = comp.properties_df.head(15).to_dict(orient='records')

#print(comps_dict)

first = comp.Component(**comps_dict[0])

#print(first)

items = parse_obj_as(List[comp.Component], comps_dict)

for i in items:
    print(f'Component {i.name} mw {i.molecular_weight}')


# comps = parse_obj_as(
#     List[comp.Component],
#     comps_dict
# )

