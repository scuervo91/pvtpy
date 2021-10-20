import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyval
from scipy.interpolate import interp1d
from enum import Enum
from typing import Union, List
from pydantic import BaseModel, Field, validate_arguments
from ..units import Pressure, Temperature, CriticalProperties

#####################################################################################
#####################################################################################
############################ OIL CORRELATIONS #######################################

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def api_to_sg(api:Union[np.ndarray,float,List[float]]):
    api = np.atleast_1d(api)
    sg = 141.5 / (131.5 + api)
    return sg

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def sg_to_api(sg:Union[np.ndarray,float,List[float]]):
    sg = np.atleast_1d(sg)
    api = (141.5 / sg) - 131.5
    return api


#####################################################################################
# Bubble point Correlations

# Correction by Non-Hydrocarbon gases

# Correction by N2
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def n2_correction(
    api:Union[np.ndarray,float,List[float]]=None, 
    temperature:Temperature=None, 
    y:float=0
):
    
    temp = temperature.convert_to('farenheit').value
    if y == 0:
        cn2 = 1
    else:
        cn2 = 1.0 + ((-2.65e-4 * api + 5.5e-3) * temp + (0.0931 * api - 0.8295)) * y + (
                    (1.954e-11 * np.power(api, 4.699) * temp) + (0.027 * api - 2.366)) * np.power(y, 2)
    return cn2

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def co2_correction(y:float=0, temperature:Temperature=None):
    temp = temperature.convert_to('farenheit').value
    if y == 0:
        cco2 = 1
    else:
        cco2 = 1.0 - 693.8 * y * np.power(temp, -1.553)
    return cco2

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def h2s_correction(
    api:Union[np.ndarray,float,List[float]]=None, 
    y:float=0, 
    temperature:Temperature=None
):
    temp = temperature.convert_to('farenheit').value
    if y == 0:
        ch2s = 1
    else:
        ch2s = 1.0 - (0.9035 + 0.0015 * api) * y + 0.019 * (45 - api) * np.power(y, 2)
    return ch2s


class pb_correlations(str,Enum):
    standing = 'standing'
    laster = 'laster'
    vazquez_beggs = 'vazquez_beggs'
    glaso = 'glaso'

# Bubble point
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def pb(
    rs:Union[np.ndarray,float,List[float]]=None, 
    temperature:Temperature=None, 
    sg_gas:Union[np.ndarray,float,List[float]]=None, 
    api:Union[np.ndarray,float,List[float]]=None, 
    method:Union[pb_correlations,List[pb_correlations]]=pb_correlations.standing, 
    correction:bool=True, 
    y_n2:float=0,
    y_co2:float=0,
    y_h2s:float=0,
    **kwargs
):
    """
    Estimate the bubble point pressure using Correlations

    Input: 
        rs -> (int,float,np.array) Solution Gas Ratio [scf/bbl]
        temp -> (int,float,np.array) Temperature [F]
        sg_gas -> (int,float,np.array) Specific Gravity gas (air=1)
        api -> (int,float,np.array) Oil API gravity [API]
        method -> (list, default 'standing')List of correlation methods
                  ['standing',laster,'vazquez_beggs','glaso']
        correction->(bool, default True) Apply correction factors for non-hydrocarbon elements
    
    Return:
        pb -> (pd.DataFrame) Bubble Point Pressure indexed by temperature

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    rs = np.atleast_1d(rs)
    sg_gas = np.atleast_1d(sg_gas)
    api = np.atleast_1d(api)

    assert isinstance(method, (str, list))

    methods = []

    if isinstance(method, pb_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    # Corrections for non Hydrocarbon gases

    cn2 = n2_correction(y=y_n2, api=api, temperature=temperature) if correction == True else 1
    cco2 = co2_correction(y=y_co2, temperature=temperature) if correction == True else 1
    ch2s = h2s_correction(y=y_h2s, api=api, temperature=temperature) if correction == True else 1
    pb_dict = {}

    if 'standing' in methods:
        f = np.power(rs / sg_gas, 0.83) * np.power(10, 0.00091 * temperature.convert_to('farenheit').value - 0.0125 * api)
        pb_standing = 18.2 * (f - 1.4)
        pb = pb_standing * cn2 * cco2 * ch2s
        pb_dict['pb_standing'] = pb

    if 'laster' in methods:
        # estimate oil effective molecular weight
        mo = np.zeros(api.shape)
        mo[api <= 40] = 630 - 10 * api[api <= 40]
        mo[api > 40] = 73110 * np.power(api[api > 40], -1.562)

        # estimate system gas molar fraction
        sg_oil = api_to_sg(api)
        yg = ((rs / 379.3)) / ((rs / 379.3) + (350 * sg_oil / mo))

        pb_factor = np.zeros(yg.shape)
        pb_factor[yg <= 0.6] = 0.679 * np.exp(2.786 * yg[yg <= 0.6]) - 0.323
        pb_factor[yg > 0.6] = 8.26 * np.power(yg[yg > 0.6], 3.56) + 1.95

        temp_r = temperature.convert_to('rankine').value
        pb_laster = pb_factor * temp_r / sg_gas

        pb = pb_laster * cn2 * cco2 * ch2s
        pb_dict['pb_laster'] = pb

    if 'vazquez_beggs' in methods:
        # Constants c1, c2, c3
        c1 = np.zeros(api.shape)
        c2 = np.zeros(api.shape)
        c3 = np.zeros(api.shape)

        c1[api <= 30] = 0.0362
        c1[api > 30] = 0.0178

        c2[api <= 30] = 1.0937
        c2[api > 30] = 1.1870

        c3[api <= 30] = 25.724
        c3[api > 30] = 23.931

        pb_vasquez = np.power(rs / (c1 * sg_gas * np.exp((c3 * api) / temperature.convert_to('rankine').value)), 1 / c2)
        pb = pb_vasquez * cn2 * cco2 * ch2s
        pb_dict['pb_vazquez_beggs'] = pb

    if 'glaso' in methods:
        f = np.power(rs / sg_gas, 0.816) * ((np.power(temperature.convert_to('farenheit').value, 0.172)) / (np.power(api, 0.989)))

        pb_glaso = np.power(10, polyval(np.log10(f), [1.7669, 1.7447, -0.30218]))
        pb = pb_glaso * cn2 * cco2 * ch2s
        pb_dict['pb_glaso'] = pb

    pb_df = pd.DataFrame(pb_dict, index=[temperature.value]) if multiple == True else pd.DataFrame({'pb': pb}, index=[temperature.value])
    pb_df.index.name = 'temp'
    return pb_df


#####################################################################################
# gas-Oil Ratio Correlations

class rs_correlations(str,Enum):
    standing = 'standing'
    laster = 'laster'
    vazquez_beggs = 'vazquez_beggs'
    glaso = 'glaso'
    valarde = 'valarde'
    
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def rs(
    pressure:Pressure=None, 
    pb:Pressure=None, 
    temperature:Temperature=None, 
    api:Union[np.ndarray,float,List[float]]=None, 
    sg_gas:Union[np.ndarray,float,List[float]]=None, 
    rsb:Union[np.ndarray,float,List[float]]=None, 
    method:Union[rs_correlations,List[rs_correlations]]=rs_correlations.standing,
):
    """
    Estimate the Gas-Oil Ratio using Standing Correlation

    Input: 
        p -> (int,float,np.array) Interest Pressure [psi]
        pb -> (int,float,np.array) Bubble Point [psi]
        temp -> (int,float,np.array)Temperature [F]
        api -> (int,float,np.array)Oil API gravity [API]
        sg_gas -> (int,float,np.array) Gas specifi gravity
        rsb -> (int,float,np.array) Gas oil ration at bubble point
        method -> (list, default 'standing')List of correlation methods
                  ['standing',laster,'vazquez_beggs','glaso','valarde']
                  * Valarde method builds rs below pb given rsb
    
    Return:
        rs -> (pd.DataFrame) Gas Oil Ratio indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    pb = np.atleast_1d(pb.convert_to('psi').value)
    api = np.atleast_1d(api)
    sg_gas = np.atleast_1d(sg_gas)

    p = np.array(pressure.convert_to('psi').value)

    methods = []

    if isinstance(method, rs_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    rs_dict = {}
    p_sat = np.zeros(p.shape)
    p_sat[p >= pb] = pb
    p_sat[p < pb] = p[p < pb]

    if 'standing' in methods:
        rs = sg_gas * np.power(((p_sat / 18.2) + 1.4) * np.power(10, 0.0125 * api - 0.00091 * temperature.convert_to('farenheit').value), 1.2048)
        rs_dict['rs_standing'] = rs

    if 'laster' in methods:
        array_shape = np.broadcast_shapes(p_sat.shape, sg_gas.shape , np.array(temperature.value).shape , p.shape)
        mo = np.zeros(api.shape)
        mo[api <= 40] = 630 - 10 * api[api <= 40]
        mo[api > 40] = 73110 * np.power(api[api > 40], -1.562)

        pb_factor = (p_sat * sg_gas) / temperature.convert_to('rankine').value

        # estimate yg

        yg = np.zeros(array_shape)

        yg[pb_factor < 3.29] = 0.359 * np.log(1.473 * pb_factor[pb_factor < 3.29] + 0.476)
        yg[pb_factor >= 3.29] = np.power(0.121 * pb_factor[pb_factor >= 3.29] - 0.236, 0.281)

        sg_oil = api_to_sg(api)
        rs = (132755 * sg_oil * yg) / (mo * (1 - yg))
        rs_dict['rs_laster'] = rs

    if 'vazquez_beggs' in methods:
        # Constants c1, c2, c3
        c1 = np.zeros(api.shape)
        c2 = np.zeros(api.shape)
        c3 = np.zeros(api.shape)

        c1[api <= 30] = 0.0362
        c1[api > 30] = 0.0178

        c2[api <= 30] = 1.0937
        c2[api > 30] = 1.1870

        c3[api <= 30] = 25.724
        c3[api > 30] = 23.931

        rs = c1 * sg_gas * np.power(p_sat, c2) * np.exp((c3 * api) / temperature.convert_to('rankine').value)
        rs_dict['rs_vazquez_begss'] = rs

    if 'glaso' in methods:
        f = np.power(10, 2.8869 - np.power(14.1811 - 3.3093 * np.log10(p_sat), 0.5))
        rs = sg_gas * np.power(f * (np.power(api, 0.989) / np.power(temperature.convert_to('farenheit').value, 0.172)), 1.2255)
        rs_dict['rs_glaso'] = rs

    if 'valarde' in methods:
        """Method for build rs at pressures below pb by giving the rsb
        Correlation of Black Oil Properties at
        Pressures Below Bubble Point Pressure
        -A New Approach"""
        "https://wiki.pengtools.com/index.php?title=Velarde_correlation"
        rsb = np.atleast_1d(rsb)
        temp = temperature.convert_to('farenheit').value
        A0 = 9.73e-7
        A1 = 1.672608
        A2 = 0.929870
        A3 = 0.247235
        A4 = 1.056052
        alpha_1 = A0 * np.power(sg_gas, A1) * np.power(api, A2) * np.power(temp, A3) * np.power(pb, A4)

        B0 = 0.022339
        B1 = -1.00475
        B2 = 0.337711
        B3 = 0.132795
        B4 = 0.302065
        alpha_2 = B0 * np.power(sg_gas, B1) * np.power(api, B2) * np.power(temp, B3) * np.power(pb, B4)

        C0 = 0.725167
        C1 = -1.48548
        C2 = -0.164741
        C3 = -0.09133
        C4 = 0.047094
        alpha_3 = C0 * np.power(sg_gas, C1) * np.power(api, C2) * np.power(temp, C3) * np.power(pb, C4)

        pr = p_sat / pb
        rsr = alpha_1 * np.power(pr, alpha_2) + (1 - alpha_1) * np.power(pr, alpha_3)
        rs = rsr * rsb
        rs_dict['rs_valarde'] = rs

    rs_df = pd.DataFrame(rs_dict, index=p) if multiple == True else pd.DataFrame({'rs': rs}, index=p)
    rs_df.index.name = 'pressure'
    return rs_df


#####################################################################################
# Oil Volumetric Factor

class bo_correlations(str,Enum):
    standing = 'standing'
    vazquez_beggs = 'vazquez_beggs'
    glaso = 'glaso'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bo(
    pressure:Pressure=None, 
    rs:Union[np.ndarray,float,List[float]]=None, 
    pb:Pressure=None, 
    temperature:Temperature=None, 
    api:Union[np.ndarray,float,List[float]]=None, 
    sg_gas:Union[np.ndarray,float,List[float]]=None, 
    co:Union[np.ndarray,float,List[float]]=0.0,
    method:Union[bo_correlations,List[bo_correlations]]=bo_correlations.standing
):
    """
    Estimate the Oil Volumetric Factor using Correlations

    Input: 
        p -> (int,float,np.array) Interest Pressure [psi]
        rs -> (int,float,np.array) Gas Oil Ratio scf/bbl
        pb -> (int,float,np.array) Bubble Point [psi]
        temp ->  (int,float,np.array) Temperature [F]
        api -> (int,float,np.array Oil API gravity [API]
        sg_gas -> Gas specifi gravity
        method -> (list, default 'standing')List of correlation methods
                  ['standing','vazquez_beggs','glaso']
        multiple->(bool, default False) Allow to return multiple result from multiple correlation
                  If true the 'method' must be a length equal to 1
    
    Return:
        bo -> (pd.DataFrame) Oil Volumetric Factor indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    rs = np.atleast_1d(rs)
    pb = np.atleast_1d(pb.convert_to('psi').value)
    api = np.atleast_1d(api)
    sg_gas = np.atleast_1d(sg_gas)
    co = np.atleast_1d(co)

    assert isinstance(method, (str, list))

    methods = []
    if isinstance(method, bo_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    bo_dict = {}

    if 'standing' in methods:
        sg_oil = api_to_sg(api)
        f = rs * np.sqrt(sg_gas / sg_oil) + 1.25 * temperature.convert_to('farenheit').value
        bo = 0.9759 + 12e-5 * np.power(f, 1.2)
        bo[p>pb] = bo[p>pb] * np.exp(co[p>pb] * (pb - p[p>pb]))
        bo_dict['bo_standing'] = bo

    if 'vazquez_beggs' in methods:
        # Constants c1, c2, c3
        c1 = np.zeros(api.shape)
        c2 = np.zeros(api.shape)
        c3 = np.zeros(api.shape)

        c1[api <= 30] = 4.677e-4
        c1[api > 30] = 4.670e-4

        c2[api <= 30] = 1.751e-5
        c2[api > 30] = 1.1e-5

        c3[api <= 30] = -1.8106e-8
        c3[api > 30] = 1.3370e-9
        temp = temperature.convert_to('farenheit').value
        bo = 1 + c1 * rs + c2 * (temp - 60) * (api / sg_gas) + c3 * rs * (temp - 60) * (api / sg_gas)
        bo[p>pb] = bo[p>pb] * np.exp(co[p>pb] * (pb - p[p>pb]))
        bo_dict['bo_vazquez_beggs'] = bo

    if 'glaso' in methods:
        temp = temperature.convert_to('farenheit').value
        sg_oil = api_to_sg(api)
        f = rs * np.power(sg_gas / sg_oil, 0.526) + 0.968 * temp
        bo = 1 + np.power(10, -6.58511 + 2.91329 * np.log10(f) - 0.27683 * np.power(np.log10(f), 2))
        bo[p>pb] = bo[p>pb] * np.exp(co[p>pb] * (pb - p[p>pb]))
        bo_dict['bo_glaso'] = bo

    bo_df = pd.DataFrame(bo_dict, index=p) if multiple == True else pd.DataFrame({'bo': bo}, index=p)
    bo_df.index.name = 'pressure'
    return bo_df


#####################################################################################
# Oil Compressibility
class co_above_correlations(str,Enum):
    vazquez_beggs = 'vazquez_beggs'
    petrosky = 'petrosky'
    kartoatmodjo = 'kartoatmodjo'

class co_below_correlations(str,Enum):
    mccain = 'mccain'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def co(
    pressure:Pressure=None, 
    rs:Union[np.ndarray,float,List[float]]=None, 
    pb:Pressure=None, 
    temperature:Temperature=None, 
    sg_gas:Union[np.ndarray,float,List[float]]=None, 
    api:Union[np.ndarray,float,List[float]]=None, 
    method_above_pb:co_above_correlations=co_above_correlations.vazquez_beggs, 
    method_below_pb:co_below_correlations=co_below_correlations.mccain
):
    """
    Estimate the Oil compresibility in 1/psi

    Input: 
        p -> (int,float,list,np.array) Interest Pressure [psi]
        rs -> (int,float,np.array) Gas Oil Ratio scf/bbl
        pb -> (int,float,np.array) Bubble Point [psi]
        temp ->  (int,float,np.array) Temperature [F]
        sg_gas -> (int,float,np.array) Gas specifi gravity
        api -> (int,float,np.array) Oil API gravity [API]
        bo -> (list,np.array) Oil Volumetric factor
        bg -> (list,np.array) Gas Volumetric factor
        method_above_pb -> (list, default 'vazquez_beggs') method to use above the bubble point
                            ['vazquez_beggs','petrosky','kartoatmodjo']
        method_below_pb -> (list, default 'mccain') method to use below the bubble point
                            ['mccain']
    Return:
        rho -> (pd.DataFrame) Oil Density indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    rs = np.atleast_1d(rs)
    pb = np.atleast_1d(pb.convert_to('psi').value)
    sg_gas = np.atleast_1d(sg_gas)
    api = np.atleast_1d(api)

    #assert isinstance(pb, (int, float, list, np.ndarray))
    #bg = np.atleast_1d(bg)

    co = np.zeros(p.shape)

    temp = temperature.convert_to('farenheit').value
    if 'vazquez_beggs' == method_above_pb:
        co[p >= pb] = (-1433 + 5 * rs[p >= pb] + 17.2 * temp - 1180 * sg_gas + 12.61 * api) / (
                    p[p >= pb] * np.power(10, 5))

    elif 'petrosky' == method_above_pb:
        co[p >= pb] = 1.705e-7 * np.power(rs[p >= pb], 0.69357) * np.power(sg_gas, 0.1885) * np.power(api,0.3272) * np.power(temp, 0.6729) * np.power(p[p >= pb], -0.5906)

    elif 'kartoatmodjo' == method_above_pb:
        co[p >= pb] = (6.8257 * np.power(rs[p >= pb], 0.5002) * np.power(api, 0.3613) * np.power(temp, 0.76606) *
                       np.power(sg_gas, 0.35505)) / (p[p >= pb] * np.power(10, 6))
    else:
        raise ValueError('no method set')

    if 'mccain' == method_below_pb:
        rsb = rs[p >= pb].mean()
        co[p < pb] = 5.1414768e-4 * np.power(p[p < pb], -1.450) * np.power(pb, -0.383) * np.power(temp,1.402) * np.power(api,0.256) * np.power(rsb, 0.449)

    else:
        raise ValueError('no method set')

    co_df = pd.DataFrame({'co': co}, index=p)
    co_df.index.name = 'pressure'
    return co_df

class muod_correlations(str,Enum):
    beal = 'beal'
    beggs = 'beggs'
    glaso = 'glaso'


#####################################################################################
# Dead Oil Viscosity
@validate_arguments(config=dict(arbitrary_types_allowed=True))
def muod(
    temperature:Temperature=None, 
    api:Union[np.ndarray,float,List[float]]=None, 
    method:Union[muod_correlations,List[muod_correlations]]=muod_correlations.beal
):
    """
    Estimate the Dead Oil Viscosity

    Input: 
        temp -> (int,float,list,np.array) Reservoir Temperature F
        api -> (int,float,list,np.array) API crude API
        methods-> (list,default 'beal') List of methods
                ['beal','beggs','glaso']
    Return:
        muod ->(pd.Dataframe) Dead Oil Viscosity [cP] indexed by temperatures

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    api = np.atleast_1d(api)

    methods = []
    if isinstance(method, muod_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    muod_dict = {}

    if 'beal' in methods:
        temp = np.atleast_1d(temperature.convert_to('farenheit').value)
        a = np.power(10, 0.43 + (8.33 / api))
        muod = (0.32 + (1.8e7 / np.power(api, 4.53))) * np.power(360 / (temp + 200), a)
        muod_dict['muod_beal'] = muod

    if 'beggs' in methods:
        temp = np.atleast_1d(temperature.convert_to('farenheit').value)
        z = 3.0324 - 0.02023 * api
        y = np.power(10, z)
        x = y * np.power(temp, -1.163)

        muod = np.power(10, x) - 1
        muod_dict['muod_beggs'] = muod

    if 'glaso' in methods:
        temp = np.atleast_1d(temperature.convert_to('farenheit').value)
        muod = 3.141e10 * np.power(temp, -3.444) * np.power(np.log10(api), 10.313 * np.log10(temp) - 36.447)
        muod_dict['muod_glaso'] = muod

    muod_df = pd.DataFrame(muod_dict, index=temp) if multiple == True else pd.DataFrame({'muod': muod}, index=temp)
    muod_df.index.name = 'temp'
    return muod_df


#####################################################################################
# Live Oil Viscosity

class muo_below_correlations(str,Enum):
    chew = 'chew'
    beggs = 'beggs'
    kartoatmodjo = 'kartoatmodjo'
    
class muo_above_correlations(str,Enum):
    beal = 'beal'
    vazquez_beggs = 'vazquez_beggs'
    kartoatmodjo = 'kartoatmodjo'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def muo(
    pressure:Pressure=None, 
    rs:Union[np.ndarray,float,List[float]]=None, 
    pb:Pressure=None, 
    rsb:Union[np.ndarray,float,List[float]]=None,
    temperature:Temperature=None, 
    api:Union[np.ndarray,float,List[float]]=None,
    method_below_pb:muo_below_correlations='beggs', 
    method_above_pb:muo_above_correlations='vazquez_beggs', 
    method_dead:muod_correlations='beal'
):
    """
    Estimate the live Oil Viscosity

    Input: 
        p -> (int,float,list,np.array) interest Pressure [psi]
        rs -> (int,float,list,np.array)Gas Oil Ratio scf/bbl
        pb -> (int,float,list,np.array) Bubble Point psi
        method_below_pb -> (list, default 'beggs') method to use above the bubble point
                            ['chew','beggs','kartoatmodjo']
        method_above_pb -> (list, default 'vazquez_beggs') method to use below the bubble point
                            ['beal','vazquez_beggs','kartoatmodjo']
        method_dead -> (list, default 'beal') method estimate dead oil
                            ['beal','beggs','glaso']


    Return:
        mu -> (pd.DataFrame) Oil Viscosity [cP] indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    rs = np.atleast_1d(rs)
    rsb = np.atleast_1d(rsb)
    pb = np.atleast_1d(pb.convert_to('psi').value)
    api = np.atleast_1d(api)

    # Estimate the Dead oil Viscosity
    _muod = muod(temperature=temperature, api=api, method=method_dead)
    _muod = _muod['muod'].values

    muo = np.zeros(p.shape)
    muob = np.zeros(1)

    if 'chew' == method_below_pb:
        a = np.power(10, rs[p <= pb] * (2.2e-7 * rs[p <= pb] - 7.4e-4))
        b = (0.68 / np.power(10, 8.62e-5 * rs[p <= pb])) + (0.25 / np.power(10, 1.1e-3 * rs[p <= pb])) + (
                    0.062 / np.power(10, 3.74e-3 * rs[p <= pb]))
        muo[p <= pb] = a * np.power(_muod, b)

        a_b = np.power(10, rsb * (2.2e-7 * rsb - 7.4e-4))
        b_b = (0.68 / np.power(10, 8.62e-5 * rsb)) + (0.25 / np.power(10, 1.1e-3 * rsb)) + (
                    0.062 / np.power(10, 3.74e-3 * rsb))
        muob = a_b * np.power(_muod, b_b)

    if 'beggs' == method_below_pb:
        a = 10.715 * np.power(rs[p <= pb] + 100, -0.515)
        b = 5.44 * np.power(rs[p <= pb] + 150, -0.338)
        muo[p <= pb] = a * np.power(_muod, b)

        a_b = 10.715 * np.power(rsb + 100, -0.515)
        b_b = 5.44 * np.power(rsb + 150, -0.338)
        muob = a_b * np.power(_muod, b_b)

    if 'kartoatmodjo' == method_below_pb:
        b = np.power(10, -0.00081 * rs[p <= pb])
        a = (0.2001 + 0.8428 * np.power(10, -0.000845 * rs[p <= pb])) * np.power(_muod, 0.43 + 0.5165 * b)
        muo[p <= pb] = -0.06821 + 0.9824 * a + 40.34e-5 * np.power(a, 2)

        b_b = np.power(10, -0.00081 * rsb)
        a_b = (0.2001 + 0.8428 * np.power(10, -0.000845 * rsb)) * np.power(_muod, 0.43 + 0.5165 * b_b)
        muob = -0.06821 + 0.9824 * a_b + 40.34e-5 * np.power(a_b, 2)

    if 'beal' == method_above_pb:
        muo[p > pb] = (0.001 * (p[p > pb] - pb)) * (0.024 * np.power(muob, 1.6) + 0.038 * np.power(muob, 0.56)) + muob

    if 'vazquez_beggs' == method_above_pb:
        m = 2.6 * np.power(p[p > pb], 1.187) * np.exp(-11.513 - 8.98e-5 * p[p > pb])
        muo[p > pb] = muob * np.power(p[p > pb] / pb, m)

    if 'kartoatmodjo' == method_above_pb:
        muo[p > pb] = 1.00081 * muob + 1.127e-3 * (p[p > pb] - pb) * (
                    -65.17e-4 * np.power(muob, 1.8148) + 0.038 * np.power(muob, 1.59))

    muo_df = pd.DataFrame({'muo': muo}, index=p)
    muo_df.index.name = 'pressure'
    return muo_df


#####################################################################################
# Oil density

class rho_correlations(str,Enum):
    banzer = 'banzer'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def rho_oil(
    pressure:Pressure=None, 
    co:Union[np.ndarray,float,List[float]]=None, 
    bo:Union[np.ndarray,float,List[float]]=None, 
    bob:Union[np.ndarray,float,List[float]]=None,
    rs:Union[np.ndarray,float,List[float]]=None, 
    rsb:Union[np.ndarray,float,List[float]]=None,
    api:Union[np.ndarray,float,List[float]]=None, 
    pb:Pressure=None, 
    method:Union[rho_correlations,List[rho_correlations]]=rho_correlations.banzer
):
    """
    Estimate the Oil Density in lb/ft3

    Input: 
        p ->  (int,float,list,np.array) Interest Pressure [psi]
        co -> (int,float,list,np.array) Isotermic oil compressibility 1/psi
        bo -> (int,float,list,np.array) Oil Volumetric Factor
        rs -> (int,float,list,np.array) Gas Oil Ratio scf/bbl
        api -> (int,float,list,np.array) Oil API gravity [API]
        pb -> (int,float,list,np.array)Bubble Point [psi]
        method -> (list, default 'banzer') Correlation
          
    Return:
        rho -> (pd.DataFrame) Oil Density indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    co = np.atleast_1d(co)
    bo = np.atleast_1d(bo)
    rs = np.atleast_1d(rs)
    api = np.atleast_1d(api)
    pb = np.atleast_1d(pb.convert_to('psi').value)
    rsb = np.atleast_1d(rsb)

    methods = []
    if isinstance(method, rho_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    rho_oil_dict = {}

    if 'banzer' in methods:
        # Gas disolved specific gravity
        ygd = ((12.5 + api) / 50) - 3.5715e-6 * api * rs

        rho_oil = np.zeros(p.shape)
        p_sat = np.zeros(p.shape)
        p_sat[p >= pb] = pb
        p_sat[p < pb] = p[p < pb]

        sg_oil = api_to_sg(api)
        rho_oil[p <= pb] = (350 * sg_oil + 0.0764 * ygd[p <= pb] * rs[p <= pb]) / (5.615 * bo[p <= pb])

        rho_ob = (350 * sg_oil + 0.0764 * ygd[p > pb] * rsb) / (5.615 * bob)
        rho_oil[p > pb] = rho_ob * np.exp(co[p > pb] * (pb - p[p > pb]))
        rho_oil_dict['rho_banzer'] = rho_oil

    rho_df = pd.DataFrame(rho_oil_dict, index=p) if multiple == True else pd.DataFrame({'rhoo': rho_oil}, index=p)
    rho_df.index.name = 'pressure'
    return rho_df


#####################################################################################
#####################################################################################
############################ WATER CORRELATIONS #####################################

class rsw_correlations(str,Enum):
    culberson = 'culberson'
    mccoy = 'mccoy'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def rsw(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    salinity:Union[np.ndarray,float,List[float]]=None, 
    method:Union[rsw_correlations,List[rsw_correlations]]=rsw_correlations.culberson
):
    """
    Estimate Water Gas solubility is 

    Input: 
        p ->  (int,float,list,np.array) Interest Pressure [psi]
        t ->  (int,float,list,np.array) Interest Temperature [F]

        method -> (str,list, default 'culberson') Correlation
          
    Return:
        rsw -> (pd.DataFrame) water solubility indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('farenheit').value)

    assert isinstance(method, (str, list))

    methods = []
    if isinstance(method, rsw_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    rsw_dict = {}

    if 'culberson' in methods:
        a = 8.15839 - 6.12265e-2 * t + 1.91663e-4 * np.power(t, 2) - 2.1654e-7 * np.power(t, 3)
        b = 1.01021e-2 - 7.44241e-5 * t + 3.05553e-7 * np.power(t, 2) - 2.94883e-10 * np.power(t, 3)
        c = (-9.02505 + 0.130237 * t - 8.53425e-4 * np.power(t, 2) + 2.34122e-6 * np.power(t,
                                                                                           3) - 2.37049e-9 * np.power(t,
                                                                                                                      4)) * 1e-7
        rswp = a + b * p + c * np.power(p, 2)

        # Convert p in ppm to percentage %
        per_s = salinity / 1e4
        correction = np.power(10, -0.0840655 * per_s * np.power(t, -0.285854))
        rsw = rswp * correction
        rsw_dict['rws_culberson'] = rsw

    if 'mccoy' in methods:
        a = 2.12 + 3.45e-3 * t - 3.59e-5 * np.power(t, 2)
        b = 0.0107 - 5.26e-5 * t + 1.48e-7 * np.power(t, 2)
        c = -8.75e-7 + 3.9e-9 * t - 1.02e-11 * np.power(t, 2)
        rswp = a + b * p + c * np.power(p, 2)
        # Convert p in ppm to percentage %
        per_s = salinity / 1e4
        correction = (1 - (0.0753 - 1.73e-4 * t) * per_s)
        rsw = rswp * correction
        rsw_dict['rsw_mccoy'] = rsw

    rsw_df = pd.DataFrame(rsw_dict, index=p) if multiple == True else pd.DataFrame({'rsw': rsw}, index=p)
    rsw_df.index.name = 'pressure'
    return rsw_df

class bw_correlations(str,Enum):
    mccain = 'mccain'
    mccoy = 'mccoy'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bw(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    pb:Pressure=Pressure(value=14.7, unit='psi'), 
    cw:Union[np.ndarray,float,List[float]]=0, 
    salinity:Union[np.ndarray,float,List[float]]=None, 
    method:Union[bw_correlations,List[bw_correlations]]=bw_correlations.mccain
):
    """
    Estimate Water volumetric factor

    Input: 
        p ->  (int,float,list,np.array) Interest Pressure [psi]
        t ->  (int,float,list,np.array) Interest Temperature [F]
        pb ->  (int,float,list,np.array) Bubble point [psi]
        cw ->  (int,float,list,np.array) Water isothermal compressibility [1/psi]
        s ->  (int,float,list,np.array) Salinity [ppm]

        method -> (str,list, default 'mccain') Correlation
          
    Return:
        bw -> (pd.DataFrame) water volumetric factor indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('farenheit').value)
    pb = np.atleast_1d(pb.convert_to('psi').value)
    cw = np.atleast_1d(cw)
    salinity = np.atleast_1d(salinity)

    assert isinstance(method, (str, list))

    methods = []
    if isinstance(method, bw_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True
        
    bw_dict = {}

    if 'mccain' in methods:
        bwp = np.zeros(p.shape)
        delta_vw_t = np.zeros(p.shape)
        delta_vw_p = np.zeros(p.shape)
        correction = np.zeros(p.shape)

        delta_vw_t[p < pb] = -1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * np.power(t, 2)
        delta_vw_t_pb = -1.0001e-2 + 1.33391e-4 * t + 5.50654e-7 * np.power(t, 2)
        delta_vw_p[p < pb] = -1.95301e-9 * p[p <= pb] * t - 1.72834e-13 * np.power(p[p <= pb], 2) * t - 3.58922e-7 * p[p <= pb] - 2.25341e-10 * np.power(p[p <= pb], 2)
        delta_vw_p_pb = -1.95301e-9 * pb * t - 1.72834e-13 * np.power(pb,2) * t - 3.58922e-7 * pb - 2.25341e-10 * np.power(pb, 2)

        bwp[p <= pb] = (1 + delta_vw_p[p < pb]) * (1 + delta_vw_t[p < pb])
        bwb = (1 + delta_vw_p_pb) * (1 + delta_vw_t_pb)

        bwp[p > pb] = bwb * np.exp(cw[p > pb] * (pb - p[p > pb]))

        # Convert p in ppm to percentage %
        per_s = salinity / 1e4
        correction = 1 + per_s * (
                    5.1e-8 * p + (5.47e-6 - 1.95e-10 * p) * (t - 60) - (3.23e-8 - 8.5e-13 * p) * (np.power(t - 60, 2)))
        bw = bwp * correction
        bw_dict['bw_mccain'] = bw

    if 'mccoy' in methods:
        bwp = np.zeros(p.shape)
        correction = np.zeros(p.shape)

        a = 0.9911 + 6.35e-5 * t + 8.5e-7 * np.power(t, 2)
        b = -1.093e-6 - 3.497e-9 * t + 4.57e-12 * np.power(t, 2)
        c = -5e-11 + 6.429e-13 * t - 1.43e-15 * np.power(t, 2)

        bwp[p <= pb] = a + b * p[p <= pb] + c * np.power(p[p <= pb], 2)
        bwb = a + b * pb + c * np.power(pb, 2)

        bwp[p > pb] = bwb = np.exp(cw[p > pb] * (pb - p[p > pb]))

        # Convert p in ppm to percentage %
        per_s = salinity / 1e4
        correction = 1 + per_s * (
                    5.1e-8 * p + (5.47e-6 - 1.95e-10 * p) * (t - 60) - (3.23e-8 - 8.5e-13 * p) * (np.power(t - 60, 2)))
        bw = bwp * correction
        bw_dict['bw_mccoy'] = bw

    bw_df = pd.DataFrame(bw_dict, index=p) if multiple == True else pd.DataFrame({'bw': bw}, index=p)
    bw_df.index.name = 'pressure'
    return bw_df

class cw_correlations(str,Enum):
    standing = 'standing'
    osif = 'osif'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def cw(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    rsw:Union[np.ndarray,float,List[float]]=0, 
    salinity:Union[np.ndarray,float,List[float]]=0, 
    method:Union[cw_correlations,List[cw_correlations]]=cw_correlations.standing
):  # Note: Pending develop cw pressure < pb
    """
    Estimate Water compressibility

    Input: 
        p ->  (int,float,list,np.array) Interest Pressure [psi]
        t ->  (int,float,list,np.array) Interest Temperature [F]
        rsw ->  (int,float,list,np.array) water solubility 


        method -> (str,list, default 'standing') Correlation
          
    Return:
        bw -> (pd.DataFrame) water volumetric factor indexed by pressure

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('farenheit').value)
    rsw = np.atleast_1d(rsw)
    salinity = np.atleast_1d(salinity)

    assert isinstance(method, (str, list))

    methods = []
    if isinstance(method, cw_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    cw_dict = {}

    if 'standing' in methods:
        a = 3.8546 - 1.34e-4 * p
        b = -0.01052 + 4.77e-7 * p
        c = 3.9267e-5 - 8.8e-10 * p

        cwp = (a + b*t + c * np.power(t, 2)) / 1e6

        correction_rsw = 1 + 8.9e-3 * rsw

        # Convert p in ppm to percentage %
        per_s = salinity / 1e4
        correction_s = 1 + np.power(per_s, 0.7) * (
                    -5.2e-2 + 2.7e-4 * t - 1.14e-6 * np.power(t, 2) + 1.121e-9 * np.power(t, 3))

        cw = cwp * correction_rsw * correction_s

        cw_dict['cw_standing'] = cw

    if 'osif' in methods:
        per_s = salinity / 1e4
        cw = 1 / (7.033 * p + 541.5 * per_s - 537 * t + 403300)
        cw_dict['cw_osif'] = cw

    cw_df = pd.DataFrame(cw_dict, index=p) if multiple == True else pd.DataFrame({'cw': cw}, index=p)
    cw_df.index.name = 'pressure'
    return cw_df

class muw_correlations(str,Enum):
    van_wingen = 'van_wingen'
    russel = 'russel'
    meehan = 'meehan'
    brill_beggs = 'brill_beggs'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def muw(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    salinity:Union[np.ndarray,float,List[float]] = 0,  
    method:Union[muw_correlations,List[muw_correlations]]=muw_correlations.van_wingen
):
    """
    Estimate Water Viscosity

    Input:
        p ->  (int,float,list,np.array) Interest Pressure [psi]
        t ->  (int,float,list,np.array) Interest Temperature [F]

        method -> (str,list, default 'standing') Correlation

    Return:
        muw -> (pd.DataFrame) water viscosity  in cP

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('farenheit').value)
    salinity = np.atleast_1d(salinity)

    methods = []
    if isinstance(method, muw_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    muw_dict = {}

    if 'van_wingen' in methods:
        muw = np.exp(1.003 - 1.479e-2*t + 1.982e-5*np.power(t,2))
        muw_dict['muw_van_wingen'] = muw

    if 'russel' in methods:
        per_s = salinity / 1e4
        a = -0.04518 + 0.009313*per_s - 0.000393*np.power(per_s,2)
        b = 70.634 + 0.09576*np.power(per_s,2)
        f = 1 + 3.5e-12 * np.power(p,2)* (t-40)
        muw_po = a + (b/t)
        muw = muw_po*f
        muw_dict['muw_russel'] = muw

    if 'meehan' in methods:
        per_s = salinity / 1e4
        d = 1.12166 - 0.0263951*per_s + 6.79461e-4*np.power(per_s,2) + 5.47119e-5*np.power(per_s,3) - 1.55586e-6*np.power(per_s,4)
        muwt = (109.574 - 8.40564*per_s + 0.313314*np.power(per_s,2) + 8.72213e-3*np.power(per_s,3))*np.power(t,-d)
        muw = muwt*(0.9994 + 4.0295e-5*p + 3.1062e-9*np.power(p,2))
        muw_dict['muw_meehan'] = muw

    if 'brill_beggs' in methods:
        muw = np.exp(1.003 - 1.479e-2*t + 1.982e-5*np.power(t,2))
        muw_dict['muw_brill_beggs'] = muw

    muw_df = pd.DataFrame(muw_dict, index=p) if multiple == True else pd.DataFrame({'muw': muw}, index=p)
    muw_df.index.name = 'pressure'
    return muw_df

class rhow_correlations(str,Enum):
    banzer = 'banzer'
    mccain = 'mccain'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def rhow(
    pressure:Pressure=None,
    salinity:Union[np.ndarray,float,List[float]] = 0, 
    bw:Union[np.ndarray,float,List[float]] = 1, 
    method:Union[rhow_correlations,List[rhow_correlations]]=rhow_correlations.banzer
):
    """
    Estimate Water Density in lb/ft3

    Input:
        s ->  (int,float,list,np.array) dissolved solids [ppm]
        bw ->  (int,float,list,np.array) Water Volumetric Factor []
        method -> (str,list, default 'banzer') Correlation

    Return:
        rhow -> (pd.DataFrame) water density [lb/ft3]

    Source: Correlaciones Numericas PVT - Carlos Banzer
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    salinity = np.atleast_1d(salinity)
    bw = np.atleast_1d(bw)

    methods = []
    if isinstance(method, rhow_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    rhow_dict = {}

    if 'banzer' in methods:
        ge_w = 1 + 0.695e-6*salinity
        rhow = 62.4 * ge_w/bw
        rhow_dict['rhow_banzer'] = rhow

    if 'mccain' in methods:
        if len(salinity)==1:
            s_array = np.full(p.shape, salinity)
        per_s = salinity/1e4
        rhow = 62.368 + 0.438603*per_s + 1.60074e-3*np.power(per_s,2)
        rhow_dict['rhow_mccain'] = rhow

    rhow_df = pd.DataFrame(rhow_dict, index=p) if multiple == True else pd.DataFrame({'rhow': rhow}, index=p)
    rhow_df.index.name = 'pressure'
    return rhow_df

#####################################################################################
#####################################################################################
############################ GAS CORRELATIONS #######################################

class rhog_correlations(str,Enum):
    ideal_gas = 'ideal_gas'
    real_gas = 'real_gas'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def rhog(
    pressure:Pressure=None, 
    ma:Union[np.ndarray,float,List[float]]=None, 
    z:Union[np.ndarray,float,List[float]]=1, 
    r:float=10.73, 
    temperature: Temperature=None, 
    method:Union[rhog_correlations,List[rhog_correlations]]=rhog_correlations.ideal_gas
):
    """
    Estimate Gas density 

    Input:
        p ->  (int,float,list,np.array) Pressure [psi]
        t ->  (int,float,list,np.array) Temperature [F]
        ma ->  (int,float,list,np.array) Apparent molecular weight
        r ->  (int,float,list,np.array) Constant 
        z ->  (int,float,list,np.array) compressibility Factor
        method -> (str,list, default 'ideal_gas') Correlation

    Return:
        rhog -> (pd.DataFrame) gas density [lb/ft3]

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """

    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('rankine').value)# temp to R
    z = np.atleast_1d(z)
    ma = np.atleast_1d(ma)
    r = np.atleast_1d(r)

    assert isinstance(method, (str, list))

    methods = []
    if isinstance(method, rhog_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    rhog_dict = {}

    if 'real_gas' in methods:
        rhog = (p*ma)/(z*r*t)
        rhog_dict['real_gas'] = rhog

    if 'ideal_gas' in methods:
        rhog = (p*ma)/(r*t)
        rhog_dict['ideal_gas'] = rhog

    rhog_df = pd.DataFrame(rhog_dict, index=p) if multiple == True else pd.DataFrame({'rhog': rhog}, index=p)
    rhog_df.index.name = 'pressure'
    return rhog_df

class z_correlations(str,Enum):
    papay = 'papay'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def z_factor(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    critical_properties:CriticalProperties=None,
    method:Union[z_correlations,List[z_correlations]]=z_correlations.papay
):
    """
    Estimate Gas compressibility Factor 

    Input:
        p ->  (int,float,list,np.array) Pressure [psi]
        t ->  (int,float,list,np.array) Temperature [F]
        ppc ->  (int,float,list,np.array) pressure pseudo critical[F]
        tpc ->  (int,float,list,np.array) temperature pseudo critical[F]
        method -> (str,list, default 'papay') Correlation

    Return:
        z -> (pd.DataFrame) Compressibility Factor

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('rankine').value) # temp to R

    ppc = np.atleast_1d(critical_properties.critical_pressure.convert_to('psi').value) # temp to R
    tpc = np.atleast_1d(critical_properties.critical_temperature.convert_to('rankine').value) # temp to R

    methods = []
    if isinstance(method, z_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    z_dict = {}

    #Estimate Pseudo-reduced Properties
    ppr = p/ppc
    tpr = t/tpc

    if 'papay' in methods:
        z = 1 - ((3.52*ppr)/(np.power(10,0.9813*tpr))) + ((0.274*np.power(ppr,2))/(np.power(10,0.8157*tpr)))
        z_dict['z_papay'] = z

    z_df = pd.DataFrame(z_dict, index=p) if multiple == True else pd.DataFrame({'z': z}, index=p)
    z_df.index.name = 'pressure'
    return z_df    

class bg_units(str,Enum):
    ft3scf = 'ft3scf'
    bblscf = 'bblscf'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def bg(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    z:Union[np.ndarray,float,List[float]]=1, 
    unit:Union[bg_units,List[bg_units]]=bg_units.ft3scf
):
    """
    Estimate Gas Volumetric factor 

    Input:
        p ->  (int,float,list,np.array) Pressure [psi]
        t ->  (int,float,list,np.array) Temperature [F]
        z ->  (int,float,list,np.array) Compressibility factor
        units -> (str, default 'ft3/scf') Correlation

    Return:
        bg -> (pd.DataFrame) Gas Volumetric factor

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """  
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('rankine').value)
    z = np.atleast_1d(z)


    units = []
    if isinstance(unit, bg_units):
        units.append(unit.value)
        multiple = False
    else:
        units.extend([i.value for i in unit])
        multiple = True

    bg_dict = {}

    if 'ft3scf' in units:
        bg = 0.02827*z*t/p
        bg_dict['bg_ft3/scf'] = bg 
    
    if 'bblscf' in units: 
        bg = 0.00503*z*t/p
        bg_dict['bg_bbl/scf'] = bg  

    bg_df = pd.DataFrame(bg_dict, index=p) if multiple == True else pd.DataFrame({'bg': bg}, index=p)
    bg_df.index.name = 'pressure'
    return bg_df 

class eg_units(str,Enum):
    scfft3 = 'scfft3'
    scfbbl = 'scfbbl'

@validate_arguments(config=dict(arbitrary_types_allowed=True))
def eg(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    z:Union[np.ndarray,float,List[float]]=1, 
    unit:Union[eg_units,List[eg_units]]=eg_units.scfft3
):
    """
    Estimate Gas Volumetric expansion factor

    Input:
        p ->  (int,float,list,np.array) Pressure [psi]
        t ->  (int,float,list,np.array) Temperature [F]
        z ->  (int,float,list,np.array) Compressibility factor
        units -> (str, default 'scf/ft3') Correlation

    Return:
        eg -> (pd.DataFrame) Gas Volumetric factor

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """  
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('rankine').value)
    z = np.atleast_1d(z)


    units = []
    if isinstance(unit, eg_units):
        units.append(unit.value)
        multiple = False
    else:
        units.extend([i.value for i in unit])
        multiple = True

    eg_dict = {}

    if 'scfft3' in units:
        eg = 35.37*p/(z*t)
        eg_dict['eg_scf/ft3'] = eg
    
    if 'scfbbl' in units: 
        eg = 198.6*p/(z*t)
        eg_dict['eg_scf bbl'] = eg  

    eg_df = pd.DataFrame(eg_dict, index=p) if multiple == True else pd.DataFrame({'eg': eg}, index=p)
    eg_df.index.name = 'pressure'
    return eg_df 

class cp_correlations(str,Enum):
    standing = 'standing'
    
class GasType(str,Enum):
    natural_gas = 'natural_gas'
    condensate_gas = 'condensate_gas'

@validate_arguments(config=dict(arbitrary_types_allowed=True)) 
def critical_properties(
    sg:Union[np.ndarray,float,List[float]]=None, 
    gas_type:GasType=GasType.natural_gas,
    method:cp_correlations=cp_correlations.standing
):
    """
    Estimate Gas Critial Properties from Specific Gravity of gas. Brown Correlation

    Input:
        sg ->  (int,float,list,np.array) Gas Specific Gravity
        gas -> (str, default 'natural_gas') Type of gas. Options: 'natural_gas', 'condensate_gas'
        method -> (str, default 'standing') Correlation

    Return:
        critical_properties -> (dict) Dictionary with keys 'ppc' and 'tpc'

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """    
    sg = np.atleast_1d(sg)

    methods = []
    if isinstance(method, cp_correlations):
        methods.append(method.value)
    else:
        methods.extend([i.value for i in method])

    if 'standing' in methods:
        if gas_type == 'natural_gas':
            _ppc = 677.0 + 15.0*sg - 37.5*np.power(sg,2)
            _tpc = 168.0 + 325.0*sg - 12.5*np.power(sg,2) 
        elif gas_type == 'condensate_gas':
            _ppc = 706.0 + 51.7*sg - 11.1*np.power(sg,2)
            _tpc = 187.0 + 330.0*sg - 71.5*np.power(sg,2)

        ppc = Pressure(value=_ppc, unit='psi')
        tcp = Temperature(value=_tpc, unit='rankine')

        cp = CriticalProperties(
            critical_pressure=ppc,
            critical_temperature=tcp
        )
    
    return cp

class cp_correction_correlations(str,Enum):
    wichert_aziz = 'wichert_aziz'
    carr_kobayashi_burrows = 'carr_kobayashi_burrows'

@validate_arguments(config=dict(arbitrary_types_allowed=True)) 
def critical_properties_correction(
    critical_properties:CriticalProperties=None,
    h2s:float=0,
    co2:float=0, 
    n2:float=0, 
    method:cp_correction_correlations=cp_correction_correlations.wichert_aziz
):
    """
    Correct the critical properties estimations by Non-hydrocarbon components

    Input:
        ppc ->  (int,float,list,np.array) Pressure pseudo critical
        tpc -> (str) Temperature pseudo critical
        h2s -> (int,float,list,np.array) H2S mole fraction
        co2 -> (int,float,list,np.array) co2 mole fraction
        n2 -> (int,float,list,np.array) n2 mole fraction
        method -> (str, default 'wichert_aziz') correlation. Options: 'wichert_aziz', 'carr_kobayashi_burrows'

    Return:
        critical_properties -> (dict) Dictionary with keys 'ppc' and 'tpc'

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """  
    ppc = np.atleast_1d(critical_properties.critical_pressure.convert_to('psi').value)
    tpc = np.atleast_1d(critical_properties.critical_temperature.convert_to('rankine').value)

    if method =='wichert_aziz':
        a = h2s + co2
        b = h2s
        e = 120*(np.power(a,0.9)-np.power(a,1.6)) + 15*(np.power(b,0.5) - np.power(b,4))

        tpc_c = tpc - e
        ppc_c = (ppc*tpc_c)/(tpc + b*(1-b)*e)

    elif method == 'carr_kobayashi_burrows':
        tpc_c = tpc - 80*co2 + 130*h2s - 250*n2
        ppc_c = ppc + 440*co2 + 600*h2s - 170*n2
    
    else:
        raise ValueError('No method matched')
    
    cp = CriticalProperties(
        critical_pressure = Pressure(value = ppc_c, unit = 'psi'),
        critical_temperature = Temperature(value = tpc_c, unit = 'rankine').convert_to('farenheit')
    )

    return cp

class mug_correlations(str,Enum):
    lee_gonzalez = 'lee_gonzalez'

@validate_arguments(config=dict(arbitrary_types_allowed=True)) 
def mug(
    pressure:Pressure=None, 
    temperature:Temperature=None, 
    rhog:Union[np.ndarray,float,List[float]]=None, 
    ma:Union[np.ndarray,float,List[float]]=None, 
    method:mug_correlations=mug_correlations.lee_gonzalez
):
    """
    Estimate gas viscosity

    Input:
        p ->  (int,float,list,np.array) Pressure [psi]
        t -> (int,float,list,np.array) temperature [F]
        rhog -> (int,float,list,np.array) density [lb/ft3]
        ma -> (int,float,list,np.array) Molecular Weight
        method -> (str, default 'lee_gonzalez') correlation. Options: 'lee_gonzalez'

    Return:
        critical_properties -> (dict) Dictionary with keys 'ppc' and 'tpc'

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """ 
    p = np.atleast_1d(pressure.convert_to('psi').value)
    t = np.atleast_1d(temperature.convert_to('rankine').value)
    rhog = np.atleast_1d(rhog)
    ma = np.atleast_1d(ma)

    if method == 'lee_gonzalez':
        k = ((9.4 + 0.02*ma)*np.power(t,1.5))/(209 + 19*ma + t)
        x = 3.5 + (986/t) + 0.01*ma
        y = 2.4 - 0.2*x
        mug = 1e-4 * k * np.exp(x*np.power(rhog/62.4,y))

    mug_df = pd.DataFrame({'mug': mug}, index=p)
    mug_df.index.name = 'pressure'
    return mug_df

class cg_correlations(str,Enum):
    ideal_gas = 'ideal_gas'

# TODO: Implement cg_correlations real gas
@validate_arguments(config=dict(arbitrary_types_allowed=True)) 
def cg(
    pressure:Pressure=None, 
    z:Union[np.ndarray,float,List[float]]=1, 
    method:Union[cg_correlations,List[cg_correlations]]=cg_correlations.ideal_gas
):
    """
    Estimate gas compressibility

    Input:
        p ->  (int,float,list,np.array) Pressure [psi]
        method -> (str, default 'ideal_gas') correlation. Options: 'ideal_gas', 'real_gas'

    Return:
        cg -> (pd.DataFrame) dataframe with cg indexed by pressure

    Source: Reservoir Engineer handbook -  Tarek Ahmed
    """ 
    p = np.atleast_1d(pressure.convert_to('psi').value)
    z = np.atleast_1d(z)  

    methods = []
    if isinstance(method, cg_correlations):
        methods.append(method.value)
        multiple = False
    else:
        methods.extend([i.value for i in method])
        multiple = True

    cg_dict = {}

    if 'ideal_gas' in methods:
        cg = 1/p
        cg_dict['cg_ideal_gas'] = cg

    cg_df = pd.DataFrame(cg_dict, index=p) if multiple == True else pd.DataFrame({'cg': cg}, index=p)
    cg_df.index.name = 'pressure'
    return cg_df 


class SetOilCorrelations(BaseModel):
    pb: pb_correlations = pb_correlations.standing
    rs: rs_correlations = rs_correlations.standing
    bo: bo_correlations = bo_correlations.standing
    co_above: co_above_correlations = co_above_correlations.vazquez_beggs
    co_below: co_below_correlations = co_below_correlations.mccain
    muod: muod_correlations = muod_correlations.beal
    muo_above: muo_above_correlations = muo_above_correlations.beal
    muo_below: muo_below_correlations = muo_below_correlations.beggs
    rho: rho_correlations = rho_correlations.banzer
    
class SetGasCorrelations(BaseModel):
    critical_properties: cp_correlations = cp_correlations.standing
    critical_properties_correction: cp_correction_correlations = cp_correction_correlations.wichert_aziz
    correct_critical_properties: bool = Field(True)
    z: z_correlations = z_correlations.papay
    rhog: rhog_correlations = rhog_correlations.real_gas
    bg: bg_units = bg_units.bblscf
    mug: mug_correlations = mug_correlations.lee_gonzalez
    cg: cg_correlations = cg_correlations.ideal_gas
    
class SetWaterCorrelations(BaseModel):
    rsw: rsw_correlations = rsw_correlations.culberson
    cw: cw_correlations = cw_correlations.standing
    bw: bw_correlations = bw_correlations.mccain
    rhow: rhow_correlations = rhow_correlations.banzer
    muw: muw_correlations = muw_correlations.van_wingen
