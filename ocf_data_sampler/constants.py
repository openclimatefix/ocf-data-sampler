"""Constants for the package."""

import numpy as np
import xarray as xr
from typing_extensions import override

NWP_PROVIDERS = [
    "ukv",
    "ecmwf",
    "gfs",
    "icon_eu",
]


def _to_data_array(d: dict) -> xr.DataArray:
    """Convert a dictionary to a DataArray."""
    return xr.DataArray(
        [d[k] for k in d],
        coords={"channel": list(d.keys())},
    ).astype(np.float32)


class NWPStatDict(dict):
    """Custom dictionary class to hold NWP normalization stats."""

    @override
    def __getitem__(self, key: str) -> xr.DataArray:
        if key not in NWP_PROVIDERS:
            raise KeyError(f"{key} is not a supported NWP provider - {NWP_PROVIDERS}")
        elif key in self.keys():
            return super().__getitem__(key)
        else:
            raise KeyError(
                f"Values for {key} not yet available in ocf-data-sampler {list(self.keys())}",
            )


# ------ UKV
# Means and std computed WITH version_7 and higher, MetOffice values
UKV_STD = {
    "cdcb": 2126.99350113,
    "lcc": 39.33210726,
    "mcc": 41.91144559,
    "hcc": 38.07184418,
    "sde": 0.1029753,
    "hcct": 18382.63958991,
    "dswrf": 190.47216887,
    "dlwrf": 39.45988077,
    "h": 1075.77812282,
    "t": 4.38818501,
    "r": 11.45012499,
    "dpt": 4.57250482,
    "vis": 21578.97975625,
    "si10": 3.94718813,
    "wdir10": 94.08407495,
    "prmsl": 1252.71790539,
    "prate": 0.00021497,
}

UKV_MEAN = {
    "cdcb": 1412.26599062,
    "lcc": 50.08362643,
    "mcc": 40.88984494,
    "hcc": 29.11949682,
    "sde": 0.00289545,
    "hcct": -18345.97478167,
    "dswrf": 111.28265039,
    "dlwrf": 325.03130139,
    "h": 2096.51991356,
    "t": 283.64913206,
    "r": 81.79229501,
    "dpt": 280.54379901,
    "vis": 32262.03285118,
    "si10": 6.88348448,
    "wdir10": 199.41891636,
    "prmsl": 101321.61574029,
    "prate": 3.45793433e-05,
}

UKV_STD = _to_data_array(UKV_STD)
UKV_MEAN = _to_data_array(UKV_MEAN)

# ------ ECMWF
# These were calculated from 100 random init times of UK data from 2020-2023
ECMWF_STD = {
    "dlwrf": 15855867.0,
    "dswrf": 13025427.0,
    "duvrs": 1445635.25,
    "hcc": 0.42244860529899597,
    "lcc": 0.3791404366493225,
    "mcc": 0.38039860129356384,
    "prate": 9.81039775069803e-05,
    "sd": 0.000913831521756947,
    "sr": 16294988.0,
    "t2m": 3.692270040512085,
    "tcc": 0.37487083673477173,
    "u10": 5.531515598297119,
    "u100": 7.2320556640625,
    "u200": 8.049470901489258,
    "v10": 5.411230564117432,
    "v100": 6.944501876831055,
    "v200": 7.561611652374268,
    "diff_dlwrf": 131942.03125,
    "diff_dswrf": 715366.3125,
    "diff_duvrs": 81605.25,
    "diff_sr": 818950.6875,
}

ECMWF_MEAN = {
    "dlwrf": 27187026.0,
    "dswrf": 11458988.0,
    "duvrs": 1305651.25,
    "hcc": 0.3961029052734375,
    "lcc": 0.44901806116104126,
    "mcc": 0.3288780450820923,
    "prate": 3.108070450252853e-05,
    "sd": 8.107526082312688e-05,
    "sr": 12905302.0,
    "t2m": 283.48333740234375,
    "tcc": 0.7049227356910706,
    "u10": 1.7677178382873535,
    "u100": 2.393547296524048,
    "u200": 2.7963004112243652,
    "v10": 0.985887885093689,
    "v100": 1.4244288206100464,
    "v200": 1.6010299921035767,
    "diff_dlwrf": 1136464.0,
    "diff_dswrf": 420584.6875,
    "diff_duvrs": 48265.4765625,
    "diff_sr": 469169.5,
}

ECMWF_STD = _to_data_array(ECMWF_STD)
ECMWF_MEAN = _to_data_array(ECMWF_MEAN)

# ------ GFS
GFS_STD = {
    "dlwrf": 96.305916,
    "dswrf": 246.18533,
    "hcc": 42.525383,
    "lcc": 44.3732,
    "mcc": 43.150745,
    "prate": 0.00010159573,
    "r": 25.440672,
    "sde": 0.43345627,
    "t": 22.825893,
    "tcc": 41.030598,
    "u10": 5.470838,
    "u100": 6.8899174,
    "v10": 4.7401133,
    "v100": 6.076132,
    "vis": 8294.022,
    "u": 10.614556,
    "v": 7.176398,
}

GFS_MEAN = {
    "dlwrf": 298.342,
    "dswrf": 168.12321,
    "hcc": 35.272,
    "lcc": 43.578342,
    "mcc": 33.738823,
    "prate": 2.8190969e-05,
    "r": 18.359747,
    "sde": 0.36937004,
    "t": 278.5223,
    "tcc": 66.841606,
    "u10": -0.0022310058,
    "u100": 0.0823025,
    "v10": 0.06219831,
    "v100": 0.0797807,
    "vis": 19628.32,
    "u": 11.645444,
    "v": 0.12330122,
}

GFS_STD = _to_data_array(GFS_STD)
GFS_MEAN = _to_data_array(GFS_MEAN)

# ------ ICON-EU
# Statistics for ICON-EU variables
ICON_EU_STD = {
    "alb_rad": 13.7881,
    "alhfl_s": 73.7198,
    "ashfl_s": 54.8027,
    "asob_s": 55.8319,
    "asob_t": 74.9360,
    "aswdifd_s": 21.4940,
    "aswdifu_s": 18.7688,
    "aswdir_s": 54.4683,
    "athb_s": 34.8575,
    "athb_t": 42.9108,
    "aumfl_s": 0.1460,
    "avmfl_s": 0.1892,
    "cape_con": 32.2570,
    "cape_ml": 106.3998,
    "clch": 39.9324,
    "clcl": 36.3961,
    "clcm": 41.1690,
    "clct": 34.7696,
    "clct_mod": 0.4227,
    "cldepth": 0.1739,
    "h_snow": 0.9012,
    "hbas_con": 1306.6632,
    "htop_con": 1810.5665,
    "htop_dc": 459.0422,
    "hzerocl": 1144.6469,
    "pmsl": 1103.3301,
    "ps": 4761.3184,
    "qv_2m": 0.0024,
    "qv_s": 0.0038,
    "rain_con": 1.7097,
    "rain_gsp": 4.2654,
    "relhum_2m": 15.3779,
    "rho_snow": 120.2461,
    "runoff_g": 0.7410,
    "runoff_s": 2.1930,
    "snow_con": 1.1432,
    "snow_gsp": 1.8154,
    "snowlmt": 656.0699,
    "synmsg_bt_cl_ir10.8": 17.9438,
    "t_2m": 7.7973,
    "t_g": 8.7053,
    "t_snow": 134.6874,
    "tch": 0.0052,
    "tcm": 0.0133,
    "td_2m": 7.1460,
    "tmax_2m": 7.8218,
    "tmin_2m": 7.8346,
    "tot_prec": 5.6312,
    "tqc": 0.0976,
    "tqi": 0.0247,
    "u_10m": 3.8351,
    "v_10m": 5.0083,
    "vmax_10m": 5.5037,
    "w_snow": 286.1510,
    "ww": 27.2974,
    "z0": 0.3901,
}

ICON_EU_MEAN = {
    "alb_rad": 15.4437,
    "alhfl_s": -54.9398,
    "ashfl_s": -19.4684,
    "asob_s": 40.9305,
    "asob_t": 61.9244,
    "aswdifd_s": 19.7813,
    "aswdifu_s": 8.8328,
    "aswdir_s": 29.9820,
    "athb_s": -53.9873,
    "athb_t": -212.8088,
    "aumfl_s": 0.0558,
    "avmfl_s": 0.0078,
    "cape_con": 16.7397,
    "cape_ml": 21.2189,
    "clch": 26.4262,
    "clcl": 57.1591,
    "clcm": 36.1702,
    "clct": 72.9254,
    "clct_mod": 0.5561,
    "cldepth": 0.1356,
    "h_snow": 0.0494,
    "hbas_con": 108.4975,
    "htop_con": 433.0623,
    "htop_dc": 454.0859,
    "hzerocl": 1696.6272,
    "pmsl": 101778.8281,
    "ps": 99114.4766,
    "qv_2m": 0.0049,
    "qv_s": 0.0065,
    "rain_con": 0.4869,
    "rain_gsp": 0.9783,
    "relhum_2m": 78.2258,
    "rho_snow": 62.5032,
    "runoff_g": 0.1301,
    "runoff_s": 0.4119,
    "snow_con": 0.2188,
    "snow_gsp": 0.4317,
    "snowlmt": 1450.3241,
    "synmsg_bt_cl_ir10.8": 265.0639,
    "t_2m": 278.8212,
    "t_g": 279.9216,
    "t_snow": 162.5582,
    "tch": 0.0047,
    "tcm": 0.0091,
    "td_2m": 274.9544,
    "tmax_2m": 279.3550,
    "tmin_2m": 278.2519,
    "tot_prec": 2.1158,
    "tqc": 0.0424,
    "tqi": 0.0108,
    "u_10m": 1.1902,
    "v_10m": -0.4733,
    "vmax_10m": 8.4152,
    "w_snow": 14.5936,
    "ww": 15.3570,
    "z0": 0.2386,
}

ICON_EU_STD = _to_data_array(ICON_EU_STD)
ICON_EU_MEAN = _to_data_array(ICON_EU_MEAN)

NWP_STDS = NWPStatDict(
    ukv=UKV_STD,
    ecmwf=ECMWF_STD,
    gfs=GFS_STD,
    icon_eu=ICON_EU_STD,
)
NWP_MEANS = NWPStatDict(
    ukv=UKV_MEAN,
    ecmwf=ECMWF_MEAN,
    gfs=GFS_MEAN,
    icon_eu=ICON_EU_MEAN,
)

# ------ Satellite
# RSS Mean and std values from randomised 20% of 2020 imagery

RSS_STD = {
    "HRV": 0.11405209,
    "IR_016": 0.21462157,
    "IR_039": 0.04618041,
    "IR_087": 0.06687243,
    "IR_097": 0.0468558,
    "IR_108": 0.17482725,
    "IR_120": 0.06115861,
    "IR_134": 0.04492306,
    "VIS006": 0.12184761,
    "VIS008": 0.13090034,
    "WV_062": 0.16111417,
    "WV_073": 0.12924142,
}

RSS_MEAN = {
    "HRV": 0.09298719,
    "IR_016": 0.17594202,
    "IR_039": 0.86167645,
    "IR_087": 0.7719318,
    "IR_097": 0.8014212,
    "IR_108": 0.71254843,
    "IR_120": 0.89058584,
    "IR_134": 0.944365,
    "VIS006": 0.09633306,
    "VIS008": 0.11426069,
    "WV_062": 0.7359355,
    "WV_073": 0.62479186,
}

RSS_STD = _to_data_array(RSS_STD)
RSS_MEAN = _to_data_array(RSS_MEAN)
