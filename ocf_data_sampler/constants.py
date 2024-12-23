import xarray as xr
import numpy as np


NWP_PROVIDERS = [
    "ukv",
    "ecmwf",
]


def _to_data_array(d):
    return xr.DataArray(
        [d[k] for k in d.keys()],
        coords={"channel": [k for k in d.keys()]},
    ).astype(np.float32)


class NWPStatDict(dict):
    """Custom dictionary class to hold NWP normalization stats"""

    def __getitem__(self, key):
        if key not in NWP_PROVIDERS:
            raise KeyError(f"{key} is not a supported NWP provider - {NWP_PROVIDERS}")
        elif key in self.keys():
            return super().__getitem__(key)
        else:
            raise KeyError(
                f"Values for {key} not yet available in ocf-data-sampler {list(self.keys())}"
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
    "sde": 0.000913831521756947,
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
    "sde": 8.107526082312688e-05,
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

NWP_STDS = NWPStatDict(
    ukv=UKV_STD,
    ecmwf=ECMWF_STD,
)
NWP_MEANS = NWPStatDict(
    ukv=UKV_MEAN,
    ecmwf=ECMWF_MEAN,
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
