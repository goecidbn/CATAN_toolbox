
class LoadConfig:
    """ 
    contains the global default values for the data fields 
    on generation, each session object copies this into its own structuree
    and allows modifying it

    defaults for hdf5 are as found in CaImAn standard output
    """

    def __init__(self):
        # default fields from CaImAn output
        self.fields = {
            "spatial": {
                "title": "Spatial data",
                "load": True,
                "type": "static",
                "opts": {
                    "Footprints": "/estimates/A", 
                    "Background": "/estimates/Cn",
                    # "Dimensions": "dims"
                },
            },
            "quality": {
                "title": "Quality data",
                "load": True,
                "type": "dynamic",
                "opts": {
                    "SNR": "/estimates/SNR_comp",
                    "r-value": "/estimates/r_values", 
                    "CNN": "/estimates/cnn_preds"
                },
            },
            "traces": {
                "title": "Trace data",
                "load": False,
                "type": "dynamic",
                "opts": {
                    "C": "/estimates/C",
                    "S": "/estimates/S",
                    "F_dff": "/estimates/F_dff"
                },
            },
        }

    @staticmethod
    def get_defaults():

        obj = LoadConfig()
        return obj.fields
