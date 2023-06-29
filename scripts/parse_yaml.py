import numpy as np

class AttrDict(dict):
    """ Nested Attribute Dictionary
    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)
    Adapted from: https://stackoverflow.com/a/48806603
    """

    def __init__(self, mapping=None, *args, **kwargs):
        """Return a class with attributes equal to the input dictionary."""
        super(AttrDict, self).__init__(*args, **kwargs)
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  

    def __getattr__(self, item):
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

def list_to_tuples(config):
    """
    Convert all lists in config dictionary to tuples.
    """
    for key in config.keys():
        for subkey in config[key].keys():
            if type(config[key][subkey]) == list:
                config[key][subkey] = tuple(config[key][subkey])
    return config

def expand_sampling(config, force_int=False):
    """
    Try to expand xsampling keys to to P1 by making them 
    symmetric about the reciprocal space origin. 
    
    Parameters
    ----------
    config : AttrDict object
        dictionary with setup key that contains xsampling keys
    force_int : bool
        if True, force boundaries to nearest integer
    """
    for key in ['hsampling','ksampling','lsampling']:
        if config.setup[key][0] != -1*config.setup[key][1]:
            new_val = max(np.abs(config.setup[key][0]), np.abs(config.setup[key][1]))
            config.setup[key] = (-1*new_val, new_val, config.setup[key][2])
        if force_int:
            config.setup[key] = (np.around(config.setup[key][0]),
                                 np.around(config.setup[key][1]),
                                 config.setup[key][2])
