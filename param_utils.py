#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from copy import deepcopy
import json



default_S1S2_params = {
    'metadata' : {
        'version' : None,
        'type' : 'stimulus'
    },
    'data' : {
        'method'      : 'S1S2',
        's1'          : 600,
        's2_ini'      : 1000,
        's2_end'      : 250,
        's2_step'     : 20,
        's1_per_s2'   : 9,
        'tr_off'      : 1000,
        'current'     : 100,
        'st_duration' : 2.0
    }
}

default_case_params = {
    'metadata' : {
        'version' : None,
        'type' : 'case'
    },
    'data' : {
        'myo'      : 'ordendo',
        'dt'       : 0.02,
        'run_elv'  : False,
        'run_post' : False,
        'in_path'  : False,
        'n_cores'  : 8
    }
}

default_AP_params = {
    'metadata' : {
        'version' : None,
        'type' : 'AP_data',
        'date' : None,
        'time' : None
    },
    'data' : {
        'sim_id'    : None,
        't_ini'     : 0.0,
        't_end'     : None,
        't_delta'   : 1.0,
        't_extra'   : 500.0,
        'shape'     : None,
        "save_freq" : 1000,
        "n_proc"    : 4
    }
}

default_EG_params = {
    'metadata' : {
        'version' : None,
        'type' : 'AP_data',
        'date' : None,
        'time' : None
    },
    'data' : {
        't_ini'  : -1000,
        't_end'  : None,
        'electrodes' : {}
    }
}

default_registrate_json = {
        "metadata" : {"version" : "0.1",
                      "type" : "registrate"} ,
        "data" : {"method" : "rigid_align",
                  "torso_dir" : None}
}

default_landmarks_json = {
        "metadata" : {"version" : "0.1",
                      "type" : "vent_landmark"} ,
        "data" : {  "base" : None,
                    "apex" : None,
                    "sept" : None
    }
}

def pretty_write(j,
                 f,
                 write_replacements = None):
    if write_replacements is None:
        write_replacements = [[',',',\n'],
                             ['}}','}\n }'],
                             ['{"','{\n "'],
                             ['"}','"\n}']]
    for r in write_replacements:
        j = j.replace(r[0],r[1])
    j += "\n"
    f.write(j)
#

def get_json_reader(json_filename, template):

    params = deepcopy(template)
    fname = json_filename

    def read_json(path=None, abs_path=False):

        new_params = deepcopy(params)
        try:
            if abs_path:
                json_file = path
            else:
                json_file = path + "/" + fname

            with open(json_file) as param_file:
                read_params = json.load(param_file)
                try:
                    for k in read_params['metadata']:
                        new_params['metadata'][k] = read_params['metadata'][k]
                except KeyError:
                    print(f"WARNING: {json_file} has not metadata info....")
                for k in read_params['data']:
                    new_params['data'][k] = read_params['data'][k]
        except (FileNotFoundError,TypeError):
            pass
        #

        return new_params
    #

    return read_json
#

def get_json_writer(json_filename, template):

    params = deepcopy(template)
    fname = json_filename

    def write_json(path, data=None, abs_path=False):

        if abs_path:
            json_file = path
        else:
            json_file = path + "/" + fname

        try:
            with open(json_file,'w') as param_file:
                if data:
                    for k in data['metadata']:
                        params['metadata'][k] = data['metadata'][k]
                    for k in data['data']:
                        params['data'][k] = data['data'][k]
                pretty_write(json.dumps(params),param_file)
        except FileNotFoundError:
            pass
        #

        return params
    #

    return write_json
#

def update_params(params, exclude_key=None, copy=False, **kwargs):
    """
    Update the param dictionaries with kwargs.

    Arguments:
    ------------

        params : dict
            The dictionary to be updated.

        excluded_key : list['str']
            A list containing the dict entries that wont be updated

        copy : bool
            Whether to make a copy of params.

        **kwargs

    Returns:
    ----------

        params
            The updated dictionary
    """

    if exclude_key is None:
        exclude_key = []

    for k, v in kwargs.items():
        if k not in exclude_key and kwargs[k] is not None:
            if k in params['data']:
                params['data'][k] = v

    return params
#


read_case_json = get_json_reader("elvira_case.json",
                                        default_case_params)
read_stimuli_json = get_json_reader("stimuli.json",
                                        default_S1S2_params)
read_AP_json = get_json_reader("AP_params.json",
                                        default_AP_params)
read_EG_json = get_json_reader("EG_params.json",
                                        default_EG_params)
read_registrate_json = get_json_reader("registrate.json",
                                        default_registrate_json)
read_landmarks_json = get_json_reader("landmarks.json",
                                        default_landmarks_json)

write_case_json = get_json_writer("elvira_case.json",
                                    default_case_params)
write_stimuli_json = get_json_writer("stimuli.json",
                                        default_S1S2_params)
write_AP_json = get_json_writer("AP_params.json",
                                        default_AP_params)
write_EG_json = get_json_writer("EG_params.json",
                                        default_EG_params)
write_registrate_json = get_json_writer("registrate.json",
                                        default_registrate_json)
write_landmarks_json = get_json_writer("landmarks.json",
                                        default_landmarks_json)
