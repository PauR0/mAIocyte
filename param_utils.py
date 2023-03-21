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

default_AP_data = {
    'metadata' : {
        'version' : None,
        'type' : 'AP_data',
        'date' : None,
        'time' : None
    },
    'data' : {
        't_ini'  : 0.0,
        't_end'  : None,
        't_delta': 1.0,
        't_extra'  : 500.0,
        'shape' : None
    }
}

default_EG_data = {
    'metadata' : {
        'version' : None,
        'type' : 'AP_data',
        'date' : None,
        'time' : None
    },
    'data' : {
        't_ini'  : None,
        't_end'  : None,
        'probes' : {}
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

def get_json_reader(json_filename,template):

    params = deepcopy(template)
    fname = json_filename

    def read_json(path=None,abs_path=False):

        new_params = deepcopy(params)
        try:
            if abs_path:
                json_file = path
            else:
                json_file = path + "/" + fname

            with open(json_file) as param_file:
                read_params = json.load(param_file)
                for k in read_params['metadata']:
                    new_params['metadata'][k] = read_params['metadata'][k]
                for k in read_params['data']:
                    new_params['data'][k] = read_params['data'][k]
        except (FileNotFoundError,TypeError):
            pass
        #

        return new_params
    #

    return read_json
#

def get_json_writer(json_filename,template):

    params = deepcopy(template)
    fname = json_filename

    def write_json(path,data=None,abs_path=False):

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



read_case_json = get_json_reader("elvira_case.json",
                                        default_case_params)
read_stimuli_json = get_json_reader("stimuli.json",
                                        default_lnr_stimuli_params)
read_AP_json = get_json_reader("AP_data.json",
                                        default_AP_data)
read_EG_json = get_json_reader("EG_data.json",
                                        default_EG_data)

write_case_json = get_json_writer("elvira_case.json",
                                    default_case_params)
write_stimuli_json = get_json_writer("stimuli.json",
                                        default_lnr_stimuli_params)
write_AP_json = get_json_writer("AP_data.json",
                                        default_AP_data)
write_EG_json = get_json_writer("EG_data.json",
                                        default_EG_data)