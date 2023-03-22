#!/usr/bin/env python3

import os, sys

from datetime import datetime

import re

import argparse

from subprocess import Popen

from distutils.dir_util import copy_tree

from shutil import rmtree

from utils import check_s2_params

from param_utils import (read_case_json, write_case_json,
                         read_stimuli_json, write_stimuli_json,
                         update_params)


#Template priority is os.environ > venv template_case > template located where this file is.
FILE_DIR = sys.path[0]
if not FILE_DIR:
    FILE_DIR='.'
TEMP_PATH = FILE_DIR+"/case"
if 'SC_TEMP_CASE' in os.environ:
    print("[make_elvira_case] Using template case in "+os.environ['SC_TEMP_CASE'])
    TEMP_PATH = os.environ['CENTERLINE_TEMPLATE_CASE']
elif not (sys.prefix == sys.base_prefix): #Determine if running inside a venv
        venv_temp_case_path = f"{sys.prefix}/share/template_case"
        if os.path.exists(venv_temp_case_path) and os.path.isdir(venv_temp_case_path):
            print(f"[make_elvira_case] Using template case in {venv_temp_case_path}")
            TEMP_PATH = venv_temp_case_path


def make_case_id(path, stim_params):
    """
    Provides an id for a case and its files
    """

    if stim_params['method'] == 'S1S2':
        cid =f"{stim_params['s1']}_{stim_params['s2_step']}"

    else:
        cid = datetime.now().strftime("%H:%M:%S_%d-%m%Y")

    return cid
#

def make_S1S2_stimulus_file(path,
                            case_id,
                            s1,
                            s2_ini=None,
                            s2_end=250,
                            s2_step=20,
                            s1_per_s2=9,
                            train_offset=0,
                            st_duration=2.0,
                            current=100.0,
                            w=False,
                            **kwargs):

    """
    Function to make S1S2 stimuli file for ELVIRA case.
    It starts with a train of s1_per_s2 stimuli at s1 CL
    followed by an s2 stimulus. This is repeated varying the
    value of s2 according to a linear progrssion from s2_ini
    to s2_end with steps of s2_step. An offset between S1S2
    trains can be added using train_offset.

    The sign of s2_step is checked to be conistent with s2_ini
    and s2_end.

    Arguments:
    ----------

        s1 : float
            The S1 or basic cycle length expressed in ms.

        s2_ini : float
            The first s2 value expressed in ms.

        s2_end : float
            The last s2 value expressed in ms.

        s2_step : float
            The step to transite from s2_ini to s2_end expressed in ms.

        s1_per_s2 : int
            The number of s1 to perform at each S1S2 trains.

        train_offset : float
            An offset to be added at the end of an S1S2 train expressed in ms.

        st_duration : float
            The duration of the stimuli in ms.

        current : float
            The current to impose at the stimuli line in mV.


    """


    fname = f"{path}/data/file_stimulus_{case_id}.dat"
    if os.path.exists(fname) and not w:
        print(f"ERROR: file {fname} already exists and overwite is set to false...")
        return False, False

    if s2_ini is None:
        s2_ini = s1

    s2_step, cond = check_s2_params(s2_ini=s2_ini, s2_end=s2_end, s2_step=s2_step)

    t = 0.0
    stim_line = f""
    n_stim = 0
    s2 = s2_ini
    while cond(s2):
        for i in range(s1_per_s2):
            stim_line += f"{t} {st_duration} {current} "
            if i == s1_per_s2-1:
                t += s2
            else:
                t += s1
            n_stim += 1

        stim_line += f" {t} {st_duration} {current} "
        t += s1 + train_offset
        n_stim+=1
        s2 +=s2_step

    stim_line = f"1\t {n_stim} " + stim_line + '\n'

    #Read template lines
    with open(f"{TEMP_PATH}/data/file_stimulus.dat", 'r') as ftemp:
        lines = ftemp.readlines()

    #Write Lines
    with open(fname, 'w') as f_stim:
        for i, l in enumerate(lines):
            if i == 1:
                l = stim_line
            f_stim.write(l)

    return t
#

def make_stimulus_file(path, case_id, params, w=False):
    """
    TODO: Make other stimuli methods.
    """

    t_max = 0

    if params['method'] == 'S1S2':
        t_max = make_S1S2_stimulus_file(path, case_id=case_id, **params, w=w)

    else:
        print("ERROR: Only S1S2 method is supported currently....")

    return t_max
#

def make_fibre_file(path, myo, w=False):
    """
    Create the fibre file according to the element properties and cell models defined at PROP_NOD (main).

    Arguments:
    ------------
        path :  str
            Path to the case directory

        myo  : int
            Cell model expressed in the PROP_NOD

        w : bool
            Wether to overwrite
    """


    fibre_fname = path+'/data/fibre.dat'
    if os.path.exists(fibre_fname) and not w:
        print(f"WARNING: file {fibre_fname} already exists and overwite is set to false...")

    if myo.lower() in ['ttendo', 'ttmid', 'ttepi']:
        material = 8
    elif myo.lower() in ['ttendobz','ttmidbz','ttepibz']:
        material = 9
    elif myo.lower() in ['ordendo', 'ordmid', 'ordepi']:
        material = 10
    elif myo.lower() == 'CM':
        material = 2
    else:
        print("ERROR: myo arg could not be understood. Available values are: \n",
              "\t CM       : Courtemanche "
              "\t ttendo   : tenTusscher ENDO,\n",
              "\t ttmid    : tenTusscher MID,\n",
              "\t ttepi    : tenTusscher EPI, \n"
              "\t ttendobz : tenTusscher ENDO BZ,\n",
              "\t ttmidbz  : tenTusscher MID BZ,\n",
              "\t ttepibz  : tenTusscher EPI BZ",
              "\t ordendo  : O'Hara Rudy ENDO\n",
              "\t ordmid   : O'Hara Rudy MID\n",
              "\t ordepi   : O'Hara Rudy EPI\n")
        return

    with open(fibre_fname, 'r') as fr:
        lines = fr.readlines()

    with open(fibre_fname, 'w') as fw:
        for i, l in enumerate(lines):
            if i == 1:
                l = f"1\t {material}\t 0\t 3\t 0.0 1.0 0.0\n"
            fw.write(l)
#

def make_node_file(path, myo, w=False):

    nodes_fname = path+'/data/nodes.dat'
    if os.path.exists(nodes_fname) and not w:
        print(f"WARNING: file {nodes_fname} already exists and overwite is set to false...")

    if myo == 'CM':
        myo_num = 1
    elif myo == 'ttendo':
        myo_num = 2
    elif myo == 'ttmid':
        myo_num = 3
    elif myo.lower() == 'ttepi':
        myo_num = 4
    elif myo == 'ttendobz':
        myo_num = 5
    elif myo == 'ttmidbz':
        myo_num = 6
    elif myo.lower() == 'ttepibz':
        myo_num = 7
    elif myo.lower() == 'ordendo':
        myo_num = 9
    elif myo.lower() == 'ordmid':
        myo_num = 10
    elif myo.lower() == 'ordepi':
        myo_num = 11

    else:
        print("ERROR: myo arg could not be understood. Available values are: \n",
              "\t CM       : Courtemanche "
              "\t ttendo   : Ten Tusscher ENDO,\n",
              "\t ttmid    : Ten Tusscher MID,\n",
              "\t ttepi    : Ten Tusscher EPI, \n"
              "\t ttendobz : Ten Tusscher ENDO BZ,\n",
              "\t ttmidbz  : Ten Tusscher MID BZ,\n",
              "\t ttepibz  : Ten Tusscher EPI BZ\n",
              "\t ordendo  : O'Hara Rudy ENDO\n",
              "\t ordmid   : O'Hara Rudy MID\n",
              "\t ordepi   : O'Hara Rudy EPI\n")
        return

    if myo != 1:
        #Read lines
        with open(nodes_fname, 'r') as fr:
            lines = fr.readlines()
        with open(nodes_fname, 'w') as fw:
            for i, l in enumerate(lines):
                if i > 1:
                    l = re.sub('\s+1\s+', f'\t{myo_num}\t', l, count=1)
                fw.write(l)
        make_fibre_file(path, myo, w=w)
#

def make_output_dir(path, case_id, w=False):

    dname = f"{path}/output_{case_id}"

    if os.path.exists(dname) and not w:
        print(f"ERROR: directory {dname} already exists and overwritting is set to false....")
        return False
    elif os.path.exists(dname):
        rmtree(dname)

    os.mkdir(dname)

    return dname
#

def make_main_file(path, case_id, t_max, dt):


    main_fname = f'{path}/data/main_{case_id}.dat'

    #Read template lines
    with open(f"{TEMP_PATH}/data/main_file.dat", 'r') as ftemp:
        lines = ftemp.readlines()

    edit_next = False
    with open(main_fname,'w') as fmain:
        for line in lines:

            if edit_next:
                fmain.write(new_line)
                edit_next = False
                continue

            if '*TIME_INC' in line:
                edit_next = True
                new_line = f'{dt} {dt} {t_max:.2f} {dt} 1\n'#dt valid for Coutermanche case

            if '#TITLE' in line:
                edit_next = True
                new_line = f'Slab3D_{case_id}\n'

            if '*STIMULUS' in line:
                line = f'*STIMULUS, FILE:\"file_stimulus_{case_id}.dat\"\n'

            fmain.write(line)


    return main_fname
#

def make_run_post_config_file(path, ncores, s1, s2_step, ftype=1):

    fname = f"{path}/data/post_config_{s1}_{s2_step}.dat"
    #Read template lines
    with open(f"{TEMP_PATH}/data/post_config.dat", 'r') as ftemp:
        lines = ftemp.readlines()

    #Write file
    with open(fname,'w') as f:

        edit_next=False
        for line in lines:

            if line.startswith('!'):
                continue

            if "#POTENFILE" in line:
                f.write(line)
                f.write(f"{ncores}\n")
                for i in range(ncores):
                    f.write(f"ecg_prc_{i}.bin\n")
                continue

            if "#POSTFILENAME" in line:
                f.write(line)
                f.write(f"Bloque_{s1}_{s2_step}\n")
                continue

            if "#FILETYPE" in line:
                f.write(line)
                f.write(f"{ftype}\n")
                continue
            f.write(line)

    return fname
#

def make_case(path,
              myo,
              dt,
              stim_params,
              in_path=False,
              run=False,
              wait=False,
              run_post=False,
              n_cores=None,
              w=False):

    if in_path:
        if not os.path.exists(path) or not os.path.isdir(path):
            print("ERROR: Given path does not exist....")
            return False
    else:
        if os.path.exists(path) and not w:
            print("ERROR: Given path already exists and overwrite is set to false. Try with -w flag.")
            return False
        else:
            copy_tree(TEMP_PATH, path)
            for f in ['/data/file_stimulus.dat', '/data/main_file.dat', '/data/post_config.dat']:
                os.remove(path+f)
            w = True


    cid = make_case_id(path=path, stim_params=stim_params)

    t_max = make_stimulus_file(path=path,
                               case_id=cid,
                               params=stim_params,
                               w=w)

    if myo is not None and not in_path:
        make_node_file(path, myo, w=w)

    output_dname = make_output_dir(path, case_id=cid, w=w)

    main_file_fname = make_main_file(path=path,
                                     case_id=cid,
                                     t_max=t_max,
                                     dt=dt)

    exec_command = [FILE_DIR + "/runelvBZ.sh", str(n_cores), f"data/main_{cid}.dat", f"output_{cid}", f"output_{cid}/log"]
    print(exec_command)
    if run:
        spro = Popen(exec_command, cwd=path)

    if wait or run_post:
        spro.communicate()
    if run_post:
        make_run_post_config_file(path, n_cores, case_id=cid)
        exec_command = [FILE_DIR+"/runpost.sh", f"data/post_config_{cid}.dat", f"output_{cid}"]
        spro = Popen(exec_command, cwd=path)

    return cid
#


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to create ELVIRA
                                    cases, using a template case.""",
                                    usage = """ """)

    parser.add_argument('-c',
                        '--case-params',
                        dest='case_params',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to an case_params.json.""")

    parser.add_argument('-s',
                        '--stim-params',
                        '--stimuli-params',
                        dest='stim_params',
                        action='store',
                        default=None,
                        type=str,
                        nargs='?',
                        help="""Path to an stimuli_params.json.""")

    parser.add_argument('-t',
                        '--dt',
                        dest='dt',
                        default=None,
                        type=str,
                        help="""Time differential - to be chosen appropriately considering
                        the model used (eg. CM 0.01, tenTusscher 0.02).""")

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        default=None,
                        type=str,
                        help="""Flag to specify if ttepi - ttmid - ttendo. Default ttendo.""")

    parser.add_argument('-r',
                        '--run-elv',
                        dest='run_elv',
                        action='store_true',
                        default=None,
                        help="""Run ELVIRA on the generated case.""")

    parser.add_argument('-o',
                        '--run-post',
                        dest='run_post',
                        action='store_true',
                        default=None,
                        help="""Make ensight files after the simulation.""")

    parser.add_argument('--template-path',
                        dest='temp_path',
                        type=str,
                        nargs='?',
                        default=None,
                        help=""" A path to an alternative template case.""")

    parser.add_argument('-i',
                        '--in-path',
                        dest='in_path',
                        action='store_true',
                        default=None,
                        help="""Save files in an already existing case.""")

    parser.add_argument('--wait',
                        dest='wait',
                        action='store_true',
                        default=None,
                        help=""" Make the python process wait for the simulation
                        to finish.""")

    parser.add_argument('-w',
                        '--overwrite',
                        dest='w',
                        action='store_true',
                        help=""" Overwrite existing files.""")

    parser.add_argument('-n',
                        '--n-cores',
                        dest='n_cores',
                        default=None,
                        action='store',
                        type=int,
                        help="""Number of cores to use for runing the simulation.""")


    parser.add_argument('path',
                        action='store',
                        type=str,
                        nargs='?',
                        help="""Path to an existing case or to a new case to be created.""")



    args = parser.parse_args()

    if args.temp_path is not None:
        if not os.path.exists(args.temp_path) or not os.path.isdir(args.temp_path):
            print("ERROR: Given template path does not exist or is not a valid directory")
        else:
            TEMP_PATH = args.temp_path

    if args.path is None:
        print("ERROR: Wrong path given ....")
        sys.exit()

    if args.case_params is not None:
        case_params = read_case_json(path=args.case_params, abs_path=True)
    else:
        case_params = read_case_json(path=args.case_params)
    case_params = update_params(params   = case_params,
                                myo      = args.myo,
                                dt       = args.dt,
                                run_elv  = args.run_elv,
                                run_post = args.run_post,
                                in_path  = args.in_path,
                                n_cores  = args.n_cores)

    if args.stim_params is not None:
        stim_params = read_stimuli_json(path=args.stim_params, abs_path=True)
    else:
        stim_params = read_stimuli_json(path=args.path)


    cid = make_case(path        = args.path,
                    myo         = case_params['data']['myo'],
                    dt          = case_params['data']['dt'],
                    stim_params = stim_params['data'],
                    in_path     = case_params['data']['in_path'],
                    run         = case_params['data']['run_elv'],
                    wait        = args.wait,
                    run_post    = case_params['data']['run_post'],
                    n_cores     = case_params['data']['n_cores'],
                    w           = args.w)

    write_case_json(args.path+f'/case_{cid}.json', case_params, abs_path=True)
    write_stimuli_json(args.path+f'/stimuli_{cid}.json', stim_params, abs_path=True)
