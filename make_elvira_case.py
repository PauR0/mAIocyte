#!/usr/bin/env python3

import os, sys

import re

import argparse

from subprocess import Popen

from distutils.dir_util import copy_tree

from shutil import rmtree

from utils import check_s2_params


FILE_DIR = sys.path[0]
if not FILE_DIR:
    FILE_DIR='.'
TEMP_PATH = FILE_DIR+'/BLOQUE_12x12_250micras'

def make_stimulus_file(path,
                       s1,
                       s2_ini=None,
                       s2_end=250,
                       s2_step=20,
                       s1_per_s2=9,
                       train_offset=0,
                       st_duration=2.0,
                       current=100.0,
                       abs_path=False,
                       w=False):

    if not abs_path:
        fname = f"{path}/data/file_stimulus_{s1}_{s2_step}.dat"
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

    return os.path.split(fname)[1], t
#

def make_node_file(path, myo, bz, w=False):

    nodes_fname = path+'/data/nodes.dat'
    if os.path.exists(nodes_fname) and not w:
        print(f"WARNING: file {nodes_fname} already exists and overwite is set to false...")

    if myo == 'ttendo':
        myo = 1
    elif myo == 'ttmid':
        myo = 2
    elif myo.lower() == 'ttepi':
        myo = 3
    else:
        print("ERROR: myo arg could not be understood. Available values are: \n",
             "ttendo : tenTusscher ENDO,\n",
             "ttmid : tenTusscher MID,\n",
             "ttepi : tenTusscher EPI")
        return
    if bz:
        myo+=3

    if myo != 1:
        #Read lines
        with open(nodes_fname, 'r') as fr:
            lines = fr.readlines()
        with open(nodes_fname, 'w') as fw:
            for i, l in enumerate(lines):
                if i > 1:
                    l = re.sub('\t1\t', f'\t{myo}\t', l)
                fw.write(l)

def make_output_dir(path, s1, s2_step, w=False):

    dname = f"{path}/output_{s1}_{s2_step}"

    if os.path.exists(dname) and not w:
        print(f"ERROR: directory {dname} already exists and overwritting is set to false....")
        return False
    elif os.path.exists(dname):
        rmtree(dname)

    os.mkdir(dname)

    return dname

def make_main_file(path, s1, s2_step, sti_fname, t_max):


    main_fname = f'{path}/data/main_{s1}_{s2_step}.dat'

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
                new_line = f'0.02 0.02 {t_max:.2f} 0.02 0\n'

            if '#TITLE' in line:
                edit_next = True
                new_line = f'Bloque3D_{s1}_{s2_step}\n'

            if '*STIMULUS' in line:
                line = f'*STIMULUS, FILE:"{sti_fname}"\n'

            fmain.write(line)


    return main_fname

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

def make_case(path,
              s1,
              s2_ini=None,
              s2_end=250,
              s2_step=-20,
              s1_per_s2=9,
              train_offset=0,
              current=100.0,
              myo=None,
              in_path=False,
              bz=False,
              run=False,
              wait=False,
              run_post=False,
              n_cores=6,
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

    stimulus_fname, t_max = make_stimulus_file(path=path,
                                               s1=s1,
                                               s2_ini=s2_ini,
                                               s2_end=s2_end,
                                               s2_step=s2_step,
                                               s1_per_s2=s1_per_s2,
                                               train_offset=train_offset,
                                               st_duration=2,
                                               current=current,
                                               w=w)

    if (myo is not None or bz) and not in_path:
        make_node_file(path, myo, bz, w=w)

    output_dname = make_output_dir(path, s1, s2_step, w=w)

    main_file_fname = make_main_file(path, s1, s2_step, stimulus_fname, t_max)

    exec_command = [FILE_DIR + "/runelvBZ.sh", str(n_cores), main_file_fname, f"{output_dname}/", f"{output_dname}/log"]
    if run:
        spro = Popen(exec_command)

    if wait or run_post:
        spro.communicate()
    if run_post:
        post_config_fname = make_run_post_config_file(path, n_cores, s1, s2_step)
        exec_command = FILE_DIR + f"/runpost.sh {post_config_fname} {output_dname} &"
        spro = Popen(exec_command)
#


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to create ELVIRA
                                    cases, using a template case.""",
                                    usage = """ """)

    parser.add_argument('-s',
                        '--s1',
                        dest='s1',
                        action='store',
                        default=1000,
                        type=int,
                        help="""Lapse of time between main consecutive stimuli
                        expressed in ms.""")

    parser.add_argument('-p',
                        '--s2-step',
                        dest='s2_step',
                        default=20,
                        action='store',
                        type=int,
                        help="""Lapse of time between the 10th S1 and the S2 stimuli
                        expressed in ms.""")

    parser.add_argument('--s2-end',
                        dest='s2_end',
                        default=250,
                        action='store',
                        type=float,
                        help="""Last reached value of S2.""")

    parser.add_argument('--s2-ini',
                        dest='s2_ini',
                        type=float,
                        default=None,
                        action='store',
                        help="""Initial S2 for the stimuli train.""")

    parser.add_argument('--tr-off',
                        dest='tr_off',
                        type=float,
                        default=0,
                        action='store',
                        help="""An extra offset time at the end of each stimuli train.""")

    parser.add_argument('-c',
                        '--current',
                        dest='current',
                        type=int,
                        default=100,
                        action='store',
                        help="""Current amplitude of the stimulus.""")

    parser.add_argument('--s1-per-s2',
                        dest='s1_per_s2',
                        type=int,
                        default=9,
                        action='store',
                        help="""Number of S1 stimuli before each S2.""")

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        default='ttendo',
                        type=str,
                        help="""Flag to specify if ttepi - ttmid - ttendo. Default ttendo.""")

    parser.add_argument('-b',
                        '--border-zone',
                        dest='bz',
                        action='store_true',
                        help="""Set the node type to Border Zone.""")

    parser.add_argument('-r',
                        '--run-elv',
                        dest='r',
                        action='store_true',
                        help="""Run ELVIRA on the generated case.""")

    parser.add_argument('-o',
                        '--run-post',
                        dest='o',
                        action='store_true',
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
                        help=""" Save files in an already existing case.""")

    parser.add_argument('-w',
                        '--overwrite',
                        dest='w',
                        action='store_true',
                        help=""" Overwrite existing files.""")

    parser.add_argument('-n',
                        '--n-cores',
                        dest='nc',
                        default=8,
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
        exit()

    make_case(path=args.path,
              s1=args.s1,
              s2_ini=args.s2_ini,
              s2_end=args.s2_end,
              s2_step=args.s2_step,
              s1_per_s2=args.s1_per_s2,
              train_offset=args.tr_off,
              current=args.current,
              myo=args.myo,
              in_path=args.in_path,
              bz=args.bz,
              run=args.r,
              run_post=args.o,
              n_cores=args.nc,
              w=args.w)
