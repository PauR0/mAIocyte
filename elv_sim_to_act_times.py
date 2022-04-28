
import os

import argparse


import pandas as pd

from sens_elv_exp import SensElvExp, check_attrs



def node_simulation_to_act_t(elv_exp, fname=None, dbg=False, w=False):

    if not check_attrs(elv_exp, ['nodes'], "Can't compute activation times"):
        return

    if fname is None:
        fname = elv_exp.output_path + '/act_times.pickle'

    if os.path.exists(fname) and not w:
        print(f"Warning {fname} already exists, nothing will be saved...")
        return

    data = []
    for node_id in elv_exp.nodes.keys():
        node = elv_exp.nodes[node_id]
        if dbg:
            node.plot(show=True)
        data.append({
            "act_t"     : (node.peaks['min'][0]).tolist(),
            "node_id"   : node_id,
            "cell_type" : elv_exp.cell_type
        })

    act_times = pd.DataFrame(data)
    act_times.to_pickle(fname)
    #


def elv_sim_to_act_times(path,
                        output_path,
                        cell_type=None,
                        dbg=False,
                        fout=None,
                        w=False):

    exp = SensElvExp()
    exp.path=path
    exp.output_path=output_path

    exp.debug = dbg
    exp.cell_type = cell_type

    exp.load_node_info()
    exp.save_nodes_npy()

    node_simulation_to_act_t(exp, fname=fout, dbg=dbg, w=w)
    #

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""Script to create ELVIRA
                                    cases, using a template case.""",
                                    usage=""" """)

    parser.add_argument('-m',
                        '--myo',
                        dest='myo',
                        type=str,
                        help="""Flag to specify if {'ttepi','ttmid','ttendo', 'ttepibz','ttmidbz','ttendobz'}.""")

    parser.add_argument('-w',
                        '--overwrite',
                        dest='w',
                        action='store_true',
                        help=""" Overwrite existing files.""")

    parser.add_argument('-d',
                        '--debug',
                        dest='debug',
                        action='store_true',
                        help="""Run in debug mode. Which essentialy is showing some plots...""")

    parser.add_argument('path',
                        action='store',
                        type=str,
                        help="""Path to an existing case or to a new case to be created.""")

    parser.add_argument('otptdir',
                        type=str,
                        default=None,
                        help=""" The path to output dir, if it does not conform to standard naming.""")


    args = parser.parse_args()
    elv_sim_to_act_times(path=args.path, cell_type=args.myo, output_path=args.otptdir, dbg=args.debug, w=args.w)
