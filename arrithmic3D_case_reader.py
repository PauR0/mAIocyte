#! /usr/bin/env python3

import os

import numpy as np

import pyvista as pv


class A3DENS:

    def __init__(self) -> None:

        self.path : str = None
        self.case_fname : str  = None
        self.geom_fname : str  = None
        self.fields  : dict = None
    #


    def read_ens_case_file(self, fname):

        if not fname.endswith('.case'):
            fname += '.case'
        time_ = {}

        self.path, self.case_fname = os.path.split(fname)

        with open(fname, 'r') as case_file:

            mode = None
            aux=False
            for line in case_file:

                if line.endswith('\n'):
                    line = line[:-1]
                if line:
                    if not line.isupper():
                        if mode == "GEOMETRY":
                            for token in line.split():
                                if token.endswith('.geo'):
                                    self.geom_fname = token

                        elif mode == "VARIABLE":
                            _, info = line.split(':')
                            field, field_fname = info.split()
                            self.fields[field] = {}
                            self.fields[field]['file_names'] = field_fname

                        elif mode == "TIME":
                            if aux:
                                time_['time values'] = [float(t) for t in line.split()]
                                aux=False
                            elif 'time values' in line:
                                aux=True
                            else:
                                k, v = line.split(':')
                                time_[k] = eval(v)

                    else:
                        mode = line
                        if line == "VARIABLE":
                            self.fields = {}


        for field in self.fields.keys():
            self.fields[field]['time_data'] = time_
    #

    def load_variables(self, var=None):
        if self.path is None:
            print("ERROR: Can't load variables, path is None...")

        if var is None:
            var=self.fields.keys()

        for f, fd in self.fields.items():
            if f in var:
                name_re = fd['file_names']
                fd['data'] = []
                zpad = name_re.count('*')
                for i in range(fd['time_data']['number of steps']):
                    n =  name_re.replace("*" * zpad, f"{i}".zfill(zpad))
                    fname = f"{self.path}/{n}"
                    fd['data'].append(np.loadtxt(fname, skiprows=4))
#end Ar3DENS




if __name__ == '__main__':

    case_dir = "/Users/pau/Electrophys/Automata/p2_Bivent"
    ens_dir  = "A_Map_Berruezo_p2_Bivent_IDPacing-39016_6S1-600_1S2-285"
    enscase_fname = f"{case_dir}/{ens_dir}/VentA_Map_Berruezo_p2_Bivent.case"
    mesh_fname = f"/Users/pau/Electrophys/Automata/p2_Bivent/ventricle_Tagged.vtk"


    mesh = pv.read(mesh_fname)
    ens = A3DENS()
    ens.read_ens_case_file(enscase_fname)
    ens.load_variables()

    mesh['LAT0'] = ens.fields['Activation_Map(ms)']['data'][0]
    mesh_bz = mesh.threshold([1,1.2],'Cell_type')
    mesh_bz = mesh_bz.threshold(0,'LAT0')
    mesh_bz.plot(scalars='LAT0')
