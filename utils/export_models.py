from abaqusConstants import *

import odbAccess

import glob

import numpy as np

import json,os,sys

def export_models(jobname_pattern, output_dir, file_name=None):
    for odbfile in glob.glob(jobname_pattern):
        print("Processing ODB file:", odbfile)
        odb = odbAccess.openOdb(odbfile)

        inst_name = ""
        for k in odb.rootAssembly.instances.keys():
            if k != "ASSEMBLY":
                inst_name = k
                break
        assert inst_name != "", "No instance found in ODB file"

        inst = odb.rootAssembly.instances[inst_name]

        step_name = odb.steps.keys()[0]
        step = odb.steps[step_name]

        N_n = len(inst.nodes)
        N_e = len(inst.elements)
        N_T = len(step.frames)
        N_ip = 1

        X = np.zeros((N_n, 3), dtype=np.float32)
        mesh = []
        data_stress = np.zeros((N_T, N_e, N_ip, 3, 3), dtype=np.float32)
        data_strain = np.zeros((N_T, N_e, N_ip, 3, 3), dtype=np.float32)
        data_U = np.zeros((N_T, N_n, 3), dtype=np.float32)

        for node in inst.nodes:
            X[node.label - 1, :] = node.coordinates
        
        for element in inst.elements:
            e_mesh = []
            for node in element.connectivity:
                e_mesh.append(node - 1)
            mesh.append(e_mesh)
        
        idx = [(0,0),(1,1),(2,2),(0,1),(0,2),(1,2)]

        for t in range(N_T):
            frame = step.frames[t]
            print("Processing frame", t + 1, "step time:", frame.frameValue)
            u_field = frame.fieldOutputs['U']
            le_field = frame.fieldOutputs['LE']
            s_field = frame.fieldOutputs['S']
            for uv in u_field.values:
                data_U[t, uv.nodeLabel - 1, :] = uv.data
            
            for ev in le_field.values:
                tensor = np.zeros((3,3), dtype=np.float32)
                for ki in range(6):
                    tensor[idx[ki]] = ev.data[ki]
                    if ki >= 3:
                        tensor[idx[ki][0], idx[ki][1]] = ev.data[ki]/2.0
                        tensor[idx[ki][1], idx[ki][0]] = ev.data[ki]/2.0
                assert ev.integrationPoint == 1, "Only one integration point expected"
                data_strain[t, ev.elementLabel - 1, ev.integrationPoint - 1, :] = tensor

            for ev in s_field.values:
                tensor = np.zeros((3,3), dtype=np.float32)
                for ki in range(6):
                    tensor[idx[ki]] = ev.data[ki]
                    if ki >= 3:
                        tensor[idx[ki][0], idx[ki][1]] = ev.data[ki]
                        tensor[idx[ki][1], idx[ki][0]] = ev.data[ki]
                assert ev.integrationPoint == 1, "Only one integration point expected"
                data_stress[t, ev.elementLabel - 1, ev.integrationPoint - 1, :] = tensor
        odbsavedir = os.path.splitext(os.path.basename(odbfile))[0]
        savedir = os.path.join(output_dir, odbsavedir)
        print("Save to ", output_dir,odbsavedir)
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        np.save(savedir+'/nodeX.npy', X)
        with open(os.path.join(savedir+'/mesh.json'), 'w') as f:
            json.dump(mesh, f)
        np.save(savedir+'/elementS.npy', data_stress)
        np.save(savedir+'/elementLE.npy', data_strain)
        np.save(savedir+'/nodeU.npy', data_U)
        odb.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_models.py <jobname_pattern> <output_dir>")
        sys.exit(1)
    
    jobname_pattern = sys.argv[1]
    output_dir = sys.argv[2]

    export_models(jobname_pattern, output_dir)