import abaqus
from abaqusConstants import *

import odbAccess

import numpy as np

model_name = abaqus.getInput("Enter the model name: ", "Model-1")
assert mdb.models.has_key(model_name), "Model not found: {}".format(model_name)

section_name = abaqus.getInput("Enter the section name: ", "Section-1")
assert mdb.models[model_name].sections.has_key(section_name), "Section not found"

model = mdb.models[model_name]

pt_file_prefix = abaqus.getInput("Enter pt file format: ", "SGHMC_%03d")

range_str = abaqus.getInput("Enter the range of pt files: ", "range(10)")

submit_type = abaqus.getInput("Submit type (CREATE_ONLY/SEQUENTIAL): ", "CREATE_ONLY/SEQUENTIAL").upper()
assert submit_type in ["CREATE_ONLY", "SEQUENTIAL"], "Invalid submit type"

ran = eval(range_str)

for i in ran:
    mat_name = pt_file_prefix % i
    if not model.materials.has_key(mat_name):
        mat = model.Material(name=mat_name)
        mat.UserMaterial()
    sec = model.sections[section_name]
    sec.setValues(material=mat_name)

    job = mdb.Job(name="Job-"+mat_name, model=model, description="Job for {}".format(mat_name),
                  numCpus=10, numDomains=10, userSubroutine='D:\\dev\\ABQnn\\UMAT_allmodels.for',)
    if submit_type == "CREATE_ONLY":
        job.writeInput()
        print("Created job input for:", mat_name)
    elif submit_type == "SEQUENTIAL":
        job.submit()
        job.waitForCompletion()
        print("Submitted and completed job for:", mat_name)
    
