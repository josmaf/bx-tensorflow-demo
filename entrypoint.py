import json
import os
import trainer  # Module with our code to train a neural network
from shutil import copyfile


# BatchX saves into /batchx/input/input.json what we passed to bx client when running the job
# So now we have to read input.json file to get a dictionary with the input parameters
with open("/batchx/input/input.json", "r") as input_file:
    input_json = json.loads(input_file.read())

# Generated files must be located below '/batchx/output/' folder. This folder has been automatically created by BatchX
output_folder = "/batchx/output/"

# The train method needs:
#     input_json_dict: a dictionary with the input parameters
#     output_folder: path of the folder where model and meta-data files will be saved
model_file_path, meta_file_path = trainer.train(input_json, output_folder)

# Additionally, we copy a script to use the trained model
copyfile('predictor.py', os.path.join(output_folder, 'predictor.py'))

# Write model and meta-data file paths into 'output.json'. BatchX will copy them into its FS. 
with open('/batchx/output/output.json', 'w+') as output_file:
    json.dump({'model_file_path': model_file_path, 'meta_file_path': meta_file_path,
               'predictor_file_path': os.path.join(output_folder, 'predictor.py')}, output_file)
