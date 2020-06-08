import json
import os
import trainer  # Module with our code to train a neural network

# BatchX saves into /batchx/input/input.json what we passed to bx client when running the job
# So now we have to read input.json file to get the input parameters
with open("/batchx/input/input.json", "r") as input_file:
    input_json_dict = json.loads(input_file.read()) 

# Get input data file local path
input_file_path = input_json_dict["input_file_path"]  

# Get number of epochs
num_epochs = input_json_dict["num_epochs"]

# Generated files must be located somewhere below '/batchx/output/' folder
# This folder has been automatically created by BatchX
output_folder = "/batchx/output/"

# Your workload-code, packed in the trainer module, goes here
# The train method needs:
#     input_data_file: path of training data file
#     num_epochs: number of epochs (iterations)
#     output_folder: path of the folder where model and meta-data files will be saved
# It will return the paths of generated model and meta-data files
model_file_path, meta_file_path = trainer.train(input_file_path, num_epochs, output_folder)

# Write model and meta-data file paths into 'output.json'. BatchX will copy them into its FS. 
with open('/batchx/output/output.json', 'w+') as output_file:
    json.dump({'model_file_path': model_file_path, 'meta_file_path' :  meta_file_path}, output_file)
