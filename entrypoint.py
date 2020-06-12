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

# Train the model and get the path of the generated model file and a meta-data info file
model_file_path, meta_file_path = trainer.train(train_images_path=input_json['training_images_path'],
                                                train_labels_path=input_json['training_labels_path'],
                                                test_images_path=input_json['testing_images_path'],
                                                test_labels_path=input_json['testing_labels_path'],
                                                num_epochs=input_json['num_epochs'],
                                                output_folder=output_folder)

# Additionally, we copy a script to use the trained model
copyfile('predictor.py', os.path.join(output_folder, 'predictor.py'))

# Write model and meta-data file paths into 'output.json'. BatchX will copy them into its FS. 
with open('/batchx/output/output.json', 'w+') as output_file:
    json.dump({'model_file_path': model_file_path, 'meta_file_path': meta_file_path,
               'predictor_file_path': os.path.join(output_folder, 'predictor.py')}, output_file)
