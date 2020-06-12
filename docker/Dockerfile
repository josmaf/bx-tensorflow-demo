FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /batchx
COPY . .
RUN chmod -R +x /batchx  && pip3 install imageio==2.8.0 && pip3 install pandas==1.0.4
ENTRYPOINT python3 /batchx/entrypoint.py
LABEL 'io.batchx.manifest-03'='{ \
	"name":"tutorial/tensorflow-gpu-demo", \
	"title":"BatchX Tensorflow GPU Demo Image", \
	"schema":{  \
		"input":{ \
			"$schema":"http://json-schema.org/draft-07/schema#", \
			"type":"object", \
			"properties":{ \
				"training_images_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Zip file with training images" \
					}, \
                "training_labels_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Csv file with training labels" \
					}, \
				"testing_images_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Zip file with testing images" \
					}, \
                "testing_labels_path":{ \
					"type":"string", \
					"required":true, \
					"format":"file", \
					"description":"Csv file with testing labels" \
					}, \
				"num_epochs":{ \
					"type":"integer", \
					"required":true, \
					"description":"Number of epochs" \
					} \
				} \
			}, \
		"output":{ \
			"$schema":"http://json-schema.org/draft-07/schema#", \
			"type":"object", \
			"additionalProperties":false, \
			"properties":{ \
				"model_file_path":{ \
					"required":true, \
					"type":"string", \
					"format":"file", \
					"description":"Tensorflow HDF5 model file path" \
					}, \
				"meta_file_path":{ \
					"required":true, \
					"type":"string", \
					"format":"file", \
					"description":"Meta-data file path" \
					}, \
				"predictor_file_path" : { \
					"required":true, \
					"type":"string", \
					"format":"file", \
					"description":"Predictor script file path" \
					} \
				} \
			} \
		}, \
	"author":"BatchX", \
	"version":"0.0.7", \
	"runtime":{"minMem":8000, \
				"gpus":"required" \
			} \
	}'
