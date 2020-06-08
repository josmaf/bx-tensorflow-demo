FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /batchx
COPY . .
RUN chmod -R +x /batchx 
ENTRYPOINT python3 /batchx/entrypoint.py
LABEL 'io.batchx.manifest-03'='{"name":"tutorial/batchx-demo-tensorflow","title":"BatchX Tensorflow Demo Image","schema":{"input":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"input_file_path":{"type":"string","required":true,"format":"file","description":"Input data file path"},"num_epochs":{"type":"integer","required":true,"description":"Number of epochs"}}},"output":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","additionalProperties":false,"properties":{"model_file_path":{"required":true,"type":"string","format":"file","description":"Tensorflow HDF5 model file path"},"meta_file_path":{"required":true,"type":"string","format":"file","description":"Meta-data file path"}}}},"author":"BatchX","version":"0.0.1","runtime":{"minMem":8000,"gpus":"supported"}}'
