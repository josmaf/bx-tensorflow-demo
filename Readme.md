# Build image
docker build -f ./docker/Dockerfile josemfer/batchx-demo-tensorflow:latest .

# Import into Batchx
bx import josemfer/batchx-demo-tensorflow:latest

# Run in BX
bx run -g=1 tutorial/demo-tensorflow-gpu:1.0.0 '{}'

# LABEL requirements to use GPU

Add: "runtime\":{"minMem":1000, "gpus\" : "required|supported"}

    - required: using "g" parameter with bx cliente is mandatory
    - supported: using "g" parameter with bx client is optional 







