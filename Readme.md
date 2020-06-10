# Build Docker image
docker build -f ./docker/Dockerfile -t josemfer/batchx-tensorflow-gpu-demo:latest .

# Run Docker image
docker run -v /batchx/input:/batchx/input -v /batchx/output:/batchx/output josemfer/batchx-tensorflow-gpu-demo:latest

# Import into Batchx
bx import josemfer/batchx-tensorflow-gpu-demo:latest

# Run in BX

## Copy training data into BatchX file fystem
bx cp ...

## Run image
bx run -g=1 tutorial/tensorflow-gpu-demo:0.0.1 '{"data_file_path": "", "num_epochs": 2}'







