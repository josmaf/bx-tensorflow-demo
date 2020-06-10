# Build Docker image
docker build -f ./docker/Dockerfile -t josemfer/batchx-tensorflow-gpu-demo:latest .

# Run Docker image
docker run -v /batchx/input:/batchx/input -v /batchx/output:/batchx/output josemfer/batchx-tensorflow-gpu-demo:latest

# Import into Batchx
bx import josemfer/batchx-tensorflow-gpu-demo:latest

# Run in BX

## Copy training data into BatchX file system
bx cp fashion-mnist.npz bx://data/fashion-mnist.npz

## Run image
bx run -v=4 -m=15000 -g=1 -f=T4 tutorial/tensorflow-gpu-demo:0.0.2 '{"data_file_path": "bx://data/fashion_mnist.npz", "num_epochs": 4}'







