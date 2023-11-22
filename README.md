# ECE9143 (Advanced Robotic AI) Project
### Title
Position control and path planning of 2D drone loaded double pendulum via reinforcement learning

![fig1](./figure/fig1.png)  

## Install instructions
Strongly recommend install via Docker. 

Implemented simulation setup as a virtual environment to run the code immediately without additional installation. Since the code is based on docker, you can download the execution environment through my Docker hub. 

### Installation via Docker

    1. Download and install Docker for your OS. 
    2. Pull the image via Docker hub
        docker pull latteishorse/ece9143:latest
    3. Create a container with the Docker image
        docker run -it --name drone latteishorse/ece9143:latest

### Attach Docker container to VS code

The environment is set up to run the simulation described earlier in the simulation setup. You can use this simulation through the terminal, but here is how to attach docker to Visual Studio Code for convenience.

    1. Download and install Visual Studio Code.
    2. Install VS Code extension ’Docker’ and Microsoft ’Remote Development’.
    3. Click the ’Docker’ sidebar and choose ’drone’ container.
    4. Right-click the mouse on ’drone’ container, and choose ’Attach to a running container’.

After the above four steps, you can easily handle code and also easily control the docker container.

### Dataset download
Dataset (Model) already included in the "examples" folder. (.zip format)

## How to run
### Train
    python train.py

### Evaluate
    python eval.py

