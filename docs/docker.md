### Docker Setup

All of the examples in this repo provide instructions for being run normally, and also using Docker. Although not required, the Docker option is recommended as this will simplify software installation, produce a more stable environment, and also allow you to avoid installing ROS and cluttering your machine, for examples which depend on ROS.

- Basic Docker installation instructions are here.
- Docker must be installed in rootless mode.
- To use your GPU you must also install the NVIDIA container toolkit.
- The version of docker installed with `snap` is incompatible with the NVIDIA container toolkit, preventing it from accessing `libnvidia-ml.so` (issue). The snap version can be uninstalled with `sudo snap remove docker`.
- Docker Desktop is also incompatible with the NVIDIA runtime (issue). Docker Desktop can be uninstalled with `sudo apt remove docker-desktop`.

If starting from scratch and your host machine is Ubuntu 22.04, you can use accomplish all of the above with the convenience scripts `scripts/docker/install_docker_ubuntu22.sh` and `scripts/docker/install_nvidia_container_toolkit.sh`.

Build the Docker image and start the container with the following command:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

To build and run the Docker image for a specific example, use the following command:
```bash
docker compose -f examples/<example_name>/compose.yml up --build
```
where `<example_name>` is the name of the example you want to run.

During the first run of any example, Docker will build the images. Go grab a coffee while this happens. Subsequent runs will be faster since the images are cached.