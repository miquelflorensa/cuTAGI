FROM nvidia/cuda:11.4.2-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive # ignore user input required

# Install essential packages
RUN apt-get -y update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    autoconf \
    automake \
    libtool \
    pkg-config \
    apt-transport-https \
    ca-certificates \
    software-properties-common \
    g++ \
    git \
    wget \
    gdb \
    valgrind \
    locales \
    locales-all &&\
    apt-get clean 

# Install cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | apt-key add - &&\
    apt-add-repository "deb https://apt.kitware.com/ubuntu/ bionic main" &&\
    apt-get update &&\
    apt-get install -y cmake

# Directory in docker images that stores cutagi's code
ARG WDC=/usr/src/cutagi

# Create environement variable to pass the cfg file. NOTE: We should expolore the entry point
ENV NAME VAR1

# Copy code from the host device to docker images. Note that we do not copy 
# the data and cfg to docker images b/c we want to keep them on the host device
COPY src/ ${WDC}/src
COPY include/ ${WDC}/include
COPY CMakeLists.txt ${WDC}/CMakeLists.txt
COPY Dockerfile ${WDC}/Dockerfile
COPY main.cpp ${WDC}/main.cpp
COPY main.cu ${WDC}/main.cu
COPY README.md ${WDC}/README.md
COPY RELEASES.md ${WDC}/RELEASES.md
COPY visualizer.py ${WDC}/visualizer.py
COPY docker_main.sh ${WDC}/docker_main.sh

# Work directory for the docker image
WORKDIR ${WDC}

# Run cmake to compile the code
RUN mkdir -p ./build 
RUN cmake -B/build -S .
RUN cmake --build /build

CMD ["/bin/bash","docker_main.sh"]