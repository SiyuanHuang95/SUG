FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT=/point_dg
WORKDIR $PROJECT
RUN mkdir -p  $PROJECT/logs && mkdir -p $PROJECT/workspace/ && mkdir $PROJECT/data

# Update and install various packages
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget curl git build-essential vim sudo lsb-release locales bash-completion tzdata

# Copy the repo file to docker
COPY . $PROJECT/workspace
# Recommand to pip install after docker built
# RUN cd $PROJECT/workspace && pip install -r requirements.txt && conda install -n base ipykernel --update-deps --force-reinstall

# RUN python3 -m pip install --editable .
# Complie the project environment if needed
# RUN rm -rf $PROJECT

# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user