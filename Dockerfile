FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT=/point_dg
WORKDIR $PROJECT
RUN mkdir -p  $PROJECT/logs && mkdir -p $PROJECT/workspace/ && mkdir $PROJECT/data

RUN apt-get update -y  ; exit 0
# use exit 0 to avoid errors from nvidia-gpg which could cause exit
RUN apt-get install curl -y  && apt-get install -y --no-install-recommends \
       build-essential libopenblas-dev git python3-pip  && apt-get install ffmpeg libsm6 libxext6  -y

# Copy the repo file to docker
COPY . $PROJECT/workspace 
RUN cd $PROJECT/workspace && pip install -r requirements.txt \ 
        && conda install -n base ipykernel --update-deps --force-reinstall
# Recommand to pip install after docker built

# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user
