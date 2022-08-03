FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV PROJECT=/point_dg
WORKDIR $PROJECT
RUN mkdir -p  $PROJECT/logs && mkdir -p $PROJECT/workspace/ && mkdir $PROJECT/data

# Copy the repo file to docker
COPY . $PROJECT/workspace
RUN pip install -r requirements.txt

# RUN python3 -m pip install --editable .
# Complie the project environment if needed
# RUN rm -rf $PROJECT

# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user