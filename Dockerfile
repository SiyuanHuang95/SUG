FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

ENV PROJECT=/point_dg
RUN mkdir -p $PROJECT
RUN mkdir -p /point_dg/logs

ENV DATA=/point_dg/data
RUN mkdir -p $DATA

# Install project related dependencies
WORKDIR $PROJECT
COPY . $PROJECT
# Copy the repo file to docker
RUN python3 -m pip install --editable .
RUN rm -rf $PROJECT

# Add user to share files between container and host system
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user