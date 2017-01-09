FROM ubuntu:16.04

RUN     apt-get update && \
        apt-get upgrade -y && \
        apt-get autoremove

RUN     apt-get install -y unzip wget build-essential vim \
		cmake git python3-dev python3-numpy python3-scipy python3-pip \
		python3-setuptools && \
	    apt-get autoremove

RUN pip3 install --upgrade pip

RUN pip3 install sklearn pandas keras pillow

# Tensorflow
RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0-cp35-cp35m-linux_x86_64.whl

EXPOSE 6006

# Make a development directory
WORKDIR /development
