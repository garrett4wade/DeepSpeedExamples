# ARG base_image

FROM nvcr.io/nvidia/pytorch:23.06-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt update
RUN apt install -y ca-certificates
RUN sed -i "s@http://.*archive.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN sed -i "s@http://.*security.ubuntu.com@https://mirrors.tuna.tsinghua.edu.cn@g" /etc/apt/sources.list
RUN apt update
# RUN apt install -y --fix-missing net-tools python3-opencv ffmpeg libsm6 libxext6 python3-pip pkg-config libopenblas-base libopenmpi-dev

RUN pip3 install -U pip
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install torch
RUN pip3 install ray[default,serve,data]
RUN pip3 install transformers huggingface_hub datasets accelerate bitsandbytes
RUN pip3 install deepspeed

RUN pip3 install blosc cmake prometheus_client wandb redis scipy h5py ninja packaging tensorboardx

RUN pip3 install flash-attn --no-build-isolation

ENV PATH="${PATH}:/opt/hpcx/ompi/bin:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib:/opt/hpcx/ucx/lib/"

# install zmq with pgm
# COPY ./zeromq-4.3.4.tar.gz /
# WORKDIR /
# RUN tar zxvf zeromq-4.3.4.tar.gz
# WORKDIR /zeromq-4.3.4
# RUN ./configure --with-pgm && make && make install
# WORKDIR /
# RUN pip3 install -I --no-binary=:all: pyzmq