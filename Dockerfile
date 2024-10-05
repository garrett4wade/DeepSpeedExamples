# ARG base_image

FROM nvcr.io/nvidia/pytorch:24.07-py3

COPY ./DeepSpeedExamples/applications/DeepSpeed-Chat/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./DeepSpeed /DeepSpeed
WORKDIR /DeepSpeed
RUN python3 setup.py bdist_wheel
# RUN pip3 uninstall -y deepspeed && pip3 install -e /DeepSpeed
# COPY ./DeepSpeed/dist/deepspeed-0.12.6+unknown-py3-none-any.whl /deepspeed-0.12.6+unknown-py3-none-any.whl
RUN pip3 uninstall -y deepspeed && pip3 install ./dist/*.whl

RUN pip3 install ray[default] -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install transformers==4.34.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install -U grpcio -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY ./DeepSpeedExamples /DeepSpeedExamples
RUN pip3 install -e /DeepSpeedExamples/applications/DeepSpeed-Chat -i https://pypi.tuna.tsinghua.edu.cn/simple
WORKDIR /DeepSpeedExamples

RUN pip3 install accelerate==0.31.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
