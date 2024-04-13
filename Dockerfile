# ARG base_image

FROM 10.119.12.14:5000/llm/llm-gpu

COPY ./applications/DeepSpeed-Chat/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY ./DeepSpeed /DeepSpeed
WORKDIR /DeepSpeed
RUN python3 setup.py bdist_wheel
# RUN pip3 uninstall -y deepspeed && pip3 install -e /DeepSpeed
# COPY ./DeepSpeed/dist/deepspeed-0.12.6+unknown-py3-none-any.whl /deepspeed-0.12.6+unknown-py3-none-any.whl
RUN pip3 uninstall -y deepspeed && pip3 install ./dist/*.whl
