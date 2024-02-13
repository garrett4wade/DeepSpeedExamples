# ARG base_image

FROM 10.119.12.14:5000/llm/llm-gpu

COPY ./applications/DeepSpeed-Chat/requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
