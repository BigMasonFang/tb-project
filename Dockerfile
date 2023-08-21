FROM ubuntu:22.04
USER root

# time and lang
# RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
ENV LANG C.UTF-8
# default pip proxy
ENV PIP_SOURCE ${PIP_SOURCE:-http://mirrors.cloud.tencent.com/pypi/simple}
ENV PIP_HOST ${PIP_HOST:-mirrors.cloud.tencent.com}

# add sources
ADD etc/sources.list /etc/apt

# add packages and softwars
RUN apt-get update -y
RUN apt-get install -y software-properties-common tar vim supervisor git
# RUN apt-get install -y nginx gdb

# python
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt install -y python3.10 pip python3-venv
# python venv and activate
ENV VIRTUAL_ENV /p_3_10
RUN python3.10 -m venv $VIRTUAL_ENV
ENV PATH "$VIRTUAL_ENV/bin/:$PATH"
# copy current files to docker
RUN mkdir /app
ADD . /app
WORKDIR /app
# python requirements
RUN pip install --upgrade pip --index-url=$PIP_SOURCE --trusted-host=$PIP_HOST
RUN pip install -r requirements.txt --index-url=$PIP_SOURCE --trusted-host=$PIP_HOST

RUN echo "finish env set, start jupyter lab"
# CMD jupyter lab --notebook-dir=/app --no-browser --ip=0.0.0.0 --port=5000 --allow-root
CMD bash start.sh
