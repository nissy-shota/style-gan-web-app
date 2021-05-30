FROM ubuntu:20.04

RUN apt update && apt upgrade -y && apt install -y \
    sudo \
    python3.8 \
    python3-pip \
    libgl1-mesa-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

WORKDIR /workdir
COPY requirements.txt /workdir

RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]