FROM conda/miniconda3
MAINTAINER Chawye Hsu <h404bi@gmail.com>

ENV LANG=C.UTF-8
ENV TZ=UTC-8
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list
RUN apt-get update -q && \
    apt-get install -qy build-essential

COPY . /wende
WORKDIR /wende
RUN pip install --user pipenv

# Add user home bin to PATH
ENV PATH /root/.local/bin:$PATH

RUN pipenv install
