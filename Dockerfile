FROM python:3.9.5-slim-buster
LABEL maintainer="koreander2001 <neokamiyama@gmail.com>"

RUN apt-get update && \
    apt-get install -y git less vim

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
