FROM python:3.13.2-bullseye

RUN apt update
RUN apt install -y graphviz

WORKDIR /app

COPY ./requirements.in .

RUN pip install --upgrade pip
RUN pip install pip-tools
RUN pip-compile ./requirements.in
RUN pip install -r ./requirements.txt

COPY . ./app