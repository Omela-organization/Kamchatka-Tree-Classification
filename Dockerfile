ARG BASE_IMAGE=python:3.10
FROM $BASE_IMAGE

# system update & package install
RUN apt-get -y update && apt-get install libpq-dev -y

RUN mkdir /app
WORKDIR /app

COPY requirements.txt .

# pip & requirements
RUN python3 -m pip install --user --upgrade pip && \
    python3 -m pip install -r requirements.txt

COPY . .
