FROM python:3.11

WORKDIR /code
ENV PYTHONPATH "${PYTHONPATH}:/code"

RUN apt-get update -y && apt-get install zsh -y
RUN PATH="$PATH:/usr/bin/zsh"

RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade build
RUN python -m pip install --upgrade twine

# https://stackoverflow.com/questions/66017188/how-to-resolve-the-error-failed-building-wheel-for-h5py
RUN pip install --upgrade pip setuptools wheel

# https://stackoverflow.com/questions/78359706/docker-build-fails-for-h5py-in-python-3-9
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install --no-binary h5py h5py

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt
