FROM python:3.7-slim-buster AS compile-image

WORKDIR /install

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
  pkg-config \
  libavformat-dev \
  libavcodec-dev \
  libavdevice-dev \
  libavutil-dev \
  libavfilter-dev \
  libswscale-dev \
  libswresample-dev \
  python3-dev \
  gcc \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user -r requirements.txt
RUN pip install --user https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_x86_64.whl

COPY src src/
COPY setup.py .
RUN pip install --user .

######

FROM python:3.7-slim-buster AS build-image

RUN apt-get update \
  && apt-get install --no-install-recommends -y \
  pkg-config \
  libavformat58 \
  libavcodec58 \
  libavdevice58 \
  libavutil56 \
  libavfilter7 \
  libswscale5 \
  libswresample3 \
  gnupg \
  vim \
  wget \
  tcpdump \
  net-tools \
  && rm -rf /var/lib/apt/lists/*

# Run separately as gnupg is required from previous apt-get install entry
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > \
  /etc/apt/sources.list.d/coral-edgetpu.list \
  && wget -q -O - https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update \
  && apt-get install --no-install-recommends -y libedgetpu1-std \
  && rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /root/.local /root/.local

ENV PATH=/root/.local/bin:$PATH

WORKDIR /conf

CMD ["visionalert"]
