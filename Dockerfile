# syntax=docker/dockerfile:1

FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG OPENEB_VERSION=3.1.2

ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    googletest \
    libboost-all-dev \
    libcanberra-gtk-module \
    libglew-dev \
    libglfw3-dev \
    libgl1-mesa-dri \
    libgl1-mesa-glx \
    libgtest-dev \
    libopencv-dev \
    libusb-1.0-0-dev \
    mesa-utils \
    pkg-config \
    qml-module-qt-labs-platform \
    qml-module-qtgraphicaleffects \
    qml-module-qtquick-controls2 \
    qml-module-qtquick-layouts \
    qml-module-qtquick2 \
    qtbase5-dev \
    qtdeclarative5-dev \
    qtquickcontrols2-5-dev \
    software-properties-common \
    unzip \
    wget \
    xauth \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 --branch "${OPENEB_VERSION}" https://github.com/prophesee-ai/openeb.git \
    && cmake -S /opt/openeb -B /opt/openeb/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DCOMPILE_PYTHON3_BINDINGS=OFF \
    && cmake --build /opt/openeb/build --config Release -- -j"$(nproc)" \
    && cmake --build /opt/openeb/build --target install \
    && ldconfig \
    && rm -rf /opt/openeb

WORKDIR /app

COPY src/ /app/src/

RUN cmake -S /app/src -B /app/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /app/build --config Release -- -j"$(nproc)"

COPY docker/entrypoint.sh /usr/local/bin/e-bts-entrypoint
RUN chmod +x /usr/local/bin/e-bts-entrypoint \
    && mkdir -p /data/recordings

ENV E_BTS_RECORDINGS_DIR=/data/recordings
WORKDIR /data/recordings

ENTRYPOINT ["/usr/local/bin/e-bts-entrypoint"]
CMD ["viewer"]
