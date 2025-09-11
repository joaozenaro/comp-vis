FROM debian:stable-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs g++ gcc make cmake pkg-config gdb \
    libopencv-dev python3 python3-pip python3-opencv \
    x11-apps \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["sleep", "infinity"]
