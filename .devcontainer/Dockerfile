FROM debian:stable-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo git git-lfs g++ gcc make cmake pkg-config gdb \
    libopencv-dev python3 python3-pip python3-opencv \
    x11-apps \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m vscode \
    && echo "vscode ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/vscode \
    && chmod 0440 /etc/sudoers.d/vscode

ENTRYPOINT ["/bin/bash"]
