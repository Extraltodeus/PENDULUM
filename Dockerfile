# Pendulum in a container: llama.cpp built with CUDA, the Qt window shown through X11 (Linux or WSLg).
# docker compose up --build   (see docs/GUIDE.md, "Docker")

ARG CUDA_VERSION=12.8.1
ARG UBUNTU=24.04

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU} AS llama
ARG LLAMA_REF=master
ARG CUDA_ARCHS="75;80;86;89;90;120"
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_REF} https://github.com/ggml-org/llama.cpp /src
WORKDIR /src
RUN cmake -B build -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
	&& cmake --build build -j"$(nproc)"
RUN mkdir -p /out/build \
	&& tar --exclude=./build --exclude=./.git -cf - . | tar -xf - -C /out \
	&& cp -r build/bin /out/build/bin

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU}
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV DEBIAN_FRONTEND=noninteractive \
	NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
	PENDULUM_LLAMA=/opt/llama.cpp/build/bin \
	PENDULUM_DATA_DIR=/data \
	HF_HOME=/hf \
	QT_X11_NO_MITSHM=1 \
	PATH=/venv/bin:/opt/llama.cpp/build/bin:$PATH \
	LD_LIBRARY_PATH=/opt/llama.cpp/build/bin:$LD_LIBRARY_PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
		python3 python3-venv python3-dev git ca-certificates \
		libgl1 libegl1 libopengl0 libglib2.0-0 libdbus-1-3 libfontconfig1 libfreetype6 fonts-dejavu-core \
		libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
		libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xfixes0 libxcb-xkb1 libx11-xcb1 libxrender1 libxi6 \
		libwayland-client0 libwayland-egl1 libwayland-cursor0 \
	&& rm -rf /var/lib/apt/lists/*
COPY --from=llama /out /opt/llama.cpp
RUN python3 -m venv /venv && pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch --index-url ${TORCH_INDEX}
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
VOLUME ["/data", "/hf", "/models"]
CMD ["python", "-X", "utf8", "Pendulum.pyw"]
