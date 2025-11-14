# Builder stage with GUI support
FROM debian:bookworm-slim AS builder

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables for build
ENV CC=/usr/bin/clang
ENV CXX=/usr/bin/clang++
ENV CMAKE_C_COMPILER=/usr/bin/clang
ENV CMAKE_CXX_COMPILER=/usr/bin/clang++

# Install build dependencies and XFCE for GUI support
RUN apt update && apt install -y \
    cmake \
    ninja-build \
    clang \
    build-essential \
    libwebkit2gtk-4.1-dev \
    wget \
    unzip \
    libssl-dev \
    pkg-config \
    git \
    pybind11-dev \
    python3-dev \
    xfce4 \
    libasound2-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxcomposite-dev \
    libfreetype6-dev \
    libfontconfig1-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
RUN mkdir -p /app/external
RUN mkdir -p /app/build
WORKDIR /app/external

# Download and extract VST3 SDK
RUN wget -O VST3_SDK.zip https://www.steinberg.net/vst3sdk
RUN unzip VST3_SDK.zip
RUN rm VST3_SDK.zip

# Copy source files
COPY external /app/external
COPY src /app/src
COPY CMakeLists.txt /app

# Set working directory to app root
WORKDIR /app

# Configure and build
RUN cmake -S . -B build -G Ninja
RUN cmake --build build --parallel $(nproc)

RUN ls -la /app/build/AdaptiveEcho_artefacts/Standalone

# Runtime stage
FROM debian:bookworm-slim AS runtime

# Install only runtime dependencies
RUN apt update && apt install -y \
    xfce4 \
    xfce4-terminal \
    xvfb \
    novnc \
    websockify \
    x11vnc \
    supervisor \
    libatomic1 \
    libgomp1 \
    libopenblas-dev \
    libjemalloc-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
RUN mkdir -p /app

ENV DISPLAY=:99
RUN mkdir -p /opt/novnc \
    && cp -r /usr/share/novnc/* /opt/novnc/
RUN mkdir -p /var/log/supervisor /var/run
RUN chown root:root /var/run

COPY supervisord.conf /root/supervisord.conf

# Copy built targets
COPY --from=builder /app/build/AdaptiveEcho_artefacts/Standalone/AdaptiveEcho /app
# Copy libtorch libraries
COPY --from=builder /app/build/_deps/libtorch-src/lib/*.so /app

# Make sure the plugin runs for 1 second without crashing
RUN timeout 1s /app/AdaptiveEcho

EXPOSE 6080

CMD ["supervisord", "-c", "/root/supervisord.conf"]