FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# The X and GL libraries the bpy wheel links, even headless.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 libsm6 libx11-6 libxfixes3 libxi6 libxkbcommon0 libxrender1 libxxf86vm1 \
    && rm -rf /var/lib/apt/lists/*

# graphics mounts libnvoptix and libnvidia-rtcore; without them OptiX is far slower.
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /seascape
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8765
CMD ["uv", "run", "--no-sync", "seascape", "serve", "--host", "0.0.0.0"]
