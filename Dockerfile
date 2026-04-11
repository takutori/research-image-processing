FROM python:3.11-slim

ARG TORCH_VARIANT=gpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_SYNC=1 \
    TORCH_VARIANT=${TORCH_VARIANT}

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./

RUN uv sync \
    && if [ "${TORCH_VARIANT}" = "gpu" ]; then \
        uv pip install --python /workspace/.venv/bin/python --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0; \
    else \
        uv pip install --python /workspace/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0; \
    fi

EXPOSE 8888

CMD ["bash", "-lc", "uv sync && if [ \"$TORCH_VARIANT\" = \"gpu\" ]; then uv pip install --python /workspace/.venv/bin/python --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0; else uv pip install --python /workspace/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0; fi && sleep infinity"]
