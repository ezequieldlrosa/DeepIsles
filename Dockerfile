# Import image with python environment
# CUDA version can be specified via build argument: --build-arg CUDA_VERSION=12
# Default base image: CUDA 11.8
# For CUDA 12: Change FROM to nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04
# For CUDA 13: Change FROM to nvidia/cuda:12.4.0-cudnn8-runtime-ubuntu22.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Disable interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install some basic libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bash \
    wget \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy sources and dependency files
COPY src/FACTORIZER ./src/FACTORIZER
COPY src/HD-BET ./src/HD-BET
COPY src/NVAUTO ./src/NVAUTO
COPY src/SEALS ./src/SEALS
COPY weights ./weights
COPY pyproject.toml .

# Create a conda environment and install necessary packages
RUN conda create --name deepisles python=3.12 pip -y && \
    conda clean -afy

# Activate the environment and install packages
# Map CUDA version to PyTorch index (default: 11.8)
ARG CUDA_VERSION=11.8
RUN /bin/bash -c "source activate deepisles && \
    pip install uv && \
    if [ \"$CUDA_VERSION\" = \"12\" ]; then \
        PYTORCH_CUDA=\"cu121\"; \
    elif [ \"$CUDA_VERSION\" = \"13\" ]; then \
        PYTORCH_CUDA=\"cu131\"; \
    else \
        PYTORCH_CUDA=\"cu118\"; \
    fi && \
    echo \"Installing PyTorch with CUDA $CUDA_VERSION (index: \$PYTORCH_CUDA)\" && \
    uv pip install --no-cache --extra-index-url https://download.pytorch.org/whl/\$PYTORCH_CUDA \
        torch>=2.1.2 torchvision torchaudio && \
    uv pip install --no-cache -e ."

# Copy the source code
COPY src/isles22_ensemble.py ./src/isles22_ensemble.py
COPY src/majority_voting.py ./src/majority_voting.py
COPY src/utils.py ./src/utils.py
COPY src/__init__.py ./src/__init__.py
COPY main.py .

# Set PYTHONPATH for local modules (SEALS, FACTORIZER, HD-BET)
ENV PYTHONPATH="${PYTHONPATH}:/app/src/SEALS:/app/src/FACTORIZER/model/factorizer:/app/src/HD-BET"

# Run docker will start the main.py
ENTRYPOINT ["/bin/bash", "-c", "source activate deepisles && python main.py \"$@\"", "--"]
