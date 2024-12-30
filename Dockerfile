# Import image with python environment
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

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
    cmake \
    git \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-openslide \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH=/opt/conda/bin:$PATH

# Set the working directory
WORKDIR /app

# Copy sources for requirements
COPY src/FACTORIZER ./src/FACTORIZER
COPY src/HD-BET ./src/HD-BET
COPY src/NVAUTO ./src/NVAUTO
COPY src/SEALS ./src/SEALS
COPY weights ./weights
COPY requirements.txt .

# Create a conda environment and install necessary packages
RUN conda create --name isles_ensemble python=3.8.0 pip=23.3.1 && \
    conda clean -afy

# Activate the environment and install packages
RUN /bin/bash -c "source activate isles_ensemble && \
    conda install -y pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch && \
    conda install -y -c conda-forge openslide-python && \
    conda install -y python=3.8.0 && \
    pip install --no-cache-dir -e ./src/SEALS/ && \
    pip install --no-cache-dir -e ./src/FACTORIZER/model/factorizer/ && \
    pip install --no-cache-dir -e ./src/HD-BET/ && \
    pip install --no-cache-dir -r requirements.txt"

# Copy the source code
COPY src/isles22_ensemble.py ./src/isles22_ensemble.py
COPY src/majority_voting.py ./src/majority_voting.py
COPY src/utils.py ./src/utils.py
COPY src/__init__.py ./src/__init__.py
COPY main.py .

# Run docker will start the main.py
ENTRYPOINT ["/bin/bash", "-c", "source activate isles_ensemble && python main.py \"$@\"", "--"]