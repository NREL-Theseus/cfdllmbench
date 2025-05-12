# Start from OpenFOAM + ParaView image
FROM openfoam/openfoam10-paraview510:latest

# Switch to root
USER root

# Install extra packages
RUN apt-get update && apt-get install -y \
    vim \
    git \
    unzip \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy everything into the container
COPY . /home/foam/FoamBench/

# Change working directory
WORKDIR /home/foam/FoamBench/

# Restore to OpenFOAM user
#USER foam

# Optional: Run benchmark script automatically
# CMD ["python3", "run_benchmarks"]

