# Use the official Meta PyTorch OpenEnv base image for maximum compatibility
FROM ghcr.io/meta-pytorch/openenv-base:latest

# Set the working directory to /app as per OpenEnv standards
WORKDIR /app

# Install 'uv' for high-performance dependency management and build speed
RUN pip install --no-cache-dir uv

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies into the system python environment
RUN uv pip install --system --no-cache -r requirements.txt

# Copy the entire project code into the container
COPY . .

# EXTREMELY IMPORTANT: Expose port 7860 for Hugging Face Spaces
EXPOSE 7860

# Metadata as demanded by OpenEnv
LABEL org.opencontainers.image.source="https://github.com/your-username/NegotiationRL-Pro"
LABEL org.opencontainers.image.description="NegotiationRL-Pro for Meta PyTorch Hackathon"

# Run the FastAPI server on launch
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
