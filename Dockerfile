FROM python:3.11-slim

WORKDIR /app

# Install git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install vectorbtpro from the GitHub repo
RUN pip install --no-cache-dir git+https://github.com/aispiringcoder4302/quant-analytics-lib.git

# Test import
RUN python -c "import vectorbtpro as vbt; print(f'VectorBT Pro version: {vbt.__version__}')"

# Keep container running for testing
CMD ["python", "-c", "import vectorbtpro as vbt; print('Import successful'); print(f'Version: {vbt.__version__}')"]
