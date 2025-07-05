FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for matplotlib, plotly, and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the main script
COPY oulad_synthetic_analysis.py .

# Create directories for outputs
RUN mkdir -p clean results synthetic

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg
ENV OULAD_ROOT=/app/OULAD_data

# Default command
CMD ["python", "oulad_synthetic_analysis.py"] 