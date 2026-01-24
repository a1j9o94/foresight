FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

# Copy and install dependencies
COPY backend/requirements.txt .
RUN uv pip install --system -r requirements.txt

# Copy the backend code
COPY backend /app/demo/backend

# Create empty __init__.py for the demo package
RUN touch /app/demo/__init__.py

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "demo.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
