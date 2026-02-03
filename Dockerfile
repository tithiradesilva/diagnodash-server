# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies
# These libraries (libgl1, etc.) are often required by PyTorch/Vision image tools
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy dependency file and install Python libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your code files (model.py, server.py, etc.) into the container
COPY . .

# 6. Tell the cloud which port we are using
EXPOSE 5000

# 7. The command to start the server
CMD ["python", "server.py"]