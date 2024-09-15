#!/bin/bash

echo "Running API"

image_name="processor"

if ! docker images --format '{{.Repository}}' | grep -q "$image_name"; then
    echo "Image $image_name doesn't exist, building image..."
    docker build -t processor .
else
    echo "image $image_name exists, skipping build"
fi

# docker run -d --runtime=nvidia --gpus all  --name bev -p 8000:8000 processor
docker run --name img -p 8000:8000 processor
