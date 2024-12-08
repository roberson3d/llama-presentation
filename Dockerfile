FROM python:3.9

# Install necessary packages
RUN pip install transformers torch

# Set the working directory
WORKDIR /app

# Create a directory for the model
RUN mkdir /model

# Copy your application code (if any)
COPY app.py ./

# Install the needed packages
#RUN apt-get update && apt-get install -y gcc g++ procps
#RUN pip install huggingface_hub transformers Flask llama-cpp-python torch tensorflow flax sentencepiece nvidia-pyindex nvidia-tensorrt accelerate
RUN pip install huggingface_hub transformers Flask

# Expose the port your app will run on
EXPOSE 8000

# Command to run your app
CMD ["python", "app.py"]