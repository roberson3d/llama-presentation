FROM python:3.9

# Install necessary packages
RUN pip install transformers torch huggingface_hub transformers Flask

# Set the working directory
WORKDIR /app

# Create a directory for the model
RUN mkdir /model

# Copy your application code (if any)
COPY app.py ./
COPY index.html ./

# Expose the port your app will run on
EXPOSE 8000

# Command to run your app
CMD ["python", "app.py"]