# Setup


# Build the docker image
docker build -t my-llm-app .

# Run the docker image
docker run \
	--name my-llm-app-container -d \
	--mount type=bind,source=./models,target=/app/models,readonly \
	-p 8000:8000 \
	my-llm-app


# Message app running in docker container 

## Using local model
curl -s http://localhost:8000/llm -H "Content-Type: application/json" -d '{                 
  "model": "/app/models/Llama-3.2-1B-Instruct",
  "max_length": 256,
  "prompt": [
    {
      "role": "system",
      "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
    },
    {
      "role": "user",
      "content": "Compose a poem that explains the concept of recursion in programming."
    }
  ]
}'



### Resources

https://medium.com/@ahmedtm/a-simple-guide-to-run-the-llama-model-in-a-docker-container-a3899032995e