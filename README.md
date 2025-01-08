# Setup

## Install Docker

Docker Desktop app is available for Windows, Mac, and Linux: https://docs.docker.com/desktop/. The application includes the CLI.


## Build the docker image
```sh
docker build -t my-llm-app .
```

## Run the docker image
```sh
docker run \
	--name my-llm-app-container -d \
	--mount type=bind,source=./models,target=/app/models,readonly \
	-p 8000:8000 \
	my-llm-app
```


# Message app running in docker container 

## Using local model
```sh
curl -X POST -s http://localhost:8000/llm -H "Content-Type: application/json" -d '{                 
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
```

### Using HF model
```sh
curl -X POST -s http://localhost:8000/llm -H "Content-Type: application/json" -d '{                 
  "model": "meta-llama/Llama-3.2-1B",
  ...  same as before ...
}'
```



# Resources

#### Meta LLaMA
https://www.llama.com/

https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/

#### Ollama
https://ollama.com/

https://hub.docker.com/r/ollama/ollama

https://github.com/ollama/ollama?tab=readme-ov-file

https://xethub.com/blog/comparing-code-llama-models-locally-macbook

#### Hugging Face

https://huggingface.co/meta-llama

https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Running_Llama3_Anywhere/Running_Llama_on_HF_transformers.ipynb

#### Docker

https://docs.docker.com/build/concepts/dockerfile/

https://medium.com/@ahmedtm/a-simple-guide-to-run-the-llama-model-in-a-docker-container-a3899032995e

