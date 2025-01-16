# Running Llama as a Private Instance  
### by Ricky Roberson



## Why run a private instance

* Data Security & Privacy
* Customization and Fine-Tuning
* Offline Deployment



## Llama

##### Large Language Model Meta AI

* Developed by Meta
* Open source LLM families
* Free for research and some commerial use (non-competitive, < 700m MAU)
* Multiple varients including number of tokens, quantitization, instruct versions, and more


### Accessing Models

* Models can be retrieved directly from Meta, HuggingFace, or Kaggle
* Models require license agreements available with each distributor
* Options to run models on Linux, Mac, or Windows
* Documentation and instructions on [Llama.com](http://www.llama.com)



## HuggingFace + Colab

#### Google Colab

* Write and execute Python
* Zero configuration required
* Access to GPUs free of charge
* Easy sharing

#### HuggingFace

* Github for machine learning


#### Install Dependencies & HuggingFace Login

```python [|1-6|8]
!pip install -U "huggingface_hub[cli]"
!pip install transformers
!pip install accelerate

from google.colab import userdata
from huggingface_hub import login

login(userdata.get('HF_TOKEN'))
```

Ensure dependencies are installed, including ability to log in to HuggingFace. Using the Colab notebook secrets to store the app token.
* [Transformers](https://huggingface.co/docs/transformers/index) is HuggingFace's API that gives easy access to their database of pretained models for downloading and training. It also supports interoperability between PyTorch, TensorFlow, and JAX which are a few of the big frameworks for worrking with LLMs.
* [Accelerate](https://huggingface.co/docs/accelerate/en/index) is a library for running PyTorch code in distributed configurations (like multiple GPUs).


#### Setup Transformers Pipline

```python [|5]
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

Designate the model to use. This will have to be downloaded and built on the first use, but will be available for future queries from that point.
I am using the model id of a predefined tokenizer hosted inside a model repo on HuggingFace, but directory paths also work if the model is already local.


#### Submitting prompts

```python [|1-4|16-17]
base_prompt = "<s>[INST]\n<<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt}[/INST]"

input = base_prompt.format(system_prompt = "Answer the following programming questions as a knowledgable engineer.",
                           user_prompt = "What casting options are there in c++?")
    
sequences = pipeline(
    input,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    truncation = True,
    max_length=400,
)

for seq in sequences:
    print(f"Result: {seq['generated_text']}")
```

Build a prompt. Depending on the model used, the format may have different forms like just the prompt string to expand from or combining system instructions and user queries. AI responses are generated and the result is printed for each generated sequence.


#### Thoughts on Colab

* Really nice tool for exploring python related development.
* Long running tasks are difficult to maintain at the free tier.
* Not a good option for one off uses as it will require re-downloading and building tokens.


## Running Locally in Docker

* Containerization that allows sharing and running "anywhere".
* Builds environment via script in a consistent, predictable, and reproducable manner.
* Simulate running the application in a virtual machine or server.
* Open source



### Dockerfile
```dockerfile [1|3-4|6-7|9-10|12-13|15-16|18-19]
FROM python:3.9

# Install necessary packages
RUN pip install transformers torch huggingface_hub Flask

# Set the working directory
WORKDIR /app

# Copy the application code
COPY app.py ./

# Create a directory for the model
RUN mkdir /model

# Expose the port the app will listen to
EXPOSE 8000

# Command to run our app
CMD ["python", "app.py"]
```


### Working with the Docker image

```sh
docker build -t my-llm-app .
```

Creates the container based on the dockerfile.

```sh
docker run \
	--name my-llm-app-container -d \
	--mount type=bind,source=./models,target=/app/models,readonly \
	-p 8000:8000 \
	my-llm-app
```

Runs the container.
The `localhost:8000` port is connected to the containers `8000` port so we can send messages to our app running there.
The local `./models` directory is also mounted to the to the container's `/app/models` directory so we can access the models we have downloaded and don't have to include them in the repo. 


### App running in docker container 

```python [|1,5-6,12-13|31-32,40-56|70-71]
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Create a Flask object
app = Flask("LLM server")

# Declare the model and tokenizer variables
model = None
tokenizer = None

# Define the route for the API
@app.route('/llm', methods=['POST'])
def generate_response():
	# Use the global variables so that the model and tokenizer are not reloaded for each request
	global model, tokenizer
	
	print("Request received")

	try:
		# Get the JSON data from the flask request
		data = request.get_json()
		
		# Create the model and tokenizer if they were not previously created
		if model is None or tokenizer is None:
			# Get the location of to the desired model here.
			# This can be a local path or a URL to a Hugging Face model
			model_dir = data['model']

			# Create the model and tokenizer
			tokenizer = AutoTokenizer.from_pretrained(model_dir)
			model = AutoModelForCausalLM.from_pretrained(model_dir)

		# Check if the required fields are present in the JSON data
		if 'prompt' in data and 'max_length' in data:
			prompt = data['prompt']
			max_length = int(data['max_length'])
			
			# Create the pipeline
			text_gen = pipeline(
				"text-generation",
				model=model,
				tokenizer=tokenizer,
				torch_dtype=torch.float16,
				device_map="auto",)
			
			# Run the model
			sequences = text_gen(
				prompt,
				do_sample=True,
				top_k=10,
				num_return_sequences=1,
				eos_token_id=tokenizer.eos_token_id,
				max_length=max_length,
			)

			print("Response generated")
			print(sequences)

			return jsonify([seq['generated_text'] for seq in sequences])

		else:
			return jsonify({"error": "Missing required parameters"}), 400

	except Exception as e:
		return jsonify({"Error": str(e)}), 500 

# when running the script as the main program
if __name__ == '__main__':
	# Run the Flask app in localhost:8000
	app.run(host='0.0.0.0', port=8000, debug=True)
```

Small application that is similar to the Colab script, but accessable via REST API at local host port 8000 via a Flask app. 


### Messaging the Docker container's app 

```sh [2|4-13|6,10|7,11]
curl -x POST -s http://localhost:8000/llm -H "Content-Type: application/json" --data-binary '@curl.json'
```

cURL post command sent to our container app that uses a file for the payload as it's easier to adjust and read.

```
{                 
  "model": "/app/models/CodeLlama-7b-Instruct-hf",
  "max_length": 256,
  "prompt": [
    {
      "role": "system",
      "content": "Answer the following programming questions as a knowledgable engineer."
    },
    {
      "role": "user",
      "content": "What casting options are there in c++?"
    }
  ]
}
```

Example `curl.json` File

Send a curl resquest to the app running in docker over the exposed port. 


#### Add a better interface for messaging

* Small python script `chat.py` that makes it easier to do "continuous conversations" with the AI. This is achieved by refeeding back in the series of prompt and responses as a sequence of messages.
* Several other front end UIs exist for this sort of thing, but I haven't dug into good options.

<!--
```
import http.client
import json
import os
import argparse

def send_message_to_llm(message: str, model: str) -> dict:
	"""
	Sends a POST request with the given message as the payload
	using http.client and waits for a JSON response.

	Args:
		message (str): The message to send to the API.
		model (str): The model to use for the API.

	Returns:
		dict: The JSON response of the generated text from the LLM.
	"""
	api_host = "localhost:8000"
	api_path = "/llm"
	
	# Prepare the connection and headers
	connection = http.client.HTTPConnection(api_host)
	headers = {"Content-Type": "application/json"}
	
	# Load the history from a file
	files = ['./data/prompt.json', 'data/starting-prompt.json']
	for filename in files:
		if os.path.exists(filename):
			with open(filename, 'r') as filestream:
				history = json.load(filestream)

	history.append({
		"role": "user",
		"content": message
	})

	# Save the history to a file
	with open('./data/prompt.json', 'w') as filestream:
		json.dump(history, filestream)

	payload = json.dumps({
		"model": model,
		"max_length": 256,
		"prompt": history
	})

	try:
		# Send POST request
		connection.request("POST", api_path, payload, headers)
		response = connection.getresponse()
		
		# Read and decode the response
		if response.status == 200:
			response_data = response.read().decode("utf-8")
			return json.loads(response_data)
		else:
			print(f"Error: Received status code {response.status}")
			return None
	except Exception as e:
		print(f"An error occurred: {e}")
		return None
	finally:
		connection.close()

def main() -> None:
	"""
	Main function to run the script.
	"""
	parser = argparse.ArgumentParser(description="Chat with an LLM.")
	parser.add_argument('message', type=str, help='The message to send to the LLM')
	parser.add_argument('--model', type=str, default='/app/models/Llama-3.2-1B-Instruct', help='The model to use for the LLM')

	args = parser.parse_args()

	message = args.message
	model = args.model

	# Now you can use `message` and `model` in your script
	print(f"Message: {message}")
	print(f"Model: {model}")

	response = send_message_to_llm(message, model)
	if response:
		print("Response from API:\n", response[-1][-1]["content"])

		with open('./response.json', 'w') as filestream:
			json.dump(response, filestream)

if __name__ == "__main__":
	main()

```
-->



## Ollama

##### Create, run, and share large language models

* Open source platform for for downloading, running, and working with LLMs
* Meta suggested tool for Mac, also available for Windows & Linux


###


### Resources

| | |
|-|-|
| **Meta LLaMA** | [llama.com](https://www.llama.com/) |
| **Ollama** | [ollama.com](https://ollama.com/),<br>[ollama docker](https://hub.docker.com/r/ollama/ollama),<br>[ollama github](https://github.com/ollama/ollama),<br>[something](https://xethub.com/blog/comparing-code-llama-models-locally-macbook) |
| **Hugging Face** | [huggingface.co](https://huggingface.co/meta-llama),<br>[llama + transformers recipe](https://github.com/meta-llama/llama-recipes/blob/main/recipes/quickstart/Running_Llama3_Anywhere/Running_Llama_on_HF_transformers.ipynb) |
| **Docker** | [docker.com](https://docs.docker.com/build/concepts/dockerfile/), <br>[guide to llama in docker](https://medium.com/@ahmedtm/a-simple-guide-to-run-the-llama-model-in-a-docker-container-a3899032995e) |
|Others|[flask](https://flask.palletsprojects.com/en/stable/)<br>|
|||

https://www.linkedin.com/pulse/running-llama-model-ollama-docker-container-simple-guide-islam-bxedc/
https://collabnix.com/getting-started-with-ollama-and-docker/
https://github.com/meta-llama/llama-models
https://medium.com/bluetuple-ai/how-to-enable-gpu-support-for-tensorflow-or-pytorch-on-macos-4aaaad057e74