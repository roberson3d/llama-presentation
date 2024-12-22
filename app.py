from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import os

# Create a Flask object
app = Flask("LLM server")

# Declare the model and tokenizer variables
model = None
tokenizer = None

# Define the route for the index page
@app.route('/', methods=['GET', 'POST'])
def check_status():
	FILENAME = '/app/index.html'
	if os.path.exists(FILENAME):
		with open(FILENAME, 'r') as filestream:
			return filestream.read()
	return r'''<html><head><title>LLM server</title></head><body><p>No supplied info.</p></body></html>'''

# Define the route for the API
@app.route('/llm', methods=['POST'])
def generate_response():
	# Use the global variables so that the model and tokenizer are not reloaded for each request
	global model, tokenizer
	
	app.logger.info("Request received")

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

			app.logger.info("Response generated")
			app.logger.info(sequences)

			return jsonify([seq['generated_text'] for seq in sequences])

		else:
			return jsonify({"error": "Missing required parameters"}), 400

	except Exception as e:
		return jsonify({"Error": str(e)}), 500 

# when running the script as the main program
if __name__ == '__main__':
	# Run the Flask app in localhost:8000
	app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)