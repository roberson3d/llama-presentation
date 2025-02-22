import http.client
import json
import os
import argparse

CHAT_FILE = "data/prompt.json"
CHAT_FILE_START = "data/starting-prompt.json"

def send_message_to_llm(message: str, model: str, length: int) -> dict:
	"""
	Sends a POST request with the given message as the payload
	using http.client and waits for a JSON response.

	Args:
		message (str): The message to send to the API.
		model (str): The model to use for the API.

	Returns:
		dict: The JSON response of the generated text from the LLM.
	"""
	API_HOST = "localhost"
	API_PORT = 8000
	API_METH = 'POST'
	API_PATH = "/llm"
	HEADER = {"Content-Type": "application/json"}
	
	# Prepare the connection and headers
	connection = http.client.HTTPConnection(API_HOST, API_PORT)
	
	# Load the history from a file
	files = [CHAT_FILE, CHAT_FILE_START]
	for filename in files:
		if os.path.exists(filename):
			with open(filename, 'r') as filestream:
				history = json.load(filestream)
				break

	history.append({
		"role": "user",
		"content": message
	})

	# Save the history to a file
	with open(CHAT_FILE, 'w') as filestream:
		json.dump(history, filestream, indent=4)
	
	payload = json.dumps({
		"model": model,
		"max_length": length,
		"prompt": history
	})

	try:
		# Send POST request
		connection.request(method=API_METH, url=API_PATH, body=payload, headers=HEADER)
		response = connection.getresponse()
		
		# Read and decode the response
		if response.status == 200:
			response_data = response.read().decode("utf-8")
			return json.loads(response_data)
		else:
			print(f"Error: Received status code {response.status}: {response.reason}")
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
	parser.add_argument('--length', type=str, default=256, help='The length of the generated text')

	args = parser.parse_args()

	message = args.message
	model = args.model
	length = args.length

	# Now you can use `message` and `model` in your script
	print(f"Message: {message}")
	print(f"Model: {model}")
	print(f"Length: {length}")

	response = send_message_to_llm(message, model, length)
	if response:
		print("Response from API:\n", response[-1][-1]["content"])

		with open(CHAT_FILE, 'w') as filestream:
			json.dump(response[-1], filestream, indent=4)

if __name__ == "__main__":
	main()
