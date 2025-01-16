from flask import Flask, request, jsonify
from gpt4all import GPT4All

# Initialize the Flask app
app = Flask(__name__)

# Load the GPT-4 All model
model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # Specify the model name

@app.route('/generate', methods=['POST'])
def generate():
    try:
        # Parse the JSON request
        data = request.json
        prompt = data.get("prompt", "")
        
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        
        # Generate response using the model
        with model.chat_session():
            response = model.generate(prompt)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
