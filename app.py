from flask import Flask, request, jsonify
from llama_cpp import Llama

app = Flask("llama_server")
model = None

@app.route('/invoke_llm', methods=['POST'])
def generate_response():
    global model

    try:
        data = request.get_json()

        if 'system_message' in data and 'user_message' in data and 'max_tokens' in data and 'context' in data:
            system_message = data['system_message']
            user_message = data['user_message']
            context = data['context']
            max_tokens = int(data['max_tokens'])

            # Prompt Creation
            # prompt = f"""<s>[INST] <<SYS>>
            # {system_message}
            # <</SYS>>
            # {user_message} [/INST]"""

            prompt = f"""<s>[INST] <<SYS>>
                {system_message}
                <</SYS>>
                Context: {context}
                Question: {user_message}
                Answer:
            [/INST]"""

            if model is None:
                model_path = './app/llama-2-7b-chat.Q2_K.gguf'
                model = Llama(model_path=model_path)

            output = model(prompt, max_tokens=max_tokens, echo=True)

            return jsonify(output)
        
        else:
            return jsonify({"error":"Missing the required parameters"}), 400
        
    except Exception as e:
        return jsonify({"Error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)