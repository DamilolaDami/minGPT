from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from shakespeareGPT import GPTLanguageModel
import json
import traceback
device = 'cuda' if torch.cuda.is_available() else 'cpu'

app = Flask(__name__)
CORS(app)

# Load the trained model and stoi mapping
model_state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
model = GPTLanguageModel()
model.load_state_dict(model_state_dict)
model.eval()

with open('stoi.json', 'r') as f:
    stoi = json.load(f)

with open('itos.json', 'r') as f:
    itos = json.load(f)

max_new_tokens = 100


@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        model_state_dict = torch.load('model.pth', map_location=torch.device('cpu'))
        model = GPTLanguageModel()
        model.load_state_dict(model_state_dict)
        model.eval()

        context = torch.zeros((1, 1), dtype=torch.long)
        max_new_tokens = 100
        if request.method == 'POST':
            with open('input.txt', 'r', encoding='utf-8') as f:
                text = f.read()
            chars = sorted(list(set(text)))
            vocab_size = len(chars)
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = {i: ch for i, ch in enumerate(chars)}
            encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
            decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            generated_text = decode(model.generate(context, max_new_tokens=800)[0].tolist())
            print("Generated text: ", generated_text)
            return jsonify({'generated_text': generated_text})
        else:
            return jsonify({'error': 'Invalid request'})
        
              
    except:
        return jsonify({'error': 'Model not found'})

    
if __name__ == '__main__':
    app.run(debug=True)
