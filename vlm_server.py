"""
VLM Server for Craftax Augmented Evaluation
Hosts Qwen3-VL model and provides API for generating hidden states
"""
import os
import argparse
from flask import Flask, request, jsonify
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
import io
import base64

# VLM Configuration
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
TOKENS_GENERATED = 256

gamedesc = """Craftax is a game about exploring dungeons, mining, crafting and fighting enemies. The player can move in the four cardinal directions using WASD and can interact using SPACE. Interacting can cause the player to attempt to mine (a block), attack (a creature), drink (water or from a fountain), eat (fruit) or open a chest.

The player has 5 'intrinsics': health, hunger, thirst, energy and mana (magical energy). Hunger, thirst and energy will naturally decrease and must be replenished by eating, drinking and sleeping respectively. Mana is used for casting spells or enchanting items and will naturally recover. Health will recover when hunger, thirst and energy are non-zero and will decrease if any of these are 0. If the players health falls beneath 0 they will die and the game will restart.

To progress through the game the player needs to find the ladder on each floor, which can be used to descend to the next level. Each floor possesses unique challenges and creatures, increasing in difficulty until the final boss level. The ladders begin closed and the player must kill 8 creatures on each level to open up the respective ladders (with the exception of the overworld). There are 9 levels in total.
"""

question = """
"Think about what the character should do next, keeping in mind the intrinsics displayed on the screen. Provide a detailed action plan for the next steps the character should take to ensure survival and progress in the game."
"""

app = Flask(__name__)

# Global variables for model
vlm_model = None
processor = None

def create_consolidated_prompt(img_pil):
    """Create VLM prompt from PIL Image"""
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": gamedesc + "\n" + question},
            ],
        }
    ]
    return msg

def load_vlm_model():
    """Load the Qwen3-VL model"""
    print(f"Loading VLM model: {MODEL_ID}...")
    import flash_attn

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        quantization_config=None,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("VLM model loaded successfully!")

    return model, proc

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready',
        'model': MODEL_ID
    })

@app.route('/get_hidden_state', methods=['POST'])
def get_hidden_state():
    """
    Generate hidden state from observation image

    Expected JSON:
    {
        "obs": base64-encoded PNG image (or numpy array as list)
    }

    Returns JSON:
    {
        "hidden_state": list of floats (length 2560)
    }
    """
    try:
        data = request.get_json()

        # Handle base64 image or numpy array
        if 'obs_base64' in data:
            img_bytes = base64.b64decode(data['obs_base64'])
            img_pil = Image.open(io.BytesIO(img_bytes))
        elif 'obs' in data:
            # Assume obs is a list representing numpy array
            obs_np = np.array(data['obs'], dtype=np.float32)
            if obs_np.max() <= 1.0:
                img_pil = Image.fromarray((obs_np * 255).astype(np.uint8))
            else:
                img_pil = Image.fromarray(obs_np.astype(np.uint8))
        else:
            return jsonify({'error': 'Missing obs or obs_base64'}), 400

        # Create prompt
        prompt = create_consolidated_prompt(img_pil)

        # Process through VLM
        inputs = processor.apply_chat_template(
            [prompt],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(vlm_model.device)

        with torch.no_grad():
            outputs = vlm_model.generate(
                **inputs,
                max_new_tokens=TOKENS_GENERATED,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )

        # Extract hidden states (last layer, subsample every 8 tokens)
        last_layer_states_list = [
            step_hidden_states[-1] for step_hidden_states in outputs.hidden_states
        ]
        generated_hidden_states = torch.cat(last_layer_states_list, dim=1)
        seq_len = generated_hidden_states.shape[1]
        indices = torch.arange(
            seq_len - 1, -1, -8, device=generated_hidden_states.device
        )
        last_layer_hidden_state = generated_hidden_states[:, indices, :]  # (1, ~32, 2560)

        # Mean pool to (1, 2560)
        pooled_hidden = torch.mean(last_layer_hidden_state, dim=1)  # (1, 2560)

        # Convert to list for JSON
        hidden_np = pooled_hidden.cpu().numpy()[0]  # (2560,)
        hidden_list = hidden_np.tolist()

        return jsonify({
            'hidden_state': hidden_list,
            'shape': list(hidden_np.shape)
        })

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    args = parser.parse_args()

    # Load model globally
    global vlm_model, processor
    vlm_model, processor = load_vlm_model()

    print(f"\nStarting VLM server on {args.host}:{args.port}")
    print(f"Model: {MODEL_ID}")
    print("Endpoints:")
    print(f"  - GET  {args.host}:{args.port}/health")
    print(f"  - POST {args.host}:{args.port}/get_hidden_state")
    print("\nServer ready!\n")

    app.run(host=args.host, port=args.port, threaded=False)

if __name__ == "__main__":
    main()
