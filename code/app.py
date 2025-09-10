import os
import json
from datetime import datetime
from flask import Flask, request, Response, jsonify
from dotenv import load_dotenv
import numpy as np
import math

# Initialize app and env
app = Flask(__name__)
load_dotenv()

# Dynamic version tag for visibility in logs
COMPETITION = os.getenv("COMPETITION", "competition19")
TOPIC_ID = os.getenv("TOPIC_ID", "65")
TOKEN = os.getenv("TOKEN", "BTC")
TIMEFRAME = os.getenv("TIMEFRAME", "8h")
MCP_VERSION = f"{datetime.utcnow().date()}-{COMPETITION}-topic{TOPIC_ID}-app-{TOKEN.lower()}-{TIMEFRAME}"
FLASK_PORT = int(os.getenv("FLASK_PORT", 8001))

def sanitize_for_json(obj):
    """
    Recursively sanitize an object to ensure it's JSON serializable.
    Replaces NaN, inf, -inf with None or appropriate values.
    """
    if isinstance(obj, dict):
        return {key: sanitize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (int, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        elif np.isinf(obj):
            return 1e9 if obj > 0 else -1e9
        else:
            return float(obj)
    elif hasattr(obj, 'item'):  # numpy scalars
        return sanitize_for_json(obj.item())
    else:
        # For any other type, try to convert to string
        return str(obj)

# MCP Tools
TOOLS = [
    {
        "name": "optimize",
        "description": "Triggers model optimization using Optuna tuning and returns results.",
        "parameters": {}
    },
    {
        "name": "write_code",
        "description": "Writes complete source code to a specified file, overwriting existing content.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "The file to write to."},
                "code": {"type": "string", "description": "The code to write."}
            },
            "required": ["filename", "code"]
        }
    }
]

@app.route('/tool/optimize', methods=['POST'])
def tool_optimize():
    from model import tune_model
    results = tune_model()
    results = sanitize_for_json(results)
    return jsonify(results)

@app.route('/tool/write_code', methods=['POST'])
def tool_write_code():
    data = request.json
    filename = data['filename']
    code = data['code']
    with open(filename, 'w') as f:
        f.write(code)
    return jsonify({"status": "written"})

@app.route('/', methods=['GET'])
def index():
    return "MCP Version: " + MCP_VERSION

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=True)