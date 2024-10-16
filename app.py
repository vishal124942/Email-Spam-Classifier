from flask import Flask, request, jsonify
from spam_classifier import predict_mail  
import json
app = Flask(__name__)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the request data
        data = request.get_json()

        # Check if the 'message' key is present in the request
        if 'message' not in data:
            return jsonify({"error": "No message field provided."}), 400

        input_mail = data['message']

        # Check if the message is empty
        if not input_mail.strip():
            return jsonify({"error": "Message cannot be empty."}), 400

        # Get the prediction
        prediction = predict_mail(input_mail)

        return jsonify({"prediction": prediction})

    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON format."}), 400
    except Exception as e:
        # General error handling
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    app.run(debug=True)