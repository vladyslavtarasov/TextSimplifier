from flask import Flask, request, jsonify
from flask_cors import CORS
import simplifier

app = Flask(__name__)
CORS(app)


# Route that handles POST requests for text simplification.
# The response includes the original text, the simplified text, a mapping of words from original to simplified,
# and additional synonyms for simplified words.
@app.route('/simplify', methods=['POST'])
def simplify_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    original_text = data.get('text')
    original_threshold = data.get('originalThreshold', 0.3)
    new_threshold = data.get('newThreshold', 0.3)

    if not original_text:
        return jsonify({"error": "No text provided"}), 400

    simplified_text, word_mapping, additional_synonyms = simplifier.get_text_simplification(
        original_text, original_threshold, new_threshold)

    response = {
        "original_text": original_text,
        "simplified_text": simplified_text,
        "word_mapping": word_mapping,
        "additional_synonyms": additional_synonyms
    }

    return jsonify(response)


# Route that handles POST requests for text summarization.
# The response includes the original text and its summarized version.
@app.route('/summarize', methods=['POST'])
def summarize_text():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    original_text = data.get('text')
    summarize_coefficient = data.get('summarizeCoefficient', 0.5)

    if not original_text:
        return jsonify({"error": "No text provided"}), 400

    summarized_text = simplifier.get_text_summarization(original_text, summarize_coefficient)

    response = {
        "original_text": original_text,
        "summarized_text": summarized_text,
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(port=8080)
