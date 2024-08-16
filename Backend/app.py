from flask import Flask, request, jsonify
from document_loader import load_and_process_document
from vector_store import setup_vector_store
from qa_system import setup_qa_system
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Path to your document
PDF_PATH = 'docs/example_document.pdf'

texts = load_and_process_document(PDF_PATH)
docsearch = setup_vector_store(texts)
qa_system = setup_qa_system(docsearch)

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message is required'}), 400

    result = qa_system.invoke(message)
    response = result['result']
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
