from flask import Flask, request, jsonify
from document_loader import load_and_process_document
from vector_store import setup_vector_store
from qa_system import setup_qa_system
from espn_api import get_players, get_player_stats
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load NFL data from ESPN API
players_data = get_players()
player_stats_data = get_player_stats()

# Load the data into the RAG system (convert JSON to text chunks)
texts = load_and_process_document(players_data)
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
