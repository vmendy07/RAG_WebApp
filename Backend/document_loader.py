from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_process_document(data):
    # Convert JSON data to text (e.g., player names, stats, etc.)
    texts = []
    
    # Update this key based on the actual API response
    for player in data.get('players', []):  # Use 'players' instead of 'athletes'
        player_info = f"Name: {player['fullName']}, Position: {player['position']['name']}, Team: {player['team']['displayName']}"
        texts.append({"page_content": player_info, "metadata": {}})
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    return text_splitter.split_documents(texts)
