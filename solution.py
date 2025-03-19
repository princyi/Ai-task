import os
from flask import Flask, request, jsonify
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Step 1: Load data from the website
loader = WebBaseLoader("https://brainlox.com/courses/category/technical")
documents = loader.load()

# Step 2: Create embeddings and store them in FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Step 3: Create a conversation chain
llm = OpenAI()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

# Step 4: Set up Flask RESTful API
app = Flask(__name__)

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "Message is required"}), 400
    response = qa_chain.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



1. Set Up Your OpenAI API Key
Since we’re using OpenAI embeddings and LLM, you need to set up your API key.

Create a .env file in your project directory and add:
sh
Copy
Edit
OPENAI_API_KEY="your-api-key-here"
Update the Python script to load the API key:
python
Copy
Edit
from dotenv import load_dotenv
load_dotenv()
2. Install Required Python Packages
Ensure you have all dependencies installed. Run:

sh
Copy
Edit
pip install flask langchain openai faiss-cpu python-dotenv
3. Test Your Chatbot API
Run the Flask app:

sh
Copy
Edit
python app.py
Then, send a POST request using Postman or cURL:

sh
Copy
Edit
curl -X POST http://127.0.0.1:5000/chat -H "Content-Type: application/json" -d '{"message": "What technical courses are available?"}'
Expected Response:

json
Copy
Edit
{
  "response": "List of technical courses from brainlox.com..."
}



