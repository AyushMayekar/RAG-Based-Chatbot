## RAG-Based Document Chatbot

Welcome to the RAG-Based Document Chatbot repository! This project is an AI-powered chatbot that answers questions based solely on the content of provided documents. By utilizing the Retrieval-Augmented Generation (RAG) principle, the chatbot retrieves relevant information and generates accurate responses, making it a valuable tool for efficient knowledge management.

## Key Features

- **Context-Specific Responses**: The chatbot provides answers strictly based on the content of the uploaded document.
- **Advanced Retrieval Mechanism**: Leverages the RAG framework to retrieve the most relevant portions of the document for accurate and context-aware answers.
- **User-Friendly Interface**: Designed for simplicity and ease of interaction, making it easy to upload documents and get immediate responses.

## Technology Stack

- **Python**: Backend development and FastAPI integration.
- **LangChain**: For document loading, splitting, and vector store management.
- **MongoDB Atlas**: Vector storage and retrieval mechanism using MongoDB Atlas.
- **FastAPI**: Framework for building the REST API.
- **Ollama Embeddings**: For creating document embeddings and improving retrieval accuracy.

## Installation

1. **Clone the Repository**:
    
   ```sh
   git clone https://github.com/AyushMayekar/RAG-Based-Chatbot
2. **Install the dependencies**

   ```sh
   pip install -r requirements.txt
3. **Run the application**

   ```sh
   uvicorn main:app --reload

4. **Run the Streamlit App**

   ```sh
   streamlit run client.py
   
## Screenshots

1. Chatbot Interface Example:
 
   ![1](https://github.com/AyushMayekar/RAG-Based-Chatbot/blob/main/Screenshot%202024-08-19%20104055.png)

2. Chatbot Interface Example:

   ![2](https://github.com/AyushMayekar/RAG-Based-Chatbot/blob/main/Screenshot%202024-08-19%20104131.png)


## Demonstration

Watch our project in action on YouTube:

[![Watch the video](https://img.youtube.com/vi/xikdsRg2TZM/maxresdefault.jpg)](https://youtu.be/xikdsRg2TZM)

## Contact

For further information about the Chatbot, please reach out to the project creator via [LinkedIn](https://www.linkedin.com/in/ayush-mayekar-b9b883284)

Thank you for your interest and valuable time. 🤝
