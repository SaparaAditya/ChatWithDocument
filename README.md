# Chatbot for Document Querying
This repository contains code for a chatbot capable of answering questions based on the content of various document types such as PDFs, DOCX files, PPTX files, HTML files, LaTeX files, and plain text files.

# Features
Support for Multiple Document Types: The chatbot supports querying information from PDFs, DOCX files, PPTX files, and plain text files.
Streamlit Interface: The chatbot interface is built using Streamlit, making it easy to interact with.
Document Parsing: Utilizes libraries such as PyPDF2, docx, pptx, BeautifulSoup, and pypandoc for extracting text content from different document types.
Text Chunking: Splits the extracted text into smaller chunks for efficient processing.
Semantic Embeddings: Utilizes Google's Generative AI embeddings for understanding the semantics of text chunks.
Conversational AI: Implements a conversational AI model to answer user queries based on the provided context and question.
FAISS Vector Store: Utilizes FAISS for efficient similarity search based on semantic embeddings.

![Lang drawio](https://github.com/parthjrana01/Chat-with-Document/assets/131896580/854c5c8c-2ccf-4922-b979-4b89784be5f1)


# Setup
1. Clone the Repository:
git clone https://github.com/parthjrana01/Chat-with-Document.git
cd your-repo


2. Install Dependencies:
pip install -r requirements.txt


3. Run the Application:
streamlit run main.py

4. Upload Documents and Start Chatting:
Upload your desired documents in PDF, DOCX, PPTX, HTML, LaTeX, or plain text format.
Ask questions related to the document content and interact with the chatbot.

# Demo : 
1. One Pdf of Research Paper is uploaded & query is asked to "summarize the document"
![Screenshot 2024-03-02 111149](https://github.com/parthjrana01/Chat-with-Document/assets/131896580/d3a68b44-d776-4a09-babd-2cb8833bc7ea)

2. Another question is answered by the Model for Pdf
![Screenshot 2024-03-02 111219](https://github.com/parthjrana01/Chat-with-Document/assets/131896580/08a5ec8e-6a5f-4141-a27d-0a20b9a6e5eb)

3. Demonstration for PPTX
![Screenshot 2024-03-02 111243](https://github.com/parthjrana01/Chat-with-Document/assets/131896580/fa0771ec-f1a3-4914-ab8e-d3f7c1a067bb)

4. Demonstration for DOCX
![Screenshot 2024-03-02 111308](https://github.com/parthjrana01/Chat-with-Document/assets/131896580/fcf7d8dc-beba-4d77-b233-0c04e338fc54)






# Usage
Ask Questions: Input your question in the provided text box.
Upload Documents: Upload the documents you want to query.
Submit & Process: Click the button to submit your query and process the documents.
View Responses: The chatbot will generate responses based on the provided question and document content.


# Contributors
#### Rana Parth
#### Arafat Sable
#### Jay Valaki 
#### Aditya Sapara
#### Mehul Zalaiya
