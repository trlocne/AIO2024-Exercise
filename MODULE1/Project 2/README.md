# Chatbot with Retrieval-Augmented Generation (RAG)

## Introduction

This project leverages Retrieval-Augmented Generation (RAG) to boost question-answering capabilities using the AIO2024 module's course materials.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/trlocne/AIO2024-Exercise.git
   cd AIO2024-Exercise
   ```

2. **Install Dependencies**:
   Make sure you have Python installed. Install the required dependencies using pip:
   ```bash
   pip install -q -r requirements.txt
   ```

## Features
 - *PDF Processing*: Upload a PDF file which is then transformed into a vector database using Chroma.
 - *Question Answering*: Ask relevant questions based on the content of the PDF, and receive accurate answers.
 - *Efficient Model*: Utilizes the Vicuna Vicious-7B-v1.5 model with quantization for efficient performance.
 - *User-Friendly Interface*: Built with Chainlit for an intuitive and interactive user experience.

## Getting started
All implementation details are given in the file `llm_chatbot.py`.

## Acknowledgements
- Vicuna Vicious-7B-v1.5 for the language model.
- Chroma for the vector database.
- Langchain for the pipeline.
- Chainlit for the frontend framework for Conversational AI.


## This `README.md` provides a brief overview and installation instructions, adhering to your preference for concise documentation.