# FastApi-App
A setup to implement FastAPI using Langchain, WatsonX, LLM 


This project sets up a FastAPI application that acts as an interface for interacting with a language model, specifically the Watson X language model. 

Here's a description of the project:

### Imports and Setup: The necessary modules and libraries are imported, including FastAPI, pydantic, uvicorn, and others. Environment variables are loaded using dotenv.

### Model Initialization: The code initializes a Watson X language model with specified parameters such as decoding method, maximum new tokens, and stop sequences.

### Prompt Definition: A conversation prompt template is defined using the ChatPromptTemplate and HumanMessagePromptTemplate. This template provides guidelines for the conversation flow between the user and the language model.

### Memory Management: A conversation buffer memory is utilized to store context and conversation history.

### LLM Chain Setup: An LLMChain is instantiated with the Watson X language model, prompt template, and memory configuration. This chain manages the interaction between the user and the language model.

### API Endpoint: An API endpoint ("/get_model_response") is defined to receive user queries, process them using the language model, and return the model's response. The endpoint expects JSON data containing a query field.

### Response Model: A response model (ResponseCustom) is defined to specify the structure of the response returned by the API endpoint.

### Server Execution: The FastAPI application is run using the uvicorn server on a specified port.

### Overall, this project sets up a web server that provides a RESTful API for interacting with a Watson X language model. Users can submit queries to the API, and the model generates responses based on the provided guidelines and context. 
