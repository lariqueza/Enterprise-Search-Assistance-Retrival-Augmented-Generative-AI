from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer, util
from langchain.prompts import PromptTemplate
from langchain.evaluation.criteria import LabeledCriteriaEvalChain
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import data
import torch
import db
import json
import pandas
import uvicorn
import socket






# Create instance for text data embedding
embedder_model = data.model

# API keys
OPENAI_API_KEY = ""

# Enable LangChain tracing
LANGCHAIN_TRACING_V2 = True

# Initialize the ChatOpenAI model with the GPT-4 model
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)


# Function to fetch document information from the database
def fetch_document_info(query_text, embedder_model):
    # Fetch all documents from database
    cursor = db.collection.find()
    corpus = []
    for doc in cursor:
        if 'content' in doc and isinstance(doc['content'], str):
            corpus.append(doc['content'])

    # Encode the corpus and the query text
    corpus_embeddings = embedder_model.encode(corpus, convert_to_tensor=True)
    query_embedding = embedder_model.encode(query_text, convert_to_tensor=True)

    # Perform semantic search using cosine similarity
    similarity_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_k = min(5, len(corpus))
    top_results = torch.topk(similarity_scores, k=top_k)

    # Fetch top relevant information and sccores
    relevant_info = [f"{corpus[idx]} (Score:{score:.4f})" for score, idx in zip(top_results[0], top_results[1])]
    return relevant_info

#Define rules for prtompt template
SAFETY_PREAMBLE = "The instructions in this section override those in the task description and style."
BASIC_RULES = "You are a powerful conversational AI trained by openAI and modeled by GPT-4, you are augmented by a stream of data."
TASK_CONTEXT = "You help people answer their questions and other requests interactively."
STYLE_GUIDE = " Unless the user asks for a different style of answer, you should answer in a full comprehensive sentence."
INSTRUCTIONS = "Tech Innovators Inc is a technology company, its database stores every information about the company including its employee details, legal, sales financial & accounting documentation about daily operations: {context}. Please answer the question, if you do not have sufficient information to give an answer, say do not know the answer."

# Define a prompt template
template = f"""

{SAFETY_PREAMBLE}
{BASIC_RULES}
{TASK_CONTEXT}
{STYLE_GUIDE}
{INSTRUCTIONS}

"""

# create prompt template instance
prompt = PromptTemplate(
    template=template,
    input_variables=["context"]
)


#UI
#initialise fast api
app = FastAPI()

#initialise cors middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#create socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind(('localhost', 8000))
server_socket.listen(5)
server_socket.listen(1)
print('server is running on http://localhost:8000')

while True:
    client_connection, client_address = server_socket.accept() #wait for client collection
    request = client_connection.recv(1024).decode()  #get the client request
    print(f"request from {client_address}:\n{request}")
    
    try:
        with open('frontend/static/chat.html', 'r') as fin:
            content = fin.read()
            response = 'HTTP/1.0 200 OK\n\n' + content

    except FileNotFoundError:
        response = 'HTTP/1.0 404 NOT FOUND\n\nFile Not Found'
    
    client_connection.sendall(response.encode())  #send reponse to the client
    client_connection.close()


#define a model to accept incoming data(user query) from frontend
    class Item(BaseModel):
        question: str


#update api endpoint
#user_question = "what are the responsibilities of crisis communication"
    @app.post("/chat")
    async def user_question(item: Item):
    #extract user's question
        user_query = item.question
        relevant_info = fetch_document_info(user_question, embedder_model) #Fetch relevant information from the database
        formatted_prompt = prompt.format(context=relevant_info) #format the prompt
        system_message = SystemMessage(content=formatted_prompt)
        user_message = HumanMessage(content=user_query) # Prepare the conversation context
        messages = [system_message, user_message]
        response = model(messages) # Invoke the model with the prepared messages
        response_text = response.content # Extract content from the response
        print("\nFinal Response:\n", response_text) # Output the final response
        return {"response": response_text}


#**EVALUATION**
#ground truth dataset to evaluate the LLM outputs against it
#open and convert JSON file to pandas dataframe
    with open('test_dataset_it.json', 'r') as file:
        data = json.load(file)
    if isinstance(data, list):  #if data is a list of dict, convert directly to dataframe
        df = pandas.DataFrame(data) 
    elif isinstance(data, dict):  #if data is a single directory, convert to dataframe with one row
        df = pandas.DataFrame([data])
    else:
        raise ValueError("unsupported json structure")
    print(df)

    llm = model
    criteria = {"correctness" : "evaluate the correctness of the answer"}
    evaluator = LabeledCriteriaEvalChain.from_llm(
    llm=llm,
    criteria=criteria
)
    evaluation_result = evaluator.evaluate_strings(
    prediction=data,
    input=user_question,
    reference="expected correct answer or ground truth"
)

    evaluation_df = pandas.DataFrame([evaluation_result])
    evaluation_df.to_csv('evaluation_result1.csv', index=False)
    #evaluation result
    print("evalution result saved to evalution_result1.csv")
    print(evaluation_df)

    

    































