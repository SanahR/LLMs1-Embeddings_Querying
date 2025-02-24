"""
This project is intended to set up a RAG application, but it only does embeddings and querying. 

The next steps in this journey are chunking (chunking is splitting up large documents into smaller parts) and prompting (this part is where the LLM comes in. ). 

RAG (Retrieval Augmented Generation) allows generative AI to respond to user input with a specific set of information that it retrieves from a database. 

If you would like to know more about RAG, here are some insightful, 2024 review papers:
Retrieval-Augmented Generation for Large Language Models: A Survey - https://arxiv.org/pdf/2312.10997
Retrieval-Augmented Generation for Natural Language Processing: A Survey - https://arxiv.org/pdf/2407.13193

Sentence Transformers, the default embedding function for chromadb, is used unless you specify that you want to use something else (eg. OpenAI. )
The specific model used is called the all-MiniLM-L6-v2. To see more about the model, view this webpage: https://www.sbert.net/

These are both needed for the code to work. You have to run them before anything else.
 !pip install python-dotenv
 !pip install chromadb

If your API Key is stored in your google drive, you'll need this part (and if you're using google colab):
 from google.colab import drive
 drive.mount('/content/Drive/')

"""

from dotenv import dotenv_values
import glob
import matplotlib.pyplot as plt
import chromadb
import chromadb.utils.embedding_functions as embedding_functions #chromadb provides you with the option to get many embedding functions, so we're importing one here.
from langchain_text_splitters import RecursiveCharacterTextSplitter


####################################
#     GETTING/CREATING THE DATA    #
####################################

Brian_Light = "RAG-Project-Database-Files/Brian_Light.md"
Ava_Hatestring = "RAG-Project-Database-Files/Ava_Hatestring.md"
Arya_Arupathy = "RAG-Project-Database-Files/Arya_Arupathy.md"
Demetrius_Obole = "RAG-Project-Database-Files/Demetrius_Obole.md"

# **************************************************************************************************
#If you don't have an Open AI API Key, go to platform.openai.com. Then, create an account and click on dashboard.
#In the side-menu, click on "API Keys", then click on "Create a New Secret Key". (As of 2/16/25)
#Here, .env is the name of my file.
api_key = dotenv_values(".env")['OpenAI_APIkey']

#############################
#     SETTING UP RAG        #
#############################

#Step 1. Create a client of the vector database (set up the vector database)
chroma_client = chromadb.Client()

#Step 2. Creating the collection (you can name it anything you want)

#This is optional. If you don't do this, then your program will be using the Sentence Transformers all-MiniLM-L6-v2 embedding model.
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key = api_key,
    model_name = "text-embedding-3-small")

collection = chroma_client.create_collection(name = "Student_Profiles",embedding_function = openai_ef)
#This is the default embedding function for chroma: Sentence Transformers (SBERT)(all-MiniLM-L6-v2)

#Step 3. Adding the documents to the collection
collection.add(documents=[open(Brian_Light).read(),open(Ava_Hatestring).read(),open(Arya_Arupathy).read()],ids=["students1","students2","students3"])

#This data was added later, so it was added in a different line so as not to repeat anything that was already in the collection.
collection.add(documents=[open(Demetrius_Obole).read()],ids=["students4"])

#############################
#     QUERYING              #
#############################
question = input("Search for a student by a quality or statement they've had/made: ")
results = collection.query(query_texts = [question],n_results=1) #This is really the only necessary part of querying.

#This part is also optional.
#Here, I wanted to make the program show only the first and last name of the student.
#This required me to index into results in such a way that I could combine the two without showing anything else.
name = (results['documents'][0][0].split())[4]+" "+(results['documents'][0][0].split())[5]

print(name)

#This is optional, but I wanted to add a feature that lets you see the whole document.
question2 = input(f"Would you like to see everything {name} wrote? y/n")
if question2 == "y":
  print(results['documents'][0][0])
else:
  print("Come again next time!")
