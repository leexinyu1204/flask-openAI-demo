import json
import os

import openai
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import BertForQuestionAnswering, BertTokenizer

with open("./config.json") as config_file:
    config_details = json.load(config_file)

# Setting up the deployment name
chatgpt_model_name = config_details['CHATGPT_MODEL']

embedding_model_name = config_details['EMBEDDING_MODEL']

# This is set to `azure`
openai.api_type = "azure"

# The API key for your Azure OpenAI resource.
openai.api_key = config_details["OPENAI_API_KEY"]

# The base URL for your Azure OpenAI resource. e.g. "https://<your resource name>.openai.azure.com"
openai.api_base = config_details['OPENAI_API_BASE']

# Currently OPENAI API have the following versions available: 2022-12-01
openai.api_version = config_details['OPENAI_API_VERSION']

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_KEY"] = config_details["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = config_details['OPENAI_API_BASE']
os.environ["OPENAI_API_VERSION"] = config_details['OPENAI_API_VERSION']


from azure.cognitiveservices.vision.customvision.training import CustomVisionTrainingClient
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from azure.cognitiveservices.vision.customvision.training.models import ImageFileCreateBatch, ImageFileCreateEntry, Region
from msrest.authentication import ApiKeyCredentials
import os, time, uuid


# Replace with valid values
ENDPOINT = "https://cutomcaption-prediction.cognitiveservices.azure.com/"
prediction_key = "9f7b873b6dd045aeb66f5fac61eaf073"
prediction_resource_id = "7ceb34b8-2513-4ea6-a0a5-756ccb375366"

# Now there is a trained endpoint that can be used to make a prediction
prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
publish_iteration_name = "Iteration2"
project_id = "4d967467-eda7-4ae8-a6ad-d77c0a8c9947"


embeddings = SentenceTransformerEmbeddings(model_name="./models/qaEmbedding", )
embedding_model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
embedding_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



if __name__ == "__main__":
    print(config_details)