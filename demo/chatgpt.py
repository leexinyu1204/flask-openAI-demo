import re
from pprint import pprint

import torch
from langchain.embeddings import OpenAIEmbeddings

from langchainadaplter import search_docs,search_docs_by_vector
from authentication import *
openAIEmbedding = OpenAIEmbeddings(
    deployment=embedding_model_name,
    chunk_size=1
)
sentenceEmbedding = SentenceTransformerEmbeddings(model_name="./models/qaEmbedding")
# from prepData import *

prompt_prefix = """<|im_start|>system
You are the assistant to provide information about products
Answer ONLY with the facts listed in the list of sources below. 
If there isn't enough information below, say you don't know. 
Do not generate answers that don't use the sources below. 

Sources:
{sources}

<|im_end|>"""

turn_prefix = """
<|im_start|>user
"""

turn_suffix = """
<|im_end|>
<|im_start|>assistant
"""

prompt_history = turn_prefix


def get_response_azure(db, user_input):
    # global prompt_history

    results = search_docs_by_vector(db, user_input, embeddings = openAIEmbedding )

    content = "\n".join(results)

    prompt = prompt_prefix.format(sources=content) + prompt_history + user_input + turn_suffix

    response = openai.Completion.create(
        engine=chatgpt_model_name,
        prompt=prompt,
        temperature=1,
        max_tokens=256,
        n=1,
        stop=["<|im_end|>", "<|im_start|>"])
    answer = response.choices[0].text

    imgSRC = re.findall("\[(.*?\.(png|jpg))\]", answer, re.I)

    if len(imgSRC) != 0:
        imgSRC = imgSRC[0][0]
    else:
        imgSRC = ''


    # prompt_history += user_input + turn_suffix + response.choices[0].text + "\n<|im_end|>" + turn_prefix
    return answer, imgSRC


def get_response_bert(db, user_input, embeddings):
    results = search_docs_by_vector(db, user_input, embeddings)
    text = "\n".join(results)
    # print(text)
    pprint(text.encode('utf8'))
    input_ids = embedding_tokenizer.encode(user_input, text, max_length=512, truncation=True)
    # print("The input has a total of {} tokens.".format(len(input_ids)))

    tokens = embedding_tokenizer.convert_ids_to_tokens(input_ids)

    # first occurence of [SEP] token
    sep_idx = input_ids.index(embedding_tokenizer.sep_token_id)
    # print("SEP token index: ",
    #       sep_idx)  # number of tokens in segment A (question) - this will be one more than the sep_idx as the index in Python starts from 0
    num_seg_a = sep_idx + 1
    # print("Number of tokens in segment A: ", num_seg_a)  # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a
    # print("Number of tokens in segment B: ", num_seg_b)  # creating the segment ids
    segment_ids = [0] * num_seg_a + [1] * num_seg_b  # making sure that every input token has a segment id
    # assert len(segment_ids) == len(input_ids)
    output = embedding_model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end < answer_start or '[CLS]' in tokens[answer_start:answer_end + 1]:
        answer = "I am unable to find the answer to this question. Can you please ask another question?"
        return answer, ""

    # Start with the first token.
    answer = tokens[answer_start]

    # Select the remaining answer tokens and join them with whitespace.
    for i in range(answer_start + 1, answer_end + 1):

        # If it's a support token, then recombine it with the previous token.
        if tokens[i][0:2] == '##':
            answer += tokens[i][2:]

        # Otherwise, add a space then the token.
        else:
            answer += ' ' + tokens[i]

    return answer, ""
