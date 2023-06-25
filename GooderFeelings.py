import openai
from datasets import load_dataset
import pinecone
from tqdm.auto import tqdm
from time import sleep
import string
import re

"""
Welcome to GooderFeelings!

This module demonstrates how to leverage OpenAI, Pinecone, and HuggingFace to provide more nuanced responses 
for users requesting help

"""

__author__ = 'PantsMalone, KendrickLagrange'
__copyright__ = 'Copyright 2023, GooderFeelings'

def complete(prompt, prompt_enhancement=""):
    # query text-davinci-003
    prompt = prompt + " " + prompt_enhancement
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0.1,
        max_tokens=2000,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()


def remove_non_ascii(a_str):
    ascii_chars = set(string.printable)

    return ''.join(
        filter(lambda x: x in ascii_chars, a_str)
    )

def query_cleanup(query):
    query = re.sub('\s+', ' ', query)
    query = query.replace(".", ". ")
    return query

def main():
    # get API key from top-right dropdown on OpenAI website
    # openai.api_key = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"
    openai.api_key = "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"

    openai.Engine.list()  # check we have authenticated

    embed_model = "text-embedding-ada-002"

    res = openai.Embedding.create(
        input=[
            "Sample document text goes here",
            "there will be several phrases in each batch"
        ], engine=embed_model
    )

    data = load_dataset('Amod/mental_health_counseling_conversations', split='train')

    # initialize connection to pinecone (get API key at app.pinecone.io)
    api_key = "a1b4330b-7677-43e5-be21-781fa43c2b62"
    # find your environment next to the api key in pinecone console
    env = "asia-northeast1-gcp"

    pinecone.init(api_key=api_key, environment=env)
    pinecone.whoami()

    index_name = 'gen-qa-openai'

    # check if index already exists (it shouldn't if this is first time)
    if index_name not in pinecone.list_indexes():
        # if index does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=len(res['data'][0]['embedding']),
            metric='cosine',
            metadata_config=None
        )
    # connect to index
    index = pinecone.GRPCIndex(index_name)
    batch_size = 100

    print("Creating vectors for Pinecone...")
    populate_pinecone_embeddings(batch_size, data, embed_model, index, res)

    # Prompt user
    query = input("Hi friend! How can I help you today?").strip()
    query = query_cleanup(query)
    res = complete(query)

    # We want to match the best response to our dataset
    openai_embedding = openai.Embedding.create(
        input=[res],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = openai_embedding['data'][0]['embedding']

    # remove all escape characters to cleanup response data
    escapes = ''.join([chr(char) for char in range(1, 32)])
    translator = str.maketrans('', '', escapes)

    # get relevant responses (including the questions)
    new_res = index.query(xq, top_k=2, include_metadata=True)
    new_res_response_cleaned = new_res['matches'][0]['metadata']['Response'].translate(translator).strip()

    # re-prompt chatgpt given this therapist's response to the query
    prompt_enhancement = query_cleanup("please answer this in one detailed sentence with the " \
                                       "goal of making me feel better from the perspective of a therapist based on "
                                       "the advice here: " + new_res_response_cleaned)
    final_response = complete(query, prompt_enhancement)
    print(final_response)


def populate_pinecone_embeddings(batch_size, data, embed_model, index, res):
    for i in tqdm(range(0, 2300, batch_size)):
        # find end of batch
        i_end = min(len(data), i + batch_size)
        meta_batch = data[i:i_end]

        # get context
        # list of strings, each string contains one context
        context_batch = meta_batch['Context']

        ids_batch = map(str, range(i, i_end))

        # list of strings, each string contains one response
        response_batch = meta_batch['Response']

        # create embeddings (try-except added to avoid RateLimitError)
        try:
            res = openai.Embedding.create(input=response_batch, engine=embed_model)
        except:
            done = False
            while not done:
                sleep(5)
                try:
                    res = openai.Embedding.create(input=response_batch, engine=embed_model)
                    done = True
                except:
                    pass
        embeds = [record['embedding'] for record in res['data']]

        # cleanup metadata
        # loop over curr_iter batch size 0 -> batch_size

        for curr_iter in range(0, batch_size):
            meta_batch = [{
                'Context': context_batch[curr_iter],
                'Response': response_batch[curr_iter]
            }]

        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        index.upsert(vectors=to_upsert)


if __name__ == "__main__":
    main()
