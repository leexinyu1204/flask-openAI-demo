import os
from langchain import FAISS
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from authentication import embedding_model_name, embedding_tokenizer

SECTION_LENGTH = 200
MAX_SECTION_OVERLAP = 0
openAIEmbedding = OpenAIEmbeddings(
    deployment=embedding_model_name,
    chunk_size=1
)
sentenceEmbedding = SentenceTransformerEmbeddings(model_name="./models/qaEmbedding")


def create_index_database(rootdir, embeddings=openAIEmbedding, loadfile="./faiss_index", load=True):
    if load:
        try:
            db = FAISS.load_local(loadfile, embeddings)
            return db
        except:
            print("no local database! Please create your own database.")
    documents = []
    for subdir, dirs, files in os.walk(rootdir):

        for file in files:
            filename = os.path.join(subdir, file)
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(filename)
                documents.extend(loader.load())

    text_splitter = TokenTextSplitter(chunk_size=SECTION_LENGTH, chunk_overlap=MAX_SECTION_OVERLAP)
    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)
    db.save_local(loadfile)
    return db


def search_docs(db, user_input, top_n=3):
    docs = db.similarity_search(user_input, embedding_model_name)
    results = []
    for i in range(top_n):
        if 'imageName' in docs[i].metadata:
            fname = docs[i].metadata['imageName']

            # print(docs[i].metadata)
            res = fname + ": " + docs[i].page_content.replace("\n", "").replace("\r", "")
            results.append(res)
        elif 'source' in docs[i].metadata:
            fname = docs[i].metadata['source']
            # print(docs[i].metadata)
            res = fname + ": " + docs[i].page_content.replace("\n", "").replace("\r", "")
            results.append(res)
    return results


def search_docs_by_vector(db, user_input, embeddings=sentenceEmbedding, top_n=3):
    embedding_vector = embeddings.embed_query(user_input)

    docs = db.similarity_search_by_vector(embedding_vector, k=top_n)

    results = []
    for i in range(top_n):
        if 'imageName' in docs[i].metadata:
            fname = docs[i].metadata['imageName']

            # print(docs[i].metadata)
            res = fname + ": " + docs[i].page_content.replace("\n", "").replace("\r", "")
            results.append(res)
        elif 'source' in docs[i].metadata:
            fname = docs[i].metadata['source']
            # print(docs[i].metadata)
            res = fname + ": " + docs[i].page_content.replace("\n", "").replace("\r", "")
            results.append(res)
    return results


if __name__ == "__main__":
    create_index_database('./data', loadfile="./faiss_index", load=False)
    # from sentence_transformers import SentenceTransformer
    #
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # model.save('./models/textEmbedding')
    # with open('url.txt',encoding="utf-8") as f:
    #     lines = []
    #     for line in f:
    #         lines.append(line.replace('\n', ''))
    # print(lines)
    # loader = UnstructuredURLLoader(urls=lines)
    # documents = []
    # documents.extend(loader.load())
    # text_splitter = TokenTextSplitter(chunk_size=SECTION_LENGTH, chunk_overlap=MAX_SECTION_OVERLAP)
    # docs = text_splitter.split_documents(documents)
    # embeddings = SentenceTransformerEmbeddings(model_name="./models/qaEmbedding")
    # db = FAISS.from_documents(docs, embeddings)
    # loadfile = "./faiss_index"
    # db.save_local(loadfile)
    # db = create_index_database("./data", load=False)
    #
    # query = "Does UC ship cover eye test"
    # results = search_docs(db, query, top_n=3)
    # print((results))

    # print("The input has a total of {} tokens.".format(len(input_ids)))
