import os
from urllib.request import urlretrieve
import numpy as np
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

loader = PyPDFDirectoryLoader("./docs/")

docs_before_split = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 700,
    chunk_overlap  = 50,
)
docs_after_split = text_splitter.split_documents(docs_before_split)

docs_after_split[0]

print(docs_after_split[1])

avg_doc_length = lambda docs: sum([len(doc.page_content) for doc in docs])//len(docs)
avg_char_before_split = avg_doc_length(docs_before_split)
avg_char_after_split = avg_doc_length(docs_after_split)

print(f'Before split, there were {len(docs_before_split)} documents loaded, with average characters equal to {avg_char_before_split}.')
print(f'After split, there were {len(docs_after_split)} documents loaded, with average characters equal to {avg_char_after_split}.')

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)

sample_embedding = np.array(huggingface_embeddings.embed_query(docs_after_split[0].page_content))
print("Sample embedding of a document chunk: ", sample_embedding)
print("Size of the embedding: ", sample_embedding.shape)

vectorstore = FAISS.from_documents(docs_after_split, huggingface_embeddings)

query = """give me 2 sample test-cases for rrc reconfiguration"""  
         # Sample question, change to other questions you are interested in.
relevant_documents = vectorstore.similarity_search(query)
print(f'There are {len(relevant_documents)} documents retrieved which are relevant to the query. Display the first one:\n')
print(relevant_documents[0].page_content)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


#from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
#from huggingface_hub import login

# Log in to Hugging Face and add the token to git credentials
#login(add_to_git_credential=True)

#hf = HuggingFaceEndpoint(
#    repo_id="mistralai/Mistral-7B-v0.1",
#    model_kwargs={"temperature":0.1, "max_length":500}, huggingfacehub_api_token="Use your key")

## This is the online version
hf = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-v0.1",
    huggingfacehub_api_token="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",
    temperature=0.1,           # Explicitly pass temperature
    max_length=500             # Explicitly pass max_length
)
llm=hf

print("#########################################################################")
query = """Who is the author of attention is all you need"""  # Sample question, change to other questions you are interested in.
response = hf.invoke(query)
print (response)
print("#########################################################################")

prompt_template = """Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with five sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
# Call the QA chain with our query.
result = retrievalQA.invoke({"query": query})
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(result['result'])
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

relevant_docs = result['source_documents']
print(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')
print("*" * 100)
for i, doc in enumerate(relevant_docs):
    print(f"Relevant Document #{i+1}:\nSource file: {doc.metadata['source']}, Page: {doc.metadata['page']}\nContent: {doc.page_content}")
    print("-"*100)
    print(f'There are {len(relevant_docs)} documents retrieved which are relevant to the query.')
