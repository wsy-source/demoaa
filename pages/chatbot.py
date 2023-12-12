from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import SemanticField, SemanticConfiguration,SemanticSearch,SemanticPrioritizedFields
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage

import streamlit as st
model="gpt-4-32k"
embeddings_model="text-embedding-ada-002"


import time
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text="", display_method='markdown'):
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        time.sleep(0.020)
        self.container.chat_message("assistant").write(self.text,unsafe_allow_html=True)


AZURE_SEARCH_ENDPOINT="https://gptdocument.search.windows.net"
ADMIN_KEY="W7XVOLsbEZgcvWqLD0MxpdVbAfWf1ZIZG7HYdIVSgPAzSeDTMtqR"
INDEX_NAME="gptdocument"


col1, col2 = st.columns([0.8,0.2])

question = col1.text_input(label="input you question")
result = col2.button("search")
con = st.container()
data= con.empty()
if result:

    prompt = """
    User Requirements: {user_requirements}

    Document: {document}

    Answer in Simplified Chinese
    """
    llm=AzureChatOpenAI(
        openai_api_type="azure",azure_endpoint="https://sean-aoai-gpt4.openai.azure.com/",
        api_key="3397748fcdcb4a5fbeb6c2eb5a6a284f",api_version="2023-05-15",
        model=model,callbacks=[StreamHandler(data)],streaming=True
    )


    content = PyPDFLoader("jjad179.pdf").load()
    template= PromptTemplate.from_template(template=prompt)


    chain = LLMChain(prompt=template,llm=llm,verbose=True)
    chain.run(document=content,user_requirements=question)



