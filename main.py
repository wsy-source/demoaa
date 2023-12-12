from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from langchain.vectorstores.azuresearch import AzureSearch
from azure.search.documents.indexes.models import SemanticField, SemanticConfiguration,SemanticSearch,SemanticPrioritizedFields
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema.messages import HumanMessage
model="gpt-4-32k"
embeddings_model="text-embedding-ada-002"
llm=AzureChatOpenAI(
    openai_api_type="azure",azure_endpoint="https://sean-aoai-gpt4.openai.azure.com/",
    api_key="3397748fcdcb4a5fbeb6c2eb5a6a284f",api_version="2023-07-01-preview",
    model=model,callbacks=[StreamingStdOutCallbackHandler()],streaming=True,max_tokens=10000)


splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
pdf_content = PyPDFLoader("jjad179.pdf")
content = pdf_content.load_and_split(splitter)

AZURE_SEARCH_ENDPOINT="https://gptdocument.search.windows.net"
ADMIN_KEY="W7XVOLsbEZgcvWqLD0MxpdVbAfWf1ZIZG7HYdIVSgPAzSeDTMtqR"
INDEX_NAME="gptdocument"

prompt = """

Please help me generate a "1500-word" Simplified Chinese article, divided into sections according to background, purpose, methods, main results, discussion, and conclusion.

Document: {document}

"""
content = PyPDFLoader("jjad179.pdf").load()
template= PromptTemplate.from_template(template=prompt)


chain = LLMChain(prompt=template,llm=llm,verbose=True)
chain.run(document=content)