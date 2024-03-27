import argparse
from langchain_community.llms import LlamaCpp
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import gradio as gr

def load_multiple_documents(file_paths):
  documents = []
  for path in file_paths:
    loader = PDFMinerLoader(path)
    documents.extend(loader.load())
  return documents

def create_vector_store_from_documents(documents):
  embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/distiluse-base-multilingual-cased-v2")
  vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./storage"
  )
  return vector_store

parser = argparse.ArgumentParser(description="LLM with document support.")
parser.add_argument("--model", type=str, help="Path to the LLM model.")
parser.add_argument("--pdf", type=str, help="Path to the PDF document.")
args = parser.parse_args()

llm = LlamaCpp(
  model_path=args.model,
  n_gpu_layers=128,
  n_ctx=8196,
  verbose=True,
)

documents = PDFMinerLoader(args.pdf).load()

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=400,
  chunk_overlap=0,
)
texts = text_splitter.split_documents(documents)

vector_store = create_vector_store_from_documents(texts)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

template = """<s>[INST] <<SYS>>
あなたは誠実で優秀な日本人のアシスタントです。前提条件の情報だけで回答してください。 
<</SYS>>

前提条件:{context}

質問:{question} [/INST] """

prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()
setup_and_retrieval = RunnableParallel(
  {"context": retriever, "question": RunnablePassthrough()}
)
chain = setup_and_retrieval | prompt | llm | output_parser

def generate_response(message, history):
  output = ""
  for s in chain.stream(message): #, context=context):
    output += s
    yield output

demo = gr.ChatInterface(
  generate_response,
  title="LLM",
  description="",
  examples=["Hello"],
  cache_examples=True,
  retry_btn=None,
  undo_btn=None,
  clear_btn="Clear",
)
demo.launch()
