import streamlit as st 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
import os 
import sys

# Debugging statements
print("Python executable being used:", sys.executable)
print("Virtual environment:", os.getenv('VIRTUAL_ENV'))

try:
    from langchain.vectorstores import FAISS
    print("FAISS import successful")
except ModuleNotFoundError as e:
    print("ModuleNotFoundError:", e)
    print("Ensure langchain-community is installed by running `pip install -U langchain-community`")

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## To read the pdf and go through every page of pdf and read the text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_file = io.BytesIO(pdf.read())
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

## divide text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

## Converting chunks into vectors
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # from specific documentation
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the detail, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt=PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents":docs, "question":user_question},
        return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])

## streamlit app
def main():
    st.set_page_config("Chat With Multiple PDF")
    st.header("Chat with Multiple PDF using Gemini")

    user_question = st.text_input("Ask a Question from the PDF Files...")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files & Click on the Submit & Process", accept_multiple_files=True, type="pdf")
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload PDF files to process.")

if __name__=="__main__":
    main()
