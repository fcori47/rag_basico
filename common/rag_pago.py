import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


os.environ["OPENAI_API_KEY"] = '' #API Key de OPENAI (version paga)


llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
chroma_local = Chroma(persist_directory="./vectordb", embedding_function=OpenAIEmbeddings())
    

def prompt(texto):
    system_prompt = (
    texto+
    "\n\n"
    "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ])
    return prompt


texto = """Tú eres un asistente para tareas de respuesta a preguntas."
    "Usa los siguientes fragmentos de contexto recuperado para responder "
    "la pregunta. Si no sabes la respuesta, di que no "
    "sabes. Usa un máximo de tres oraciones y mantén la "
    "respuesta concisa."""


def respuesta(pregunta, history):
    retriever = chroma_local.as_retriever()

    chain = create_stuff_documents_chain(llm, prompt(texto))
    rag = create_retrieval_chain(retriever, chain)
    
    results = rag.invoke({"input": pregunta})
    return results['answer']