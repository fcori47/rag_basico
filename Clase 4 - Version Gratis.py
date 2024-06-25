import warnings

warnings.filterwarnings('ignore')

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


llm = ChatOllama(model='llama3', temperature=0)
chroma_local = Chroma(persist_directory="./vectordb_gratis", embedding_function=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))
    

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


def respuesta(pregunta, llm, chroma_db, prompt):
    retriever = chroma_db.as_retriever()

    chain = create_stuff_documents_chain(llm, prompt)
    rag = create_retrieval_chain(retriever, chain)
    
    results = rag.invoke({"input": pregunta})
    return results



texto = """Tú eres un asistente para tareas de respuesta a preguntas."
    "Usa los siguientes fragmentos de contexto recuperado para responder "
    "la pregunta. Si no sabes la respuesta, di que no "
    "sabes. Usa un máximo de tres oraciones y mantén la "
    "respuesta concisa."""
    
    

if __name__ == '__main__':
    print(respuesta(input('Hacé tu pregunta: '), llm, chroma_local, prompt(texto))['answer'])