# run.py (Final Version)

import os
from flask import Flask
from operator import itemgetter
from langchain_core.runnables import RunnableParallel

# Import from LangChain and our blueprints
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from src.scribe_views import scribe_bp
from src.alkharizmi_views import alkharizmi_bp

def create_app():
    # Use src/templates as the default template folder
    app = Flask(__name__, template_folder='src/templates')

    # --- Configuration ---
    project_root = os.path.abspath(os.path.dirname(__file__))
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
    app.config['CHROMA_PATH'] = os.path.join(project_root, 'chroma')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # --- Initialize RAG Components Globally for the App ---
    print("Initializing RAG components...")
    embeddings = OllamaEmbeddings(model="llama3")
    db = Chroma(persist_directory=app.config['CHROMA_PATH'], embedding_function=embeddings)
    
    app.retriever = db.as_retriever(search_kwargs={"k": 3})

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = OllamaLLM(model="llama3")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The final, corrected RAG Chain
    rag_chain_runnable = RunnableParallel({
        "answer": (
            {"context": lambda x: format_docs(x["context"]), "question": itemgetter("question")}
            | prompt
            | llm
            | StrOutputParser()
        ),
        "context": itemgetter("context"),
    })
    
    app.rag_chain = rag_chain_runnable
    
    print("RAG components ready.")

    # --- Register Blueprints ---
    app.register_blueprint(scribe_bp)
    app.register_blueprint(alkharizmi_bp)

    return app

if __name__ == '__main__':
    app = create_app()
    # A single port for the unified app
    app.run(host='0.0.0.0', port=5000, debug=True)