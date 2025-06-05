from dotenv import load_dotenv
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
# t = SystemMessagePromptTemplate.from_template("rfrf")
# print(t.prompt.template)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers import SelfQueryRetriever
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor
import chromadb

import os

from langchain_community.vectorstores.chroma import Chroma

from langchain.document_loaders.pdf import  PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

from prompt_helper import template
# markdown/sentence splitters
# ParentDocumentRetriever - cascade split chunks
# metadata?

load_dotenv()
dir = os.getcwd()
docs_path = os.path.join(dir, "docs_cut.pdf")
docs_test_path = os.path.join("pdf_one_pager.pdf")

embeddings  = OpenAIEmbeddings()

metadata_field_info = [
    AttributeInfo(
        name="page_content",
        description="The fragment from the postplatforms whitepaper",
        type='string'
    ),
    AttributeInfo(
        name="page_number",
        description="The number of the page the fragment sources from",
        type='integer'
    )
]

def docs_to_chroma(docs:str):
    loader = PyPDFLoader(file_path=docs_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.split_documents(loader.load())
    if not os.path.exists(os.path.join(dir, "./chroma.db")):
        db = Chroma.from_documents(docs, OpenAIEmbeddings(),persist_directory='./chroma.db')
    else: db = Chroma(persist_directory="./chroma.db", embedding_function=OpenAIEmbeddings())
    return db

    



def docs_to_vectorDB(docs: str) -> FAISS:
    loader = PyPDFLoader(docs_path)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(loader.load())
    db = FAISS.from_documents(docs, embeddings)
    return db

def format_context(docs):
    context = []    
    for doc in docs:
        page_content = doc.page_content.replace("\n", " ")
        page_number = doc.metadata.get('page')
        context.append({"page_content": page_content,
                        "page_number" : page_number})
    
    context = [str(entry) for entry in context]
    formatted_context = "\n".join(context)
    print( 
f"""
----------------------------------
{formatted_context}
----------------------------------
""")
    return formatted_context



def db_to_retrieval_chain(db: FAISS, query:str, k:int = 6):# -> LLMChain:
    docs = db.similarity_search(query,k)
    context = format_context(docs)
    prompt_template = PromptTemplate.from_template(template)
    llm = OpenAI(temperature=0.1)
    chain = prompt_template | llm
    return chain.invoke({"input":query, "context":context})

def db_to_agent_chain(db: Chroma, query:str, k:int = 6):
    chat = ChatOpenAI(temperature=0.5)
    
    retriever = SelfQueryRetriever.from_llm(
        llm=chat,
        vectorstore=db,
        document_contents="Fragments from the postplatforms project.",
        metadata_field_info=metadata_field_info,
    )
    
    
    tool = create_retriever_tool(
        retriever=retriever,
        name="postplatforms_docs_search",
        description="Postplatforms whitepaper search tool to extract relevant context"
    )
    tools = [tool]

    prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt.messages[0].prompt.template=template
    
    agent = create_openai_functions_agent(
        llm=chat,
        prompt=prompt,
        tools=tools
    )
    
    agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True)
    
    docs = retriever.invoke(query)
    context = format_context(docs)
    answer = agent_executor.invoke({"input":query, "context":context})
    
    return answer['output']
    






print(
"""
----------------------------------
ENTERING CHAIN
----------------------------------
""")

#

if __name__ == "__main__":
    mode = 'prod'
    mode = 'agent'
    if mode == 'test':
        db = docs_to_vectorDB(docs_path) 
    elif mode == 'selfQuery' or 'agent':
        db = docs_to_chroma(docs_path)
    
    while True:
        user_input = input()
        if mode == 'test':
            answer = db_to_retrieval_chain(db, user_input)
            print(answer)
        elif mode == 'selfQuery' or 'agent':    
            llm = ChatOpenAI(temperature=0.1)     
            retriever = SelfQueryRetriever.from_llm(
                llm =llm,
                vectorstore = db,
                document_contents = "Fragments from the postplatforms project.",
                metadata_field_info=metadata_field_info,
            )
            docs = retriever.invoke(user_input)

            if mode =='agent':
                tool = create_retriever_tool(
                    retriever=retriever,
                    name='postplatforms_docs_search',
                    description='Postplatforms whitepaper search tool to extract relevant context')
                tools = [tool]
                prompt = hub.pull("hwchase17/openai-functions-agent")
                prompt.messages[0].prompt.template=template
                agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                #agent_executor.invoke({"input": user_input, "context":format_context(docs)})
    
                answer = agent_executor.invoke({"input": "How are postplatforms better than blockchain", "context":format_context(docs)})
                print(answer['output'])
            elif mode =='selfQuery':
                prompt_template = PromptTemplate.from_template(template)
                chain = prompt_template | llm
                answer = chain.invoke({"input":user_input, "context":format_context(docs)})
                print(answer)
