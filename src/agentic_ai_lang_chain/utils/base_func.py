import pandas as pd
from sqlalchemy import create_engine
import os
from typing import Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def read_transform(*, file_path:str, db_name: str) -> pd.DataFrame:
    if f"{db_name}_database.db" in os.listdir(file_path):
        return create_engine(f'sqlite:///data_base/{db_name}_database.db')
    else:
        print('Starting to read and transform the data...')
        file_path = os.path.join(file_path, f"{db_name}.xlsx")
        df = pd.read_excel(file_path, sheet_name = None)
        df = pd.concat(df.values(), ignore_index=True)

        engine = create_engine(f'sqlite:///data_base/{db_name}_database.db')
        df.to_sql(f'{db_name}_table', con=engine, if_exists='replace', index=False)
        
        print('Transformation complete.')
        return engine

llm = ChatOllama(
    model="llama3.1",
    temperature=0.5,  # deterministic for classification
)


class AgentState(TypedDict):
    question: str
    router1: Optional[Literal["sql", "general"]]
    db_name: str
    answer_sql: Optional[str]
    answer_general: Optional[str]

def start_node(state: AgentState) -> AgentState:
    question = input("Please ask your question(General of SQL): if it is SQL related, please includ the database name.\n")

    state["question"] = question
    return state

def classify_question_node(state: AgentState) -> AgentState:
    """Decide if the question is SQL-related or general."""

    question = state["question"]

    system_prompt = (
        "You are a strict classifier.\n"
        "You will be given a user question.\n"
        "Respond with exactly one word: SQL or GENERAL.\n"
        "- Respond SQL if the user is asking to query data, filter data, "
        "analyze data, or do something that should map to SQL / database queries.\n"
        "- Otherwise respond GENERAL.\n"
        "No explanation, no extra words."
    )
    # A system message instructing the LLM how to behave.
    # A user message containing the actual question to classify.
    messages = [
        ("system", system_prompt),
        ("user", question),
    ]

    response = llm.invoke(messages)
    decision = response.content.strip().upper()

    if "SQL" in decision:
        state["router1"] = "sql"
        # do not set answer_sql here; the SQL node will answer_sql later
    else:
        state["router1"] = "general"
    return state


def router1_decision(state: AgentState) -> str:
 
    if state.get("router1") == "sql":
        return "sql_query"
    # default: reject → end of graph
    return "general_node"

def find_db_name(state: str) -> str:
    system_prampt = (
        '''
            You are a STRICT table-name classifier.
            Valid table names:
            - online_retail_ii
            - rfm_score

            Rules:
            - Look at the user's question.
            - If it contains EXACTLY one of the valid table names above, output that name.
            - Otherwise output: NONE.
            - Do NOT guess or match by meaning.
            - If the question mentions any unknown table (e.g., student, orders, users), output: NONE.
            - Output must be EXACTLY one token: online_retail_II, RFM_score, or NONE.
            - No explanations or extra text.'''
    )
    messages = [
        ("system", system_prampt),
        ("user", state["question"]),
    ]

    response = llm.invoke(messages)
    decision = response.content.strip().lower()

    if decision == "none":
        print("\n****Database name not found, asking question again****\n")
        state["db_name"] = "ASK_AGAIN"
    else :
        state["db_name"] = decision
    return state


def sql_node(state: AgentState) -> AgentState:
    
    path = os.getcwd()
    path = os.path.abspath(
        os.path.join(path, 'data_base')
    )

    engine = read_transform(file_path=path, db_name=state["db_name"])
    db = SQLDatabase(engine)

    db_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=db,
        verbose=True,  # so you can see SQL it generates
    )
    system_prampt = (
        '''
            You are a SQL generator.

            Your job:
            Given a user question and the database schema, produce ONLY a valid SQL query.

            Rules:
            - Output ONLY the SQL query.
            - Do NOT include explanations, comments, or markdown.
            - Do NOT use backticks or code fences.
            - Do NOT talk to the user.
            - Do NOT add natural language.
            - The output must be executable SQL only.
            - Please add _table suffix to the table name in the query.

            If the request cannot be answered with SQL, output ONLY:
            SELECT 'ERROR';'''
    )
    messages = [
        ("system", system_prampt),
        ("user", state["question"]),
    ]

    response = llm.invoke(messages)
    answer_sql = db.run(response.content.strip())
    # answer_sql = db_chain.run(response.content.strip())
    # answer_sql = db_chain.invoke({"query": response.content.strip()})
    state["answer_sql"] = answer_sql
    return state


def general_node(state: AgentState) -> AgentState:
    path_root = os.getcwd()
    
    CHROMA_DIR = os.path.abspath(
        os.path.join(path_root, 'data_base', 'chroma_pdf_db')
    )
    EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # 1. Embeddings model (must match what you used for indexing)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # 2. Load existing Chroma vector DB from disk
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}  # top-k chunks to retrieve
    )
    question = state["question"]

    # 1. Retrieve relevant chunks from the vector DB
    docs = retriever.invoke(question)

    if not docs:
        state["answer"] = (
            "I couldn't find anything in the documents related to your question."
        )
        return state

    context_parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown_source")
        page = d.metadata.get("page", "unknown_page")
        snippet = d.page_content
        context_parts.append(
            f"Source: {src}, page: {page}\n{snippet}"
        )

    context_text = "\n\n---\n\n".join(context_parts)

    # 3. Define a RAG-style system prompt
    system_prompt = """
        You are a helpful assistant that answers questions using ONLY the provided context.

        Rules:
        - Use the context to answer the question as accurately as possible.
        - If the answer is not in the context, say you don't know.
        - Do NOT invent facts that are not supported by the context.
        - If relevant, you may mention which source/page you are drawing from.
    """

    # 4. Build messages for the LLM
    messages = [
        ("system", system_prompt),
        (
            "user",
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}"
        ),
    ]
    response = llm.invoke(messages)
    answer_text = response.content.strip()

    state["answer_general"] = answer_text
    return state

def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("start_node", start_node)
    graph.add_node("classify_question", classify_question_node)
    graph.add_node("sql_node", sql_node)
    graph.add_node("general_node", general_node)
    graph.add_node("find_db_name", find_db_name)

    # Set entry point
    # The first node in the graph — the node where execution starts.
    # Every LangGraph must have exactly one starting node,
    # otherwise it wouldn’t know where to begin.

    graph.set_entry_point("start_node")
    graph.add_edge("start_node","classify_question")  

    # Add conditional edges from classifier
    # input1: This is the name of the node the condition applies to.
    # input2: This is the function that decides the next node.
    # input3: This is a mapping from possible return values of the function to next nodes
    graph.add_conditional_edges(
        "classify_question",
        router1_decision,
        # mapping from string returned by router1_decision to next node
        # if router1_decision returns "sql", go to "sql_node"
        # if router1_decision returns "general", go to "general_node"
        {
            "sql_query": "find_db_name",
            "general_node": "general_node",
        },
    )
    graph.add_conditional_edges(
        "find_db_name",
        lambda state: state["db_name"] if state["db_name"] == "ASK_AGAIN" else "db_name_found"
        ,
        {
            "ASK_AGAIN": "start_node",
            "db_name_found": "sql_node",
        }
    )

    # Once SQL node finishes, we end the graph
    # After running the node "sql_node", the graph execution should end
    graph.add_edge("sql_node", END)
    graph.add_edge("general_node", END)

    return graph.compile()