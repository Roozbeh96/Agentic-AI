import pandas as pd
from sqlalchemy import create_engine
import os
from typing import Literal, TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent, SQLDatabaseToolkit





def read_transform(*, file_path:str) -> pd.DataFrame:
    if 'online_retail_database.db' in os.listdir(file_path):
        return create_engine('sqlite:///data/online_retail_database.db')
    else:
        print('Starting to read and transform the data...')
        file_path = os.path.join(file_path, 'online_retail_II.xlsx')
        df = pd.read_excel(file_path, sheet_name = None)
        df = pd.concat(df.values(), ignore_index=True)

        engine = create_engine('sqlite:///data/online_retail_database.db')
        df.to_sql('online_retail_table', con=engine, if_exists='replace', index=False)
        
        print('Transformation complete.')
        return engine

llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,  # deterministic for classification
)

FIXED_REJECTION_MESSAGE = (
    "I can not answer to this question right now. "
    "Maybe in future updates I will be able of answering your question."
)

path = os.getcwd()
path = os.path.abspath(
    os.path.join(path, 'data')
)

engine = read_transform(file_path=path)
db = SQLDatabase(engine)

class AgentState(TypedDict):
    question: str
    route: Optional[Literal["sql", "reject"]]
    answer: Optional[str]



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
        state["route"] = "sql"
        # do not set answer here; the SQL node will answer later
    else:
        state["route"] = "reject"
        state["answer"] = FIXED_REJECTION_MESSAGE

    return state

def sql_node(state: AgentState) -> AgentState:
    """
    Placeholder: here you will later plug in your text-to-SQL logic
    (LangChain SQLDatabaseChain or your own agent).
    For now, we just return a dummy message.
    """

    question = state["question"]
    # db_chain = SQLDatabaseChain.from_llm(
    #     llm=llm,
    #     db=db,
    #     verbose=True,  # so you can see SQL it generates
    # )
    # answer = db_chain.run(question)
    # state["answer"] = answer

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
    result = agent.invoke({"input": question})
    state["answer"] = result
    return state



def route_decision(state: AgentState) -> str:
    """
    Decide where to go next based on state['route'].
    Must return the name of the next node or END.
    """
    if state.get("route") == "sql":
        return "sql_node"
    # default: reject → end of graph
    return END



def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_question", classify_question_node)
    graph.add_node("sql_node", sql_node)

    # Set entry point
    # The first node in the graph — the node where execution starts.
    # Every LangGraph must have exactly one starting node,
    # otherwise it wouldn’t know where to begin.
    graph.set_entry_point("classify_question")

    # Add conditional edges from classifier
    # input1: This is the name of the node the condition applies to.
    # input2: This is the function that decides the next node.
    # input3: This is a mapping from possible return values of the function to next nodes
    graph.add_conditional_edges(
        "classify_question",
        route_decision,
        # mapping from string returned by route_decision to next node
        # if route_decision returns "sql", go to "sql_node"
        # if route_decision returns END, go to END
        {
            "sql_node": "sql_node",
            END: END,
        },
    )

    # Once SQL node finishes, we end the graph
    # After running the node "sql_node", the graph execution should end
    graph.add_edge("sql_node", END)

    return graph.compile()