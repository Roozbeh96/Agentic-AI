from utils.base_func import read_transform, AgentState, build_graph
import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Optional


app = build_graph()

while True:
    state_in1: AgentState = {
        "question": "",
        "router1": None,
        "db_name": "",
        "answer_sql": None,
        "answer_general": None
    }
    result = app.invoke(state_in1)
    print(result["answer_sql"] or result["answer_general"])
    print('*'*75)
