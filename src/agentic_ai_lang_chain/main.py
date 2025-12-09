from utils.base_func import read_transform, AgentState, build_graph
import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Optional


app = build_graph()

question = input("Please ask your SQL-related question: ")

state_in1: AgentState = {
    "question": question,
    "route": None,
    "answer": None,
}
result = app.invoke(state_in1)
print("GENERAL QUESTION RESULT:")
print(result["answer"])
print("-" * 50)

question = input("Please ask your SQL-related question: ")

# Example 2: SQL-related question
state_in2: AgentState = {
    "question": question,
    "route": None,
    "answer": None,
}
result2 = app.invoke(state_in2)
print("SQL QUESTION RESULT:")
print(result2["answer"])