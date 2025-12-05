from utils.base_func import read_transform
import os
import pandas as pd
from langchain_ollama import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langgraph.graph import StateGraph, END
from typing import Literal, TypedDict, Optional


path = os.getcwd()
path = os.path.abspath(
    os.path.join(path, 'data')
)

engine = read_transform(file_path=path)
test_df = pd.read_sql('SELECT * FROM online_retail_table LIMIT 5', con=engine)


db = SQLDatabase(engine)
tables = db.get_usable_table_names()

llm = ChatOllama(
    model="llama3.1",
    temperature=0.0,  # deterministic SQL
)



db_chain = SQLDatabaseChain.from_llm(
    llm=llm,
    db=db,
    verbose=True,  # so you can see SQL it generates
)

question = "How many rows are there in my_table?"
answer = db_chain.run(question)
print(answer)