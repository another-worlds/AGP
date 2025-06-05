from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from pprint import pprint
from langchain.agents import initialize_agent, AgentType


db = SQLDatabase.from_uri(
    "sqlite:///pdp_funczone.sqlite",
)
llm = ChatOpenAI(temperature=0)

# Create a prompt to load table names and and column names to database
prompt = PromptTemplate.from_template(
    "You are a helpful assistant that provides SQL queries based on user requests. "
    "Search for the table names and columns in the database: {table_context}. "
    "When asked for a query, you will return a valid SQL query that answers the user's question."
)
prompt.invoke({"table_context": db.get_context()})

def zone_name_lookup(query):
    import pandas as pd
    df = pd.read_excel("order_505_sub.xlsx")
    # Simplified filter logic
    zone_name = df[df["code"] == int(query)]["name"].values[0]
    return zone_name

def init_sql_tool():    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

    sql_agent_tool = Tool(
        name="SQL Agent",
        func=sql_agent_executor.run,
        description="Use this tool to run SQL queries against the table pdp_functional_zones__default_workspacepdp_functional_zones of a database to find pdp_func_zone_code WHERE ID is  supplied by user ",
    )
    return sql_agent_tool

def init_multi_agent():
    sql_agent_tool = init_sql_tool()
    
    lookup_tool = Tool(
    name="Zone name lookup",
    func=zone_name_lookup,
    description="Look up zone name by zone code after you god zone code from database",
    )

    tools = [lookup_tool, sql_agent_tool]

    multi_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return multi_agent

def run_multi_agent(query: str):
    multi_agent = init_multi_agent()
    return multi_agent.run(query)


# if __name__ == '__main__':
#     print(run_multi_agent("What is the name of zone by id of 2726883"))