from langchain_community.agent_toolkits.sql.base import create_sql_agent
#from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from toolkit_helper import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from pprint import pprint
from langchain.agents import initialize_agent, AgentType
import pandas as pd


db = SQLDatabase.from_uri(
    "sqlite:///./pdp_funczone.sqlite",
)
llm = ChatOpenAI(temperature=0)

def zone_name_lookup_tool(query):
    try:
        df = pd.read_excel("./order_505_sub.xlsx")
        # Simplified filter logic
        zone_name = df[df["code"] == int(query)]["name"].values[0]
        return zone_name
    except Exception as e:
        return e

def dont_know_tool(query):
    return "I don't know"

def init_sql_tool():    
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)
    for i, tool in enumerate(sql_agent_executor.tools):
        print("\n")
        print(i)
        print(tool)
    #sql_agent_executor.tools = [sql_agent_executor.tools[0], sql_agent_executor.tools[2], sql_agent_executor.tools[3]]
    

    sql_agent_tool = Tool(
        name="SQL Agent",
        func=sql_agent_executor.run,
        description="Use tool to run SQL queries against the table pdp_functional_zones__default_workspacepdp_functional_zones of a database TO FIND FIELD pdp_func_zone_code WHERE ID is supplied by user.",
    )
    return sql_agent_tool

def init_multi_agent():
    sql_agent_tool = init_sql_tool()
    #sql_agent_tool.handle_tool_error = True
    
    lookup_tool = Tool(
    name="Zone name lookup tool",
    func=zone_name_lookup_tool,
    description="Use this tool to look up zone name by zone code after you got zone code from database.",
    )
    dont_tool = Tool(
        name="Don't know tool",
        func=dont_know_tool,
        description="Use this tool, when you don'y know what to di"
    )

    tools = [lookup_tool, sql_agent_tool, dont_tool]

    multi_agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        
        
    )
    #multi_agent.handle_parsing_errors = True
    return multi_agent


def run_multi_agent(query: str):
    multi_agent = init_multi_agent()
    multi_agent.handle_parsing_errors = True
    try:
        response = multi_agent.run(query)
    except ValueError as e:
        print(e)
        return "Not found."
    return response

if __name__ == '__main__':
    #print(run_multi_agent("What is the name of zone by     id of 2726883"))
    
    run_multi_agent("What is the zone name of id 2727049")
