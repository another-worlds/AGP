from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import Agent

# load env
load_dotenv()
# init chat
llm = OpenAI(temperature=0.5)
# init agent
#agent = Agent(llm=llm)
# init prompt
prompt = PromptTemplate.from_template("""
    You are a not helpful assistant. Answer questions with devious mischief.
    {input}
    """
    )
#init chain
chain = prompt | llm
# test
answer = chain.invoke({"input":"Hello!"})
print(answer)
while True:
    user_input = input()
    answer = chain.invoke({"input":user_input})
    print(answer)
