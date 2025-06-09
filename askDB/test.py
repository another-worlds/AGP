import os
import re
import sqlalchemy
import pandas as pd
from typing import Dict
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import SequentialChain, LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import PydanticToolsParser

# ── Setup ───────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(temperature=0)

# ── YOUR EXISTING TOOLS ─────────────────────────────────────────────────────────
def zone_name_lookup(code: int) -> str:
    df = pd.read_excel("order_505_sub.xlsx")
    return df[df["code"] == code]["name"].values[0]

def fetch_zone_code(zone_id: int) -> int:
    engine = sqlalchemy.create_engine("sqlite:///pdp_funczone.sqlite")
    with engine.connect() as conn:
        row = conn.execute(
            sqlalchemy.text(
                "SELECT pdp_func_zone_code FROM "
                "pdp_functional_zones__default_workspacepdp_functional_zones "
                "WHERE id = :id"
            ), {"id": zone_id}
        ).fetchone()
    return row[0] if row else None

# ── 1) BASE RESPONSE ────────────────────────────────────────────────────────────
base_template = """User asked: "{query}"
Provide a quick “base” answer in plain English, e.g.:
"The zone code for id 2727049 is 12345 and the zone name is FooZone."
"""
base_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(input_variables=["query"], template=base_template),
    output_key="base_response",
)

# ── 2) PLAN VERIFICATION QUESTIONS ──────────────────────────────────────────────
plan_template = """
Given this base answer:

{base_response}

Extract each factual claim and turn it into a verification question.
E.g. if the response says “zone code is 12345”, you should ask:
“What is the zone code for that id?”

Return JSON as:
{{
  "facts_and_verification_questions": {{
    "<fact1>": "<question1>",
    "<fact2>": "<question2>"
  }}
}}
"""
class PlanOutput(BaseModel):
    facts_and_verification_questions: Dict[str, str]

plan_parser = PydanticOutputParser(pydantic_object=PlanOutput)
plan_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
      input_variables=["base_response"],
      template=plan_template + "\n\n" + plan_parser.get_format_instructions()
    ),
    output_key="plan",
    output_parser=plan_parser,
)

# ── 3) RUN CODE-BASED VERIFICATION ───────────────────────────────────────────────
def run_verification(plan: PlanOutput, zone_id: int) -> Dict[str, str]:
    verified: Dict[str,str] = {}
    for fact, question in plan.facts_and_verification_questions.items():
        if "zone code" in question.lower():
            code = fetch_zone_code(zone_id)
            verified[question] = str(code or "Not found")
        elif "zone name" in question.lower():
            code = fetch_zone_code(zone_id)
            name = zone_name_lookup(code) if code else "Not found"
            verified[question] = name
        else:
            verified[question] = "N/A"
    return verified

# ── 4) FINAL ANSWER ─────────────────────────────────────────────────────────────
final_template = """
Original question: {query}

Original (possibly hallucinated) answer:
{base_response}

Verified facts:
{verify_results}

Please give me a clean final answer that only includes the verified information.
"""
final_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
      input_variables=["query","base_response","verify_results"],
      template=final_template,
    ),
    output_key="final_answer",
)

# ── WRAPPER FUNCTION ────────────────────────────────────────────────────────────
def lookup_zone(query: str) -> str:
    # 1) extract numeric ID
    match = re.search(r"\b(\d+)\b", query)
    if not match:
        return "Invalid ID."
    zone_id = int(match.group(1))

    # 2) run the three-stage chain
    base = base_chain.run(query=query)
    plan = plan_chain.run(base_response=base)
    verified = run_verification(plan, zone_id)
    verify_str = "\n".join(f"{q} → {a}" for q, a in verified.items())
    final = final_chain.run(
        query=query,
        base_response=base,
        verify_results=verify_str
    )
    return final

# ── Example ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(lookup_zone("What is the zone name of id 2727049?"))