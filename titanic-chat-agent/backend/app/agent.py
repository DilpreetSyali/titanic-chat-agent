import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

from .tools import pandas_query, pandas_plot

SYSTEM = """You are a friendly Titanic dataset analyst.
You have access to tools that can run pandas code and generate matplotlib plots.

Rules:
- If user asks for a chart/plot/histogram/bar chart/visualization, use pandas_plot.
- Otherwise use pandas_query.
- For pandas_query, produce python code that sets `result` to a helpful, clear text answer.
- For pandas_plot, produce python code that creates the requested plot using matplotlib (plt).
- Always be concise, correct, and user-friendly.
- Use df columns as present in the dataset (common: Survived, Pclass, Sex, Age, Fare, Embarked, etc.)
"""

def build_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)

def _df_to_json(df: pd.DataFrame) -> str:
    # split format is efficient and stable
    return df.to_json(orient="split")

def run_agent(question: str, df: pd.DataFrame):
    llm = build_llm()
    df_json = _df_to_json(df)

    # Simple classifier prompt: decide "plot" vs "text"
    router = ChatPromptTemplate.from_messages([
        SystemMessage(content="Return ONLY one word: PLOT or TEXT."),
        ("human", "User question: {q}")
    ]) | llm | StrOutputParser()

    mode = (router.invoke({"q": question}) or "").strip().upper()
    wants_plot = ("PLOT" in mode)

    if wants_plot:
        # Ask LLM to write plotting code
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=SYSTEM),
            ("human", """Write python matplotlib code to plot based on the user request.

Data is in a pandas dataframe named df.
Return ONLY code (no markdown fences).

User request: {q}
""")
        ])
        code = (prompt | llm | StrOutputParser()).invoke({"q": question})

        b64 = pandas_plot.invoke({"code": code, "df_json": df_json})
        # Also provide a short caption
        caption_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Write a 1-2 sentence caption for the chart."),
            ("human", "User request: {q}")
        ])
        caption = (caption_prompt | llm | StrOutputParser()).invoke({"q": question}).strip()
        return {"answer": caption, "image_b64": b64}

    # TEXT path
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=SYSTEM),
        ("human", """Write python pandas code to answer the question.
Data is in df. Set `result` to a final string answer.
Return ONLY code (no markdown fences).

User question: {q}
""")
    ])
    code = (prompt | llm | StrOutputParser()).invoke({"q": question})

    text = pandas_query.invoke({"code": code, "df_json": df_json})
    return {"answer": text, "image_b64": None}
