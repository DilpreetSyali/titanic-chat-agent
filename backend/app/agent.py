from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser

from .tools import pandas_query, pandas_plot


SYSTEM = """You are a friendly Titanic dataset analyst.
You have access to tools that can run pandas code and generate matplotlib plots.

Rules:
- If the user asks for a chart/plot/histogram/bar chart/visualization, use pandas_plot.
- Otherwise use pandas_query.
- For pandas_query: write python code that sets `result` to a final human-friendly string.
  - If percentage → include % and round to 2 decimals.
  - If fare or average → round to 2 decimals.
  - If count → include context like "Total passengers: 891".
- For pandas_plot: create plots using matplotlib via plt.
- Do NOT import anything.
- Do NOT use markdown fences.
"""


def build_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )


def df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split")


def run_agent(question: str, df: pd.DataFrame):

    llm = build_llm()
    df_json = df_to_json(df)

    # decide TEXT vs PLOT
    router = (
        ChatPromptTemplate.from_messages(
            [
                SystemMessage(content="Return ONLY one word: PLOT or TEXT"),
                ("human", "{q}")
            ]
        )
        | llm
        | StrOutputParser()
    )

    mode = router.invoke({"q": question}).strip().upper()

    if "PLOT" in mode:

        plot_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=SYSTEM),
                    (
                        "human",
                        """Write matplotlib code using df.

Rules:
- plt and df already exist
- do NOT import anything
- drop missing values if needed
- add title and labels
- return ONLY code

Question: {q}
"""
                    ),
                ]
            )
            | llm
            | StrOutputParser()
        )

        code = plot_prompt.invoke({"q": question})

        img_b64 = pandas_plot.invoke(
            {
                "code": code,
                "df_json": df_json
            }
        )

        caption_prompt = (
            ChatPromptTemplate.from_messages(
                [
                    SystemMessage(
                        content="Write a short 1 sentence caption describing the chart. Plain text only."
                    ),
                    ("human", "{q}")
                ]
            )
            | llm
            | StrOutputParser()
        )

        caption = caption_prompt.invoke({"q": question}).strip()

        return {
            "answer": caption,
            "image_b64": img_b64
        }

    # TEXT MODE

    text_prompt = (
        ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM),
                (
                    "human",
                    """Write pandas code using df.

Rules:
- do NOT import anything
- set variable result to final string answer
- return ONLY code

Question: {q}
"""
                ),
            ]
        )
        | llm
        | StrOutputParser()
    )

    code = text_prompt.invoke({"q": question})

    answer = pandas_query.invoke(
        {
            "code": code,
            "df_json": df_json
        }
    )

    return {
        "answer": answer,
        "image_b64": None
    }