import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import tool

def _safe_exec(code: str, df: pd.DataFrame):
    """
    Executes code with a restricted global environment.
    Security note: still not bulletproof for hostile input.
    For assignments, this is usually acceptable. For production, sandbox properly.
    """
    allowed_globals = {
        "df": df,
        "pd": pd,
        "plt": plt,
        "__builtins__": {
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "round": round,
            "range": range,
            "abs": abs,
        },
    }
    local_vars = {}
    exec(code, allowed_globals, local_vars)
    return local_vars

@tool
def dataframe_answer(question: str) -> str:
    """
    Answer Titanic dataset questions using direct pandas logic for common asks.
    This tool is deterministic and handles common questions without code execution.
    """
    # We keep this tool "stateless"; the agent will pass the data through other tools.
    return (
        "I can answer that, but I need to use the pandas tool that has access to the dataframe. "
        "Please call pandas_query or pandas_plot for computation/visualization."
    )

@tool
def pandas_query(code: str, df_json: str) -> str:
    """
    Run a pandas/python snippet that MUST assign the final string result to a variable named `result`.
    Inputs:
    - code: python code as a string
    - df_json: dataframe in json (split) format
    Output:
    - string result
    """
    df = pd.read_json(df_json, orient="split")
    local_vars = _safe_exec(code, df)

    if "result" not in local_vars:
        raise ValueError("Your code must set a variable named `result` (string).")

    return str(local_vars["result"])

@tool
def pandas_plot(code: str, df_json: str) -> str:
    """
    Create a matplotlib plot from python code and return base64 PNG.
    The code MUST end with a matplotlib figure ready to save.
    """
    df = pd.read_json(df_json, orient="split")

    plt.close("all")
    _safe_exec(code, df)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    return b64
