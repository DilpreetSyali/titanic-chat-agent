import base64
import io
from io import StringIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from langchain_core.tools import tool


# Allow only these modules to be imported inside tool-executed code
_ALLOWED_IMPORTS = {
    "pandas": pd,
    "numpy": np,
    "matplotlib": __import__("matplotlib"),
    "matplotlib.pyplot": plt,
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Restricted __import__ allowing only a small allowlist of safe modules."""
    if name in _ALLOWED_IMPORTS:
        return _ALLOWED_IMPORTS[name]

    # Common patterns:
    # - import matplotlib
    # - import matplotlib.pyplot as plt
    # - from matplotlib import pyplot
    if name == "matplotlib.pyplot":
        return _ALLOWED_IMPORTS["matplotlib.pyplot"]

    if name.startswith("matplotlib"):
        return _ALLOWED_IMPORTS["matplotlib"]

    raise ImportError(f"Import of '{name}' is not allowed.")


def _safe_exec(code: str, df: pd.DataFrame):
    """
    Execute model-generated python in a restricted environment.
    Provides: df, pd, np, plt
    Allows only a small set of safe builtins + restricted imports.
    """
    allowed_globals = {
        "df": df,
        "pd": pd,
        "np": np,
        "plt": plt,
        "__builtins__": {
            # safe builtins
            "len": len,
            "min": min,
            "max": max,
            "sum": sum,
            "sorted": sorted,
            "round": round,
            "range": range,
            "abs": abs,
            "map": map,
            "filter": filter,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "float": float,
            "int": int,
            "str": str,
            # restricted import
            "__import__": _safe_import,
        },
    }

    local_vars = {}
    exec(code, allowed_globals, local_vars)
    return local_vars


@tool
def pandas_query(code: str, df_json: str) -> str:
    """Run a pandas/python snippet over the Titanic dataframe and return a text answer.
    Code MUST set `result`.
    """
    df = pd.read_json(StringIO(df_json), orient="split")

    local_vars = _safe_exec(code, df)

    if "result" not in local_vars:
        raise ValueError("Code must set a variable named `result`.")

    return str(local_vars["result"])


@tool
def pandas_plot(code: str, df_json: str) -> str:
    """Run matplotlib code and return the resulting figure as base64 PNG."""
    df = pd.read_json(StringIO(df_json), orient="split")

    # Reset matplotlib state for consistent output
    plt.close("all")
    plt.figure()
    plt.gca()

    _safe_exec(code, df)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")