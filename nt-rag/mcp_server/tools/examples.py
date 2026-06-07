"""Example non-RAG tools to demonstrate the MCP protocol."""

from __future__ import annotations

import ast
import operator
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config


def _list_doc_paths(docs_dir: Path) -> list[Path]:
    if not docs_dir.is_dir():
        return []
    paths: list[Path] = []
    for path in sorted(docs_dir.iterdir()):
        name = path.name
        if name.startswith(".") or name.startswith("~$"):
            continue
        if path.suffix.lower() in (".pdf", ".docx", ".doc"):
            paths.append(path)
    return paths


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_ast(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_ast(node.left), _eval_ast(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_ast(node.operand))
    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


def calculate(expression: str) -> dict:
    """Evaluate a simple arithmetic expression, e.g. '(30+1)*19'."""
    expr = expression.strip()
    if not expr:
        return {"expression": expression, "error": "Empty expression"}
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_ast(tree)
        if result == int(result):
            result = int(result)
        return {"expression": expr, "result": result}
    except Exception as exc:
        return {"expression": expr, "error": str(exc)}


def get_current_time() -> dict:
    """Return the current UTC time."""
    now = datetime.now(timezone.utc)
    return {
        "iso": now.isoformat(),
        "formatted": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


def count_words(text: str) -> dict:
    """Count words in a text string."""
    words = [w for w in text.split() if w.strip()]
    return {"word_count": len(words), "char_count": len(text)}


def echo_message(message: str) -> dict:
    """Echo a message back (demo tool)."""
    return {"echo": message}


def list_document_sources() -> dict:
    """List PDF/DOCX files available in docs/."""
    paths = _list_doc_paths(config.DOCS_DIR)
    return {
        "docs_dir": str(config.DOCS_DIR),
        "files": [p.name for p in paths],
        "count": len(paths),
    }
