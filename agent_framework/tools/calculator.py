from __future__ import annotations

import ast
import operator as op
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from session.models import Session

# Whitelisted binary operators
_BIN_OPS: dict[type[ast.AST], Callable[[int | float, int | float], int | float]] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

# Whitelisted unary operators
_UNARY_OPS: dict[type[ast.AST], Callable[[int | float], int | float]] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError(f"disallowed expression: {ast.dump(node)}")


class Calculator:
    """Arithmetic eval via ast whitelist. No eval/exec/calls/attr-access."""

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "Evaluate arithmetic expressions (+ - * / // % ** and parens). "
        "No function calls or attribute access."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "expr": {
                "type": "string",
                "description": "arithmetic expression, e.g. (1+2)*3",
            }
        },
        "required": ["expr"],
    }

    async def run(self, args: dict[str, Any], session: Session) -> str:
        expr = args.get("expr")
        if not isinstance(expr, str):
            return "ERROR: expr must be a string"
        try:
            tree = ast.parse(expr, mode="eval")
            return str(_eval(tree))
        except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as e:
            return f"ERROR: {e}"
