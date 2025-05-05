# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing utilities for evaluation and compilation."""

import ast
import builtins
import inspect
import symtable

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.base import Base

__all__ = [
    "evaluate",
]


def evaluate(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate one or multiple lines of Python code and return the result of the final expression.

    Args:
        expr (str): Expression string.

            Must contain valid Python code and can be single-line or multi-line.
        context (KwargsLike): Dictionary representing the execution context.

    Returns:
        Any: Result of evaluating the final expression.
    """
    expr = inspect.cleandoc(expr)
    if context is None:
        context = {}
    if "\n" in expr:
        tree = ast.parse(expr)
        eval_expr = ast.Expression(tree.body[-1].value)
        exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
        exec(compile(exec_expr, "<string>", "exec"), context)
        return eval(compile(eval_expr, "<string>", "eval"), context)
    return eval(compile(expr, "<string>", "eval"), context)


def get_symbols(table: symtable.SymbolTable) -> tp.List[symtable.Symbol]:
    """Recursively retrieve all symbols from a symbol table.

    Args:
        table (symtable.SymbolTable): Symbol table to traverse.

    Returns:
        List[symtable.Symbol]: A list of symbols found in the table and its child tables.
    """
    symbols = []
    children = {child.get_name(): child for child in table.get_children()}
    for symbol in table.get_symbols():
        if symbol.is_namespace():
            symbols.extend(get_symbols(children[symbol.get_name()]))
        else:
            symbols.append(symbol)
    return symbols


def get_free_vars(expr: str) -> tp.List[str]:
    """Parse the provided code and return free variable names, excluding built-in names.

    Args:
        expr (str): Expression string.

            Must contain valid Python code and can be single-line or multi-line.

    Returns:
        List[str]: A list of free variable names found in the code.
    """
    expr = inspect.cleandoc(expr)
    global_table = symtable.symtable(expr, "<string>", "exec")
    symbols = get_symbols(global_table)
    builtins_set = set(dir(builtins))
    free_vars = []
    free_vars_set = set()
    not_free_vars_set = set()
    for symbol in symbols:
        symbol_name = symbol.get_name()
        if symbol.is_imported() or symbol.is_parameter() or symbol.is_assigned() or symbol_name in builtins_set:
            not_free_vars_set.add(symbol_name)
    for symbol in symbols:
        symbol_name = symbol.get_name()
        if symbol_name not in not_free_vars_set and symbol_name not in free_vars_set:
            free_vars.append(symbol_name)
            free_vars_set.add(symbol_name)
    return free_vars


class Evaluable(Base):
    """Abstract class for objects that can be evaluated.

    This class provides an interface to check whether an instance's evaluation id meets a given evaluation id.
    """

    def meets_eval_id(self, eval_id: tp.Optional[tp.Hashable]) -> bool:
        """Return whether the instance's evaluation id matches the provided evaluation id.

        Args:
            eval_id (Optional[Hashable]): Evaluation identifier.

        Returns:
            bool: True if the instance's evaluation id satisfies the given evaluation id, False otherwise.
        """
        if self.eval_id is not None and eval_id is not None:
            if checks.is_complex_sequence(self.eval_id):
                if eval_id not in self.eval_id:
                    return False
            else:
                if eval_id != self.eval_id:
                    return False
        return True
