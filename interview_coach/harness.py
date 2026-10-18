"""Runs the candidate's solution and the coach's reference solution on the same tests.

agent.py launches this as a separate Python process, so an infinite loop or a
crash in the submitted code can't take down adk web. It reads one JSON job from
stdin and prints one JSON line per event, flushing as it goes so the results
that finished still come through if the process gets killed for taking too long.
Standard library only.
"""

import collections
import contextlib
import copy
import io
import itertools
import json
import math
import random
import string
import sys
import threading
import time
import traceback

MARKER = "@@RESULT@@ "
MAX_SHOWN = 300
MAX_NODES = 200_000
REAL_STDOUT = sys.stdout

# What LeetCode's Python environment gives you without imports, plus its node classes.
PRELUDE = """
from typing import *
from collections import *
from heapq import *
from bisect import *
from itertools import *
from functools import *
from math import inf
import collections, heapq, bisect, itertools, functools, math, string, re, random, operator


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""


def emit(**event):
    REAL_STDOUT.write(MARKER + json.dumps(event, default=repr) + "\n")
    REAL_STDOUT.flush()


def shorten(value, is_text=False):
    """Python-style repr (so None, not null), cut down to MAX_SHOWN characters."""
    text = value.rstrip() if is_text else repr(value)
    if len(text) <= MAX_SHOWN:
        return text
    return f"{text[:MAX_SHOWN]}... ({len(text)} chars total)"


def describe_error(exc, filename):
    """e.g. 'IndexError: list index out of range (line 7)', pointing into the submitted code."""
    if isinstance(exc, SyntaxError):
        return f"SyntaxError: {exc.msg} (line {exc.lineno})"
    line = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == filename:
            line = frame.lineno
    where = f" (line {line})" if line else ""
    return f"{type(exc).__name__}: {exc}{where}"


def load(code, filename, function_name):
    """Runs the code and returns (namespace, callable). Accepts LeetCode's class Solution style or a plain function."""
    namespace = {"__name__": "submission"}
    exec(PRELUDE, namespace)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(code, filename, "exec"), namespace)
    solution_class = namespace.get("Solution")
    if isinstance(solution_class, type) and hasattr(solution_class, function_name):
        return namespace, getattr(solution_class(), function_name)
    func = namespace.get(function_name)
    if callable(func):
        return namespace, func
    raise LookupError(f"no function or Solution method named {function_name!r}")


# --- Linked lists and binary trees, in LeetCode's list formats ----------------


def build_linked_list(spec, namespace):
    """[1, 2, 3] -> 1->2->3. {"values": [...], "cycle_pos": k} links the tail back to node k."""
    values, cycle_pos = spec, -1
    if isinstance(spec, dict):
        values, cycle_pos = spec.get("values", []), spec.get("cycle_pos", -1)
    nodes = [namespace["ListNode"](v) for v in values]
    for node, following in zip(nodes, nodes[1:]):
        node.next = following
    if nodes:
        nodes[-1].next = nodes[cycle_pos] if 0 <= cycle_pos < len(nodes) else None
    return nodes[0] if nodes else None


def build_tree(values, namespace):
    """Level-order list with None for gaps, e.g. [3, 9, 20, None, None, 15, 7]."""
    if not values or values[0] is None:
        return None
    tree_node = namespace["TreeNode"]
    root = tree_node(values[0])
    queue = collections.deque([root])
    i = 1
    while queue and i < len(values):
        node = queue.popleft()
        for side in ("left", "right"):
            if i < len(values) and values[i] is not None:
                child = tree_node(values[i])
                setattr(node, side, child)
                queue.append(child)
            i += 1
    return root


def find_in_tree(root, value):
    queue = collections.deque([root])
    while queue:
        node = queue.popleft()
        if node is None:
            continue
        if node.val == value:
            return node
        queue.extend((getattr(node, "left", None), getattr(node, "right", None)))
    return None


def build_args(raw_args, arg_kinds, namespace):
    args = copy.deepcopy(raw_args)
    last_tree = None
    for i, kind in enumerate(arg_kinds[: len(args)]):
        if kind == "linked_list":
            args[i] = build_linked_list(args[i], namespace)
        elif kind == "tree":
            args[i] = last_tree = build_tree(args[i], namespace)
        elif kind == "node_in_tree":
            args[i] = find_in_tree(last_tree, args[i])
    return args


def linked_list_to_list(head):
    out, seen = [], set()
    while head is not None and len(out) < MAX_NODES:
        if id(head) in seen:
            out.append("<cycle>")
            break
        seen.add(id(head))
        out.append(to_plain(head.val))
        head = head.next
    return out


def tree_to_list(root):
    out, queue = [], collections.deque([root])
    while queue and len(out) < MAX_NODES:
        node = queue.popleft()
        if node is None:
            out.append(None)
            continue
        out.append(to_plain(node.val))
        queue.append(getattr(node, "left", None))
        queue.append(getattr(node, "right", None))
    while out and out[-1] is None:
        out.pop()
    return out


# --- Comparing results --------------------------------------------------------


def sort_key(value):
    return json.dumps(value, sort_keys=True, default=repr)


def to_plain(value):
    """Turns a return value into JSON-friendly data so two solutions can be compared."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, collections.deque)):
        return [to_plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((to_plain(v) for v in value), key=sort_key)
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if hasattr(value, "val") and (hasattr(value, "left") or hasattr(value, "right")):
        return tree_to_list(value)
    if hasattr(value, "val") and hasattr(value, "next"):
        return linked_list_to_list(value)
    return repr(value)


def canonical(value, order, top=True):
    """order: 'exact', 'any' (outer list order doesn't matter), or 'any_nested' (no list order matters)."""
    if isinstance(value, list):
        items = [canonical(v, order, top=False) for v in value]
        if order == "any_nested" or (order == "any" and top):
            items.sort(key=sort_key)
        return items
    if isinstance(value, dict):
        return {k: canonical(v, order, top=False) for k, v in value.items()}
    return value


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


# --- Running ------------------------------------------------------------------


def call(func, args):
    """Returns (result, error, milliseconds, printed output)."""
    printed = io.StringIO()
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(printed):
            result = func(*args)
        error = None
    except (Exception, SystemExit) as exc:
        result, error = None, exc
    return result, error, (time.perf_counter() - start) * 1000, printed.getvalue()


def run_case(label, raw_args, loaded, job):
    if not isinstance(raw_args, list):
        emit(kind="case", case=label, status="bad_test", error="a test case must be a list of arguments")
        return
    arg_kinds = job.get("arg_kinds") or []
    order = job.get("order") or "exact"
    in_place = bool(job.get("in_place"))
    shown_input = "(generated large input)" if label == "large" else shorten(raw_args)

    ref_namespace, ref_func = loaded["reference"]
    ref_args = build_args(raw_args, arg_kinds, ref_namespace)
    emit(kind="progress", case=label, stage="reference")
    ref_result, ref_error, _, _ = call(ref_func, ref_args)
    if ref_error:
        emit(kind="case", case=label, status="reference_error", input=shown_input,
             error=describe_error(ref_error, "<reference>"))
        return
    expected = canonical(to_plain(ref_args[0] if in_place and ref_args else ref_result), order)

    sol_namespace, sol_func = loaded["solution"]
    sol_args = build_args(raw_args, arg_kinds, sol_namespace)
    emit(kind="progress", case=label, stage="solution")
    result, error, ms, printed = call(sol_func, sol_args)

    event = {"kind": "case", "case": label, "input": shown_input, "ms": round(ms, 1)}
    if printed:
        event["printed"] = shorten(printed, is_text=True)
    if error:
        emit(**event, status="error", error=describe_error(error, "<solution>"))
        return
    got = canonical(to_plain(sol_args[0] if in_place and sol_args else result), order)
    if same(expected, got):
        emit(**event, status="pass")
    else:
        emit(**event, status="fail", expected=shorten(expected), got=shorten(got))


def run(job):
    loaded = {}
    for who in ("solution", "reference"):
        try:
            loaded[who] = load(job[who], f"<{who}>", job["function_name"])
        except (Exception, SystemExit) as exc:
            emit(kind="load_error", who=who, error=describe_error(exc, f"<{who}>"))
            return

    cases = [(str(i), args) for i, args in enumerate(job.get("cases") or [], start=1)]
    if job.get("large_input"):
        random.seed(0)
        helpers = {"random": random, "math": math, "string": string, "itertools": itertools}
        try:
            cases.append(("large", list(eval(job["large_input"], helpers))))
        except Exception as exc:
            emit(kind="case", case="large", status="bad_test",
                 error=f"couldn't build the large input: {type(exc).__name__}: {exc}")

    for label, raw_args in cases:
        run_case(label, raw_args, loaded, job)
    emit(kind="done")


def main():
    run(json.loads(sys.stdin.read()))


if __name__ == "__main__":
    # Deep recursion (DFS on a long linked list or skewed tree) needs more room than the defaults.
    sys.setrecursionlimit(100_000)
    threading.stack_size(256 * 1024 * 1024)
    worker = threading.Thread(target=main)
    worker.start()
    worker.join()
