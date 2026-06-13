import importlib


def test_noop_import_does_not_raise() -> None:
    mod = importlib.import_module("app.observability")
    assert mod is not None


def test_noop_trace_id_is_empty_string() -> None:
    from app.observability import start_trace

    trace = start_trace("test", input={"transaction_id": "txn_test"})
    assert trace.id == ""


def test_noop_span_has_end_method() -> None:
    from app.observability import start_trace

    trace = start_trace("test", input={})
    span = trace.span(name="tool.check_blacklist", input={"user_id": "u1"})
    assert callable(span.end)
    span.end(output={"match": False})


def test_noop_generation_has_end_method() -> None:
    from app.observability import start_trace

    trace = start_trace("test", input={})
    generation = trace.generation(name="claude.completion", model="claude-sonnet-4-6", input=[])
    assert callable(generation.end)
    generation.end(output="ok", usage={"input": 100, "output": 50})
