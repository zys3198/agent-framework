import json

from trace.logger import TraceLogger


def test_log_appends_jsonl(tmp_path):
    t = TraceLogger(tmp_path / "sid-1.jsonl")
    t.log_step(0)
    t.log_route("plan_required")
    t.log_tool_call(0, "todo.create", {"title": "A"})
    t.log_tool_result(0, "created #1")
    t.log_truncated()
    t.close()

    lines = (tmp_path / "sid-1.jsonl").read_text(encoding="utf-8").strip().split("\n")
    recs = [json.loads(line) for line in lines]
    assert recs[0]["type"] == "step" and recs[0]["step"] == 0
    assert recs[1]["type"] == "route" and recs[1]["value"] == "plan_required"
    assert recs[2]["name"] == "todo.create"
    assert recs[3]["result"] == "created #1"
    assert recs[4]["type"] == "truncated"
    assert all("ts" in r for r in recs)
