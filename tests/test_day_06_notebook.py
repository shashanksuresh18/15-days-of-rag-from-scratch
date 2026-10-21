"""Day 06 smoke tests.

1. Execute the notebook top to bottom in a fresh kernel, from two different working
   directories. The notebook's section 9 holds the behavioural assertions, so any failing
   check makes the cell (and this test) fail. The optional live model is switched off for the
   tests (DAY06_LIVE_MODEL is removed from the kernel's environment), so nothing is ever called
   or billed, even if your .env configures a model.
2. Check that the committed copy has no error outputs, no machine-specific paths and no secrets.
3. Run the notebook's code once more and test the parts the notebook can only show once: the
   HTTP client's error handling (against small fake servers on 127.0.0.1, no network), the .env
   reader, and the citation checker on unusual answer formats.
"""

import gc
import json
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "day_06_grounded_rag.ipynb"
PASSED = "All Day 06 sanity checks passed."


def _stream_text(nb) -> str:
    return "".join(
        output.get("text", "")
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
    )


@pytest.mark.parametrize("working_dir", [NOTEBOOK.parent, REPO_ROOT], ids=["from-notebooks", "from-repo-root"])
def test_notebook_runs_top_to_bottom(working_dir: Path, monkeypatch) -> None:
    monkeypatch.delenv("DAY06_LIVE_MODEL", raising=False)     # the kernel inherits this environment
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(
        nb,
        timeout=900,   # per cell; only a first-run model download takes long
        kernel_name="python3",
        resources={"metadata": {"path": str(working_dir)}},
    )
    client.execute()  # raises CellExecutionError at the first failing cell
    text = _stream_text(nb)
    assert PASSED in text and "Live model: off." in text


def test_committed_outputs_are_clean() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    errors = [
        output
        for cell in nb.cells
        if cell.cell_type == "code"
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors, "Committed notebook contains error outputs; re-run it before committing."

    text = _stream_text(nb)
    assert PASSED in text, "Commit the notebook with its outputs."
    assert "Live model: off." in text, "Commit the notebook with the live model switched off."
    for marker in ("/home/", "/Users/", ":\\Users\\", "/root/", "/tmp/", "AppData", "sk-proj", "sk-live"):
        assert marker not in text, f"Machine-specific path or secret in saved outputs: {marker!r}"
    source = "".join("".join(c.source) for c in nb.cells)
    assert "LIVE_MODEL = os.environ.get(\"DAY06_LIVE_MODEL\") == \"1\"" in source, "keep the live model off by default"


# ----------------------------------------------------------------------------- the notebook's code, tested directly
@pytest.fixture(scope="module")
def day06():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "\n\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    module = types.ModuleType("day_06_grounded_rag")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(REPO_ROOT)
        mp.delenv("DAY06_LIVE_MODEL", raising=False)
        mp.setitem(sys.modules, module.__name__, module)   # dataclasses look up their module
        exec(compile(source, NOTEBOOK.name, "exec"), module.__dict__)
    yield module.__dict__
    module.__dict__.clear()     # free the notebook's model and indexes before the next test file starts
    gc.collect()


class FakeServer:
    """Answers every POST with the next (status, body, delay) from a list, and records the requests."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                outer.requests.append({"headers": dict(self.headers),
                                       "body": json.loads(self.rfile.read(int(self.headers["Content-Length"])))})
                status, body, delay = outer.responses.pop(0)
                time.sleep(delay)
                data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:          # the client gave up (timeout test)
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/"   # trailing slash on purpose

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def ok(content, finish_reason="stop", usage=None):
    return {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            **({"usage": usage} if usage else {})}


KEY = "sk-test-0123456789abcdefghijklmnop"


def test_chat_client_handles_failures_without_leaking_the_key(day06: dict) -> None:
    ChatModel, GeneratorError, context = day06["ChatModel"], day06["GeneratorError"], day06["context"]
    responses = [
        (200, ok("Primary caregivers receive 16 weeks [parental-leave-policy#leave-entitlement].",
                 usage={"prompt_tokens": 512, "completion_tokens": 64}), 0),
        (500, {"error": {"message": f"internal error while handling key {KEY}"}}, 0),   # echoes the key
        (429, {"error": {"message": "Rate limit reached"}}, 0),
        (401, {"error": {"message": f"Incorrect API key provided: {KEY[:8]}***{KEY[-4:]}"}}, 0),
        (200, b"<html>proxy error</html>", 0),
        (200, {"choices": []}, 0),
        (200, ok(None, finish_reason="length"), 0),
        (200, ok("too late"), 3),
    ]
    with FakeServer(responses) as fake:
        model = ChatModel("openai", "gpt-6-luna", KEY, fake.url, timeout=1)
        reply = model.generate(context)
        assert reply.text.startswith("Primary caregivers") and reply.usage == {"input_tokens": 512, "output_tokens": 64}
        errors = []
        for _ in responses[1:]:
            with pytest.raises(GeneratorError) as error:
                model.generate(context)
            errors.append(str(error.value))
        request = fake.requests[0]
    assert request["headers"]["Authorization"] == f"Bearer {KEY}"
    assert [m["role"] for m in request["body"]["messages"]] == ["developer", "user"]
    assert request["body"]["max_completion_tokens"] == 1000 and request["body"]["reasoning_effort"] == "low"
    assert "HTTP 500" in errors[0] and "[key]" in errors[0]
    assert "HTTP 429: Rate limit reached" in errors[1]
    assert "HTTP 401, authentication failed" in errors[2]
    assert "was not JSON" in errors[3]
    assert "unexpected response format" in errors[4]
    assert "empty reply (finish_reason: length)" in errors[5]
    assert "could not reach" in errors[6] and "timed out" in errors[6]
    for text in errors + [repr(model), str(model)]:
        assert KEY not in text and KEY[-4:] not in text and KEY[:12] not in text


def test_ollama_preset_sends_no_key_and_its_own_parameter_names(day06: dict) -> None:
    ChatModel, context = day06["ChatModel"], day06["context"]
    with FakeServer([(200, ok("I don't know based on the provided documents."), 0)]) as fake:
        reply = ChatModel("ollama", "some-local-model", base_url=fake.url).generate(context)
        request = fake.requests[0]
    assert "Authorization" not in request["headers"] and reply.usage == {"input_tokens": None, "output_tokens": None}
    body = request["body"]
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["max_tokens"] == 1000 and body["temperature"] == 0 and body["seed"] == 6
    assert "max_completion_tokens" not in body and "reasoning_effort" not in body
    assert ChatModel("ollama", "m").url == "http://localhost:11434/v1/chat/completions"
    with pytest.raises(ValueError, match="needs RAG_CHAT_API_KEY"):
        ChatModel("openai", "gpt-6-luna")
    with pytest.raises(ValueError, match="unknown provider"):
        ChatModel("anthropic", "x", "key")


def test_env_file_reader(day06: dict, tmp_path: Path) -> None:
    read_env_file, chat_model_from_env = day06["read_env_file"], day06["chat_model_from_env"]
    env = tmp_path / ".env"
    env.write_text("\n".join([
        "# a comment", "", "RAG_CHAT_PROVIDER=openai   # inline comment", 'RAG_CHAT_MODEL="gpt-6-luna"',
        f"RAG_CHAT_API_KEY='{KEY}'", "RAG_CHAT_BASE_URL = http://127.0.0.1:9/v1", "NOT A SETTING",
        "WITH_EQUALS=a=b", "HASH_INSIDE=abc#def",
    ]), encoding="utf-8")
    values = read_env_file(env)
    assert values == {"RAG_CHAT_PROVIDER": "openai", "RAG_CHAT_MODEL": "gpt-6-luna", "RAG_CHAT_API_KEY": KEY,
                      "RAG_CHAT_BASE_URL": "http://127.0.0.1:9/v1", "WITH_EQUALS": "a=b", "HASH_INSIDE": "abc#def"}
    model = chat_model_from_env(env)
    assert model.name == "gpt-6-luna via openai" and model.url == "http://127.0.0.1:9/v1/chat/completions"
    assert KEY not in repr(model)
    assert read_env_file(tmp_path / "missing.env") == {} and chat_model_from_env(tmp_path / "missing.env") is None
    # The committed template configures nothing: every RAG_CHAT_ line in it is commented out
    assert chat_model_from_env(REPO_ROOT / ".env.example") is None
    assert not any(k.startswith("RAG_CHAT") for k in read_env_file(REPO_ROOT / ".env.example"))


def test_citation_checker_on_other_answer_formats(day06: dict) -> None:
    check, context = day06["check_citations"], day06["context"]
    entitlement, eligible = "parental-leave-policy#leave-entitlement", "parental-leave-policy#who-is-eligible"
    two_ids = check(f"Leave is 16 weeks after six months of service [{entitlement}, {eligible}].", context)
    assert two_ids.passed and two_ids.claims[0].cited == (entitlement, eligible)
    bullets = check(f"- Primary caregivers: 16 weeks [{entitlement}]\n- Secondary caregivers: 6 weeks [{entitlement}]",
                    context)
    assert bullets.passed and len(bullets.claims) == 2
    numbered = check("Leave is 16 weeks [1].", context)
    assert numbered.problems == ["[1] was not among the sources sent"]
    abstain_plus = check("I don't know based on the provided documents. But maybe 20 weeks.", context)
    assert abstain_plus.abstained and abstain_plus.passed
    assert not check("", context).passed and not check("   ", context).passed
    assert check(f"Primary caregivers get 16 weeks of leave. [{entitlement}]", context).passed
