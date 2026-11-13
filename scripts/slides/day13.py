"""Build the Day 13 LinkedIn carousel (whiteboard style) as editable SVGs in assets/day-13/.

    python scripts/slides/day13.py                    # writes the SVGs
    python scripts/slides/render.py assets/day-13     # renders PNGs + PDF

Every number and ID on the slides comes from the executed Day 13 notebook or the answer baseline it is compared with.
`check_facts()` confirms each quoted line appears in the notebook's saved outputs before anything is written, and
`from_baseline()` recounts the semantic cache's and the fallback's results from the committed baseline. Timings are
one run on one machine (the notebook prints which); the slides say so.
"""

from __future__ import annotations

import json
from pathlib import Path

from kit import COLORS, INK, M, TEXT, W, Slide, check_contrast

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "assets" / "day-13"
NOTEBOOK = ROOT / "notebooks" / "day_13_rag_operations_tracing_latency_cost.ipynb"
BASELINE = ROOT / "evals" / "baselines" / "day-13-answers-v5.json"
TOTAL = 4
DAY = "13"
PAPER_WHITE = "#FFFFFF"

# The trace drawn on slide 1, span by span: (name, start ms, ms) as the notebook printed them
TRACE = [("embed_query", 0.0, 15.0), ("retrieve.bm25", 15.0, 0.1), ("retrieve.dense", 15.1, 0.2),
         ("retrieve.fuse", 15.3, 0.1), ("gate", 15.4, 0.0), ("context", 15.4, 0.1), ("generate", 15.4, 102.7),
         ("check", 118.2, 0.1)]
FACTS = [
    "236 v4 questions x 2 pipelines compared with evals/baselines/day-12-answers-v4.json: no changes",
    "req-0001  user alice ['all-employees']  question sha256 e5a84cae394b1030 (8 words)",
    *[f"  {name:<20}{start:>5.1f}  {ms:>6.1f} ms" for name, start, ms in TRACE],
    "gate                 15.4     0.0 ms #                        threshold 0.62; dense_top 0.6758; decision pass",
    "sent travel-policy#safety-while-travelling, security-incident-reporting#what-not-to-do, "
    "security-incident-reporting#how-to-report; dropped -; words 113; prompt_tokens_estimate 497",
    "  total 118.3 ms -> answered, served by extractive reader (API-free), cites travel-policy#safety-while-travelling",
    "labelled answer chunk: security-incident-reporting#how-to-report | outcome: wrong",
    "287 questions traced: every answer identical to the untraced pipeline's | spans per trace: 5 to 8 | document text, "
    "question text or keys found: 0",
    "answerable questions whose trace shows the labelled section sent and another one cited: 18",
    "59 requests | whole request p50 119.56 ms, p95 167.05 ms",
    "  generate          50 spans   p50  107.953 ms   p95  151.995 ms    86% of all the time",
    "precomputed at index time: 102 sentence and table-row vectors in 1.57 s (once)",
    "236 requests | whole request p50 12.10 ms, p95 16.77 ms",
    "  embed_query      236 spans   p50   11.597 ms   p95   15.992 ms    96% of all the time",
    "Repeated (the same user asks the same question again: the answer cache): 236 requests | whole request p50 0.01 ms",
    "p50 1.10 s, p95 1.64 s, slowest 2.63 s: about 91 times the warm API-free request's p50",
    "estimate (characters / 4) against the provider's count of input tokens: actual / estimate 0.91 on average",
    "where the input tokens go (estimated): instructions 38%, sources 58%, question 4% | the longest prompt: about 543 "
    "tokens, under the 1024 the provider's prompt cache needs",
    "gpt-6-luna   USD 0.10 / 0.50 per million: USD 0.000059 per request (input 76%) = USD 0.059 per 1,000 requests that "
    "reach the model",
    "gpt-6.1-sol  USD 2.00 / 10.00 per million: USD 0.001187 per request (input 76%) = USD 1.187 per 1,000 requests",
    "of the 236 dev questions, the gate refuses 29 without calling a generator",
    "the API-free path instead: about 12 ms of this machine's CPU per request (warm)",
    "QuestionOnlyCache: maria asks, then alice asks the same question -> alice gets [answered]",
    "  LEAK: it cites manager-pay-review-guide#overview, which only managers may read",
    "the password policy changes from 14 to 16 characters",
    "  after the update: Passwords must be at least 14 characters long [password-and-mfa-standard#passwords]. "
    "(served by answer cache)",
    "theta = 0.89 (the rule's value)",
    "d13-u13: the pipeline's own answer is refused",
    "  HIT  d13-u13 paraphrase     0.9436 nearest d10-u05 own refused  served answered If I give money to a charity, "
    "will Acme match it?",
    "better 0, worse 1 (d13-u13), near ties set apart 0, sign test p = 1 | value change at cost 1 / 2 / 3: -1 / -2 / -3",
    "D1, the semantic cache at theta 0.89: not adopted",
    "one 503, then fine                    2",
    "429 with Retry-After 0.2 s            2    0.20s",
    "429 with Retry-After 5 s              1    0.00s   0.02s   fallback reader  HTTP 429",
    "401, wrong key                        1    0.00s   0.02s   fallback reader  HTTP 401",
    "slow (2 s per reply)                  3    0.04s   1.52s   fallback reader  timeout, timeout, timeout",
    "trickle (2.5 s per reply)             1    0.00s   1.55s   fallback reader  deadline",
    "not JSON (a proxy page)               1    0.00s   0.02s   fallback reader  not JSON",
    "Day 06's ChatModel with a 0.5 s timeout, same trickling server: waited 2.44 s",
    "an outage, 8 requests, retries only          : 24 calls to the failing server",
    "an outage, 8 requests, retries and a breaker :  9 calls to the failing server",
    "breaker during the outage: open; 1.5 s later one trial request (1 call) is answered by gpt-6-luna via openai -> "
    "breaker closed",
    "  correct                   20 of 33",
    "  unanswerable answered     12 of 18",
    "value of answering with it instead of 'unavailable' (worth 0), at cost 1 / 2 / 3: +1 / -18 / -37",
    "even if every one of them went the best way, the value would be -9 at cost 2",
    "D2, the API-free reader as the fallback when the model fails: not adopted (value -18 at cost 2)",
    "compared with evals/baselines/day-13-answers-v5.json: no changes",
    "All Day 13 sanity checks passed.",
]
SETTINGS = "live pass: 20 questions"     # the client settings are printed on this line
CLIENT_LINE = "client: attempts 3, timeout 10, deadline 30, backoff_base 0.5, backoff_cap 8, breaker after 3 failed " \
              "requests for 30 s, fallback: the API-free reader"


def notebook_outputs() -> str:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join("".join(o.get("text", "")) for c in nb["cells"] if c["cell_type"] == "code"
                   for o in c.get("outputs", []))


def check_facts() -> None:
    outputs = notebook_outputs()
    missing = [f for f in FACTS + [SETTINGS, CLIENT_LINE] if f not in outputs]
    if missing:
        raise SystemExit(f"Slide facts not found (re-run the notebook?): {missing}")


def from_baseline() -> dict:
    # The semantic cache's and the fallback's results, recounted from the committed baseline
    b = json.loads(BASELINE.read_text(encoding="utf-8"))
    sc = b["semantic_cache"]["v5 test"]
    hits = sc["hits"]
    same = [qid for qid in hits if b["questions"][qid]["semantic_cache"]["served_outcome"] ==
            b["questions"][qid]["current"]["outcome"]]
    m = b["metrics"]["v5 test"]["current"]
    facts = {"theta": b["semantic_cache"]["theta"], "hits": len(hits), "same": len(same), "better": len(sc["better"]),
             "worse": sc["worse"], "delta": sc["delta"], "fallback": b["fallback"]["v5 test"]["value"],
             "correct": m["correct"], "wrong": m["wrong"], "u_answered": m["unanswerable answered"],
             "decisions": sorted(b["decisions"][k] for k in b["decisions"] if k.startswith("D"))}
    expected = {"theta": 0.89, "hits": 10, "same": 9, "better": 0, "worse": ["d13-u13"],
                "delta": {"1": -1, "2": -2, "3": -3}, "fallback": {"1": 1, "2": -18, "3": -37}, "correct": 20,
                "wrong": 7, "u_answered": 12, "decisions": ["not adopted", "not adopted"]}
    if facts != expected:
        raise SystemExit(f"the slides' numbers don't match the baseline: {facts}")
    return facts


def heading(s: Slide, title: str, subtitle: str, size: int = 54) -> None:
    s.text(M, 190, title, size, 800, INK)
    s.text(M, 238, subtitle, 25, 500, TEXT["muted"])


def tcolor(color: str) -> str:
    return TEXT.get(color, INK) if color != "yellow" else INK


def zone_title(s: Slide, y: float, icon: str, color: str, title: str, right: str | None = None,
               right_width: float = 300) -> None:
    s.icon(icon, M + 38, y + 40, 28, COLORS[color][0] if color != "gray" else TEXT["gray"], sw=2.4)
    s.text(M + 64, y + 49, title, 22, 800, tcolor(color), max_width=W - 2 * M - 90 - (right_width if right else 0))
    if right:
        s.text(W - M - 28, y + 49, right, 19, 700, TEXT["muted"], anchor="end", max_width=right_width)


def signed(n: int) -> str:
    return f"{n:+d}".replace("-", "\u2212")


def mark(s: Slide, x: float, y: float, good: bool, size: int = 20) -> None:
    s.icon("check" if good else "x", x, y, size, COLORS["green" if good else "red"][0], sw=3.2)


# ----------------------------------------------------------------------------- slide 1
def slide_cover() -> Slide:
    s = Slide(1, TOTAL, DAY, "Trace every request", seed=131)
    heading(s, "Trace every request", "Day 13: where a request spends its time, and where it loses the answer.")

    zy = 274
    s.box(M, zy, W - 2 * M, 470, "blue", "wash", r=22, dashed=True, sw=3)
    zone_title(s, zy, "chart-gantt", "blue", "ONE REQUEST, ONE TRACE", right="cold, one machine", right_width=230)
    s.text(M + 30, zy + 92, "“How soon must I report a lost laptop?”", 19, 700, TEXT["muted"], max_width=W - 2 * M - 60)
    x0, full, total_ms = M + 230, 470, 120.0
    rows = [("embed_query", 0.0, 15.0, "blue", "15.0 ms", "the question → a vector"),
            ("retrieve ×3", 15.0, 0.4, "blue", "0.4 ms", "BM25, dense, fusion"),
            ("gate", 15.4, 0.0, "green", "0.0 ms", "0.6758 ≥ 0.62: pass"),
            ("context", 15.4, 0.1, "green", "0.1 ms", "3 sections, ≈ 497 tokens"),
            ("generate", 15.4, 102.7, "violet", "102.7 ms", "the reader"),
            ("check", 118.2, 0.1, "gray", "0.1 ms", "citations: ok")]
    for i, (name, start, ms, color, label, note) in enumerate(rows):
        ry = zy + 124 + i * 46
        s.text(M + 30, ry + 24, name, 19, 500, INK, family="mono", max_width=190)
        s.line(x0, ry + 17, x0 + full, ry + 17, "#DEE2E6", sw=1.6, bow=0.2)
        bx = x0 + full * start / total_ms
        bw = max(full * ms / total_ms, 8)
        s.box(bx, ry + 4, bw, 26, color, "fill", r=6, sw=2.0, jitter=0.4)
        if bx + bw + 150 < W - M - 20:
            s.text(bx + bw + 12, ry + 24, label, 18, 800, INK, max_width=120)
        else:
            s.text(bx - 12, ry + 24, label, 18, 800, INK, anchor="end", max_width=120)
    s.text(W - M - 30, zy + 124 + 6 * 46 + 18, "total 118.3 ms", 21, 800, TEXT["blue"], anchor="end", max_width=260)
    s.text(M + 30, zy + 124 + 6 * 46 + 18, "ids, scores, decisions, times: no text", 19, 700, TEXT["muted"],
           max_width=W - 2 * M - 330)
    s.text(M + 30, zy + 450, "the reader is 86% of a cold request; with vectors saved at index time, 1%", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    zy = 766
    s.box(M, zy, W - 2 * M, 260, "orange", "wash", r=22, sw=2.8)
    zone_title(s, zy, "search", "orange", "THE TRACE SAYS WHICH STEP LOST IT")
    s.text(M + 30, zy + 100, "sent", 20, 800, INK, max_width=90)
    s.chip(M + 130, zy + 74, "security-incident-reporting#how-to-report", "green", size=18)
    mark(s, W - M - 40, zy + 94, True, 22)
    s.text(M + 30, zy + 160, "cited", 20, 800, INK, max_width=90)
    s.chip(M + 130, zy + 134, "travel-policy#safety-while-travelling", "red", size=18)
    mark(s, W - M - 40, zy + 154, False, 22)
    s.text(M + 30, zy + 208, "retrieval found the answer; the reader picked another sentence", 19, 800, TEXT["orange"],
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 240, "18 answerable questions show this pattern: sent, not cited", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)

    zy = 1048
    s.box(M, zy, W - 2 * M, 190, "yellow", "fill", r=22, sw=3)
    s.icon("eye-off", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "287 TRACES, NO TEXT IN ANY OF THEM", 22, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 102, "the question is a hash; documents, names and keys stay out", 20, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 146, "checked: no 6 words of a document, no 4 of the question", 20, 800, TEXT["teal"],
           max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 2
def slide_latency_cost() -> Slide:
    s = Slide(2, TOTAL, DAY, "Time and money, per step", seed=132)
    heading(s, "Time and money, per step", "p50 and p95 per step; the cost of the requests that reach a model.")

    zy = 274
    s.box(M, zy, W - 2 * M, 470, "violet", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "timer", "violet", "LATENCY: WHICH REQUEST?", right="one run, one machine", right_width=270)
    s.text(M + 610, zy + 96, "p50", 17, 800, TEXT["muted"], anchor="end", max_width=100)
    s.text(W - M - 46, zy + 96, "the biggest step", 17, 800, TEXT["muted"], anchor="end", max_width=230)
    rows = [("cold", "nothing cached", "119.56 ms", "generate 86%", "violet"),
            ("warm", "sentence vectors saved once", "12.10 ms", "embed_query 96%", "blue"),
            ("repeat", "the answer cache", "0.01 ms", "the lookup", "green"),
            ("a model", "Day 12's calls: ≈ 91 × warm", "1.10 s", "the model call", "orange")]
    for i, (name, note, p50, biggest, color) in enumerate(rows):
        ry = zy + 142 + i * 66
        s.box(M + 24, ry - 28, W - 2 * M - 48, 54, color, PAPER_WHITE, r=14, sw=2.0, jitter=0.6)
        s.text(M + 46, ry + 8, name, 21, 800, tcolor(color), max_width=110)
        s.text(M + 160, ry + 8, note, 18, 700, TEXT["muted"], max_width=310)
        s.text(M + 610, ry + 9, p50, 23, 800, INK, anchor="end", max_width=140)
        s.text(W - M - 46, ry + 8, biggest, 18, 800, INK, anchor="end", max_width=230)
    s.text(M + 30, zy + 412, "the biggest step changes with the request: measure each kind", 19, 800, TEXT["violet"],
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 448, "p95s don't add up: the steps' 171.33 ms vs the requests' 167.05 ms (cold)", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    zy = 766
    s.box(M, zy, W - 2 * M, 472, "green", "wash", r=22, sw=2.8)
    zone_title(s, zy, "coins", "green", "COST PER 1,000 REQUESTS THAT REACH A MODEL", right_width=0)
    for i, (model, price, cost) in enumerate([("gpt-6-luna", "USD 0.10 / 0.50 per M tokens", "USD 0.059"),
                                              ("gpt-6.1-sol", "USD 2 / 10 per M tokens", "USD 1.187")]):
        ry = zy + 104 + i * 54
        s.text(M + 30, ry, model, 21, 800, INK, family="mono", max_width=220)
        s.text(M + 260, ry, price, 18, 700, TEXT["muted"], max_width=380)
        s.text(W - M - 30, ry, cost, 24, 800, TEXT["green"], anchor="end", max_width=180)
    s.text(M + 30, zy + 204, "the API-free reader: no token price, ≈ 12 ms of CPU per warm request", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)
    facts = [("ban", "the gate refuses 29 of 236 before any model call"),
             ("file-text", "input is 76% of the cost; the sources are 58% of the input"),
             ("ruler", "the provider's count ≈ 0.91 × our estimate (characters / 4)"),
             ("layers", "longest prompt ≈ 543 tokens: prompt caching starts at 1,024")]
    for i, (icon, text) in enumerate(facts):
        ry = zy + 262 + i * 46
        s.icon(icon, M + 46, ry - 7, 24, COLORS["green"][0], sw=2.3)
        s.text(M + 76, ry, text, 19, 700, INK, max_width=W - 2 * M - 106)
    s.text(M + 30, zy + 444, "prices checked 2026-11-12; tokens counted by the provider on 40 requests", 16, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 3
def slide_caches(f: dict) -> Slide:
    s = Slide(3, TOTAL, DAY, "Get the cache key right", seed=133)
    heading(s, "Get the cache key right", "Exact caches changed no answer; a missing part of the key did.")

    zy = 274
    s.box(M, zy, W - 2 * M, 200, "blue", "wash", r=22, sw=2.8)
    zone_title(s, zy, "key-round", "blue", "THE ANSWER CACHE'S KEY")
    x = M + 30
    for label in ("groups", "question", "index fingerprint", "pipeline version"):
        x += s.tag(x, zy + 118, label, "blue", size=19) + 12
    s.text(M + 30, zy + 176, "+ re-check the user may still read every cited source", 19, 800, TEXT["blue"],
           max_width=W - 2 * M - 60)

    zy = 496
    s.box(M, zy, W - 2 * M, 300, "red", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "triangle-alert", "red", "LEAVE ONE PART OUT")
    cards = [("users", "no groups", "maria (a manager) asks; alice asks it too",
              "alice gets a managers-only answer", "LEAK"),
             ("fingerprint-pattern", "no fingerprint", "the password policy: 14 → 16 characters",
              "the cache still says 14", "STALE")]
    width = (W - 2 * M - 60 - 20) / 2
    for i, (icon, title, what, result, word) in enumerate(cards):
        cx = M + 30 + i * (width + 20)
        s.box(cx, zy + 80, width, 200, "red", PAPER_WHITE, r=16, sw=2.0, jitter=0.6)
        s.icon(icon, cx + 30, zy + 112, 24, COLORS["red"][0], sw=2.3)
        s.text(cx + 54, zy + 120, title, 20, 800, INK, max_width=width - 160)
        s.tag(cx + width - 16, zy + 122, word, "red", size=17, anchor="end")
        s.paragraph(cx + 20, zy + 166, what, 17, 700, width - 40, 24, TEXT["muted"], max_lines=2)
        s.paragraph(cx + 20, zy + 234, result, 18, 800, width - 40, 24, TEXT["red"], max_lines=2)

    zy = 818
    s.box(M, zy, W - 2 * M, 420, "violet", "wash", r=22, sw=2.8)
    zone_title(s, zy, "scale", "violet", "A SEMANTIC CACHE, DECIDED IN ADVANCE")
    s.text(M + 30, zy + 98, f"serve a past answer if cosine ≥ {f['theta']}: the lowest with no dev answer worse", 19, 700,
           INK, max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 134, "51 new questions:", 19, 800, INK, max_width=200)
    groups = [(f"{f['same']} same outcome", f["same"], "gray"), (f"{f['better']} better", f["better"], "green"),
              (f"{len(f['worse'])} worse", len(f["worse"]), "red")]
    for i, (label, n, color) in enumerate(groups):
        ry = zy + 180 + i * 44
        for j in range(n):
            s.ellipse(M + 46 + j * 32, ry - 7, 12, 12, color, "fill", sw=2.2, jitter=0.6)
        s.text(M + 380, ry, label, 20, 800, tcolor(color) if color != "gray" else TEXT["muted"], max_width=300)
    s.text(M + 30, zy + 330, "“If I give money to a charity, will Acme match it?”", 18, 700, TEXT["muted"],
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 358, "nothing answers it; the pipeline refuses, the cache served an answer", 18, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 398, f"value {signed(f['delta']['2'])} at cost 2: not adopted; similar isn't the same", 20, 800,
           TEXT["violet"], max_width=W - 2 * M - 60)
    return s


# ----------------------------------------------------------------------------- slide 4
def slide_failures(f: dict) -> Slide:
    s = Slide(4, TOTAL, DAY, "Plan for the model to fail", seed=134)
    heading(s, "Plan for the model to fail", "Timeouts, retries, a deadline and a breaker, tested on a fake server.")

    zy = 274
    s.box(M, zy, W - 2 * M, 444, "orange", "wash", r=22, dashed=True, sw=2.8)
    zone_title(s, zy, "server-crash", "orange", "WHAT THE CLIENT DOES", right="real settings; the demo runs 20× faster", right_width=420)
    rows = [("503, 500, 429", "retry after a random wait (full jitter)", "refresh-cw", "blue"),
            ("429 + Retry-After", "wait as told, if it fits the deadline", "timer", "blue"),
            ("401, no credit, not JSON", "no retry: it won't pass", "ban", "red"),
            ("slow server", "10 s per attempt, 3 attempts", "hourglass", "violet"),
            ("trickling server", "a 30 s deadline for the whole call", "hourglass", "violet"),
            ("still failing", "the fallback; the trace says why", "life-buoy", "green")]
    for i, (what, does, icon, color) in enumerate(rows):
        ry = zy + 104 + i * 52
        s.text(M + 30, ry, what, 19, 800, INK, max_width=300)
        s.icon(icon, M + 356, ry - 7, 22, COLORS[color][0], sw=2.3)
        s.text(M + 382, ry, does, 18, 700, INK, max_width=W - 2 * M - 412)
    s.text(M + 30, zy + 420, "a timeout is per read: Day 06's client waited 2.44 s with a 0.5 s timeout", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    zy = 740
    s.box(M, zy, W - 2 * M, 222, "red", "wash", r=22, sw=2.8)
    zone_title(s, zy, "unplug", "red", "A CIRCUIT BREAKER IN AN OUTAGE", right="8 requests", right_width=160)
    full = W - 2 * M - 420
    for i, (label, n) in enumerate([("retries only", 24), ("with a breaker", 9)]):
        ry = zy + 84 + i * 50
        s.text(M + 30, ry + 22, label, 19, 800, INK, max_width=220)
        s.box(M + 260, ry, full * n / 24, 32, "red" if i == 0 else "green", "fill", r=10, sw=2.2, jitter=0.6)
        s.text(M + 260 + full * n / 24 + 12, ry + 24, f"{n} calls", 19, 800, INK, max_width=120)
    s.text(M + 30, zy + 200, "3 failed requests open it; after a cool-down, one good trial closes it", 17, 700,
           TEXT["muted"], max_width=W - 2 * M - 60)

    zy = 984
    s.box(M, zy, W - 2 * M, 254, "yellow", "fill", r=22, sw=3)
    s.icon("life-buoy", M + 40, zy + 44, 30, TEXT["orange"], sw=2.4)
    s.text(M + 70, zy + 54, "WHICH FALLBACK? DECIDED BY VALUE", 22, 800, INK, max_width=W - 2 * M - 100)
    s.text(M + 30, zy + 96, "the API-free reader instead of “unavailable”, on 51 new questions:", 19, 700, INK,
           max_width=W - 2 * M - 60)
    s.text(M + 30, zy + 134, f"{f['correct']} correct · {f['wrong']} wrong · {f['u_answered']} unanswerable ones "
           "answered", 20, 800, INK, max_width=W - 2 * M - 60)
    s.chip(M + 30, zy + 152, "correct +1 · refused 0 · wrong −2", "orange", size=18)
    s.text(M + 30, zy + 228, f"value {signed(f['fallback']['2'])}: not adopted; say “unavailable” instead", 21, 800,
           TEXT["teal"], max_width=W - 2 * M - 60)
    return s


def main() -> None:
    check_contrast()
    check_facts()
    f = from_baseline()
    OUT.mkdir(parents=True, exist_ok=True)
    slides = [("slide-01-cover", slide_cover()), ("slide-02-latency-and-cost", slide_latency_cost()),
              ("slide-03-cache-keys", slide_caches(f)), ("slide-04-failures-and-fallback", slide_failures(f))]
    for name, slide in slides:
        path = OUT / f"{name}.svg"
        path.write_text(slide.svg(), encoding="utf-8")
        print("wrote", path.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
