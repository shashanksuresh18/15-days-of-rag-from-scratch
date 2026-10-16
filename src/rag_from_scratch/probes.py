"""Day 03's 13 diagnostic questions, packaged for reuse by later days.

Each Probe pairs a question with its evidence: the exact strings that must ALL appear in one
retrieved chunk for that chunk to answer the question on its own. Each question probes one way
retrieval can fail (a table row without its header, words only in a heading, evidence spread over
two sections, ...). They are a diagnostic, not a benchmark; Day 07 builds the labelled set.

Copied from notebooks/day_03_chunking_and_structure.ipynb (section 6). tests/test_shared_code.py
checks that this list is identical to the notebook's QUESTIONS.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Probe:
    question: str
    evidence: tuple[str, ...]    # strings that must ALL appear in one chunk for it to answer alone
    probes: str                  # what this question tests


DAY03_PROBES = (
    Probe("What is the hotel limit per night in Singapore?", ("Singapore | USD 220", "Hotel per night"),
          "PDF table: row + column header"),
    Probe("What is the daily meal limit in New York?", ("New York | USD 300 | USD 75", "Meals per day"),
          "PDF table: row + column header"),
    Probe("Who approves admin rights?", ("Admin rights | IT Security and your manager", "Who approves"),
          "HTML table: row + column header"),
    Probe("How long is parental leave?", ("16 weeks",), "Day 01: answer was at rank 3"),
    Probe("How many failed sign-in attempts lock my account?", ("five failed sign-in attempts",), "one sentence"),
    Probe("How long are application logs kept?", ("Application logs are kept for 13 months",), "one sentence"),
    Probe("How much is the home office allowance?", ("home office allowance of USD 500",), "one sentence"),
    Probe("How long do I have to claim the home office allowance?", ("within 60 days of approval",),
          "second sentence of a section"),
    Probe("Can I use my personal computer for work?", ("personal computers must not be used for company work",),
          "last sentence of a section"),
    Probe("Can I install software from the internet myself?", ("Do not install software from the internet yourself",),
          "text before the first heading"),
    Probe("What training must new starters complete?", ("security awareness and data protection courses",),
          "topic word only in the heading"),
    Probe("What happens after I report a security incident?", ("confirm receipt",),
          "topic words only in the heading"),
    Probe("How soon must I report a lost laptop?", ("within one hour",),
          "evidence spread over two sections"),
)
