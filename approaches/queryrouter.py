import re

GRAPHRAG_TRIGGER_PATTERNS = [
    r"\brelated to\b",
    r"\bconnected\b",
    r"\blinked\b",
    r"\bcompare\b",
    r"\ball vendors?\b",
    r"\ball customers?\b",
    r"\bacross\b",
    r"\bshow everything\b",
    r"\ball interactions?\b",
    r"\beverything related\b",
    r"\ball transactions?\b",
    r"\ball invoices?\b",
    r"\ball (purchase )?orders?\b",
    r"\ball grns?\b",
    r"\btotal spend\b",
    r"\bsummary of\b",
    r"\bfind all\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in GRAPHRAG_TRIGGER_PATTERNS]


class QueryRouter:
    def should_use_graphrag(self, query: str) -> bool:
        return any(pattern.search(query) for pattern in _COMPILED)
