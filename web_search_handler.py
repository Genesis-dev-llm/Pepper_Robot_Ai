"""
Web Search Handler — DuckDuckGo (free, no API key required)

Changes from original:
- All searches run inside a ThreadPoolExecutor with a configurable timeout.
  DuckDuckGo calls can silently hang; this ensures the robot is never frozen
  waiting for a network response.
"""

import concurrent.futures
from typing import Dict, List

from duckduckgo_search import DDGS


class WebSearchHandler:
    def __init__(self, max_results: int = 3, timeout: float = 8.0):
        """
        Args:
            max_results: Maximum results per query.
            timeout:     Seconds before a hung search is abandoned.
        """
        self.max_results = max_results
        self.timeout     = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str) -> str:
        """
        Search and return a formatted string suitable for injecting into
        the LLM context.  Never hangs — returns an error string on timeout.
        """
        print(f"🔍 Web search: '{query}'")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._do_search, query)
                results = future.result(timeout=self.timeout)
        except concurrent.futures.TimeoutError:
            msg = f"Search timed out after {self.timeout}s for '{query}'"
            print(f"⏱️ {msg}")
            return msg
        except Exception as e:
            msg = f"Search error: {e}"
            print(f"❌ {msg}")
            return msg

        if not results:
            return f"No results found for '{query}'"

        lines = [f"Web search results for '{query}':\n"]
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body  = r.get("body",  "No description")
            url   = r.get("href",  "")
            lines.append(f"{i}. {title}")
            lines.append(f"   {body}")
            if url:
                lines.append(f"   Source: {url}")
            lines.append("")

        print(f"✅ Got {len(results)} search result(s)")
        return "\n".join(lines).strip()

    def search_structured(self, query: str) -> List[Dict]:
        """Return raw result dicts (for programmatic use)."""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(self._do_search, query)
                return future.result(timeout=self.timeout)
        except Exception as e:
            print(f"❌ Structured search error: {e}")
            return []

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _do_search(self, query: str) -> List[Dict]:
        """Blocking DuckDuckGo call — always run inside a thread with timeout."""
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=self.max_results)
            return list(raw) if raw else []


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    handler = WebSearchHandler(max_results=3, timeout=8.0)

    print("=== Formatted search ===")
    print(handler.search("latest AI news 2026"))

    print("\n=== Structured search ===")
    for r in handler.search_structured("python programming"):
        print(f"  - {r.get('title', 'N/A')}")