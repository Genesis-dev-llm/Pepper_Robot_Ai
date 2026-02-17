"""
Web Search Handler using DuckDuckGo
Free, unlimited searches with no API key required
"""

from duckduckgo_search import DDGS
from typing import List, Dict


class WebSearchHandler:
    def __init__(self, max_results: int = 3):
        """
        Initialize web search handler

        Args:
            max_results: Maximum number of search results to return (default: 3)
        """
        self.max_results = max_results
        # NOTE: DDGS is created fresh per search call (not stored as instance var)
        # This avoids connection-timeout issues on long-running sessions.

    def search(self, query: str) -> str:
        """
        Search the web and return formatted results.

        Args:
            query: Search query string

        Returns:
            Formatted string with search results
        """
        try:
            print(f"🔍 Searching web for: '{query}'")

            # Fresh DDGS instance per call — avoids stale connection issues
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=self.max_results)

            # Ensure we have a plain list (DDGS may return a generator)
            results: List[Dict] = list(raw) if raw else []

            if not results:
                return f"No search results found for '{query}'"

            # Format results
            formatted = f"Web search results for '{query}':\n\n"
            for i, result in enumerate(results, 1):
                title = result.get('title', 'No title')
                body  = result.get('body',  'No description')
                url   = result.get('href',  '')

                formatted += f"{i}. {title}\n"
                formatted += f"   {body}\n"
                if url:
                    formatted += f"   Source: {url}\n"
                formatted += "\n"

            print(f"✅ Found {len(results)} results")
            return formatted.strip()

        except Exception as e:
            error_msg = f"Search error: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg

    def search_structured(self, query: str) -> List[Dict]:
        """
        Search and return structured results (for programmatic use).

        Args:
            query: Search query string

        Returns:
            List of result dictionaries
        """
        try:
            with DDGS() as ddgs:
                raw = ddgs.text(query, max_results=self.max_results)
            return list(raw) if raw else []
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing DuckDuckGo search...")

    search = WebSearchHandler(max_results=3)

    results = search.search("latest AI news 2026")
    print(results)

    print("\n" + "=" * 60 + "\n")

    structured = search.search_structured("python programming")
    print(f"Found {len(structured)} structured results:")
    for r in structured:
        print(f"- {r.get('title', 'No title')}")