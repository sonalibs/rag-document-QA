import time
from pathlib import Path
import wikipedia

wikipedia.wikipedia.API_URL = "https://en.wikipedia.org/w/api.php"
wikipedia.wikipedia.USER_AGENT = "SonaliAIProjects-RAG/1.0"

TITLES = [
    "Vector database",
    "Word embedding",
    "Prompt engineering",
]

OUT_DIR = Path("data/ai_docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_CHARS = 20000

for title in TITLES:
    for attempt in range(3):
        try:
            page = wikipedia.page(title, auto_suggest=False)
            filename = page.title.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") + ".txt"
            (OUT_DIR / filename).write_text(page.content[:MAX_CHARS], encoding="utf-8")
            print(f"[OK] {title} -> {filename} ({min(len(page.content), MAX_CHARS)} chars)")
            break
        except Exception as e:
            if attempt == 2:
                print(f"[FAILED] {title}: {e}")
            else:
                time.sleep(10 * (attempt + 1))
    time.sleep(8)
