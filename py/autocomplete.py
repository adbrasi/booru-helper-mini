from aiohttp import web
from server import PromptServer
import asyncio
import os
import urllib.request
from urllib.error import URLError, HTTPError

USER_DIR = os.path.abspath(os.path.join(__file__, "../../user"))
os.makedirs(USER_DIR, exist_ok=True)
AUTOCOMPLETE_FILE = os.path.join(USER_DIR, "booru_autocomplete.txt")
DEFAULT_AUTOCOMPLETE_URL = "https://raw.githubusercontent.com/adbrasi/somethings/refs/heads/main/boorutaags.txt"
MAX_AUTOCOMPLETE_BYTES = 8 * 1024 * 1024


def ensure_default_autocomplete_file():
    if os.path.isfile(AUTOCOMPLETE_FILE) and os.path.getsize(AUTOCOMPLETE_FILE) > 0:
        return True

    req = urllib.request.Request(
        DEFAULT_AUTOCOMPLETE_URL,
        headers={"User-Agent": "booru-helper-mini"},
    )

    try:
        with urllib.request.urlopen(req, timeout=20) as response:
            raw = response.read(MAX_AUTOCOMPLETE_BYTES + 1)
            if len(raw) > MAX_AUTOCOMPLETE_BYTES:
                raw = raw[:MAX_AUTOCOMPLETE_BYTES]
        text = raw.decode("utf-8", errors="ignore")
    except (URLError, HTTPError, TimeoutError, OSError) as exc:
        print(f"(booru-helper-mini:autocomplete) bootstrap failed: {exc}")
        return False

    if not text.strip():
        return False

    with open(AUTOCOMPLETE_FILE, "w", encoding="utf-8") as f:
        f.write(text)
    return True


@PromptServer.instance.routes.get("/booruhelper/autocomplete")
async def get_autocomplete(_request):
    if not os.path.isfile(AUTOCOMPLETE_FILE) or os.path.getsize(AUTOCOMPLETE_FILE) == 0:
        await asyncio.to_thread(ensure_default_autocomplete_file)
    if os.path.isfile(AUTOCOMPLETE_FILE):
        return web.FileResponse(AUTOCOMPLETE_FILE)
    return web.Response(status=404)


@PromptServer.instance.routes.post("/booruhelper/autocomplete")
async def update_autocomplete(request):
    with open(AUTOCOMPLETE_FILE, "w", encoding="utf-8") as f:
        f.write(await request.text())
    return web.Response(status=200)
