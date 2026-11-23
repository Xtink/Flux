import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from groq import Groq

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

app = FastAPI(title="FLUX + Xbot API", version="1.0.0")

# The frontend and API are served from the same Render service.
# CORS is left open for preview/testing deployments; the Groq key stays server-side.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_SOURCE_CHARS = 60_000
MAX_ATTACHMENT_CHARS = 20_000
MAX_HISTORY_ITEMS = 12
MAX_MESSAGE_CHARS = 8_000
MAX_FILES = 40
MAX_FILE_CHARS = 300_000

MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class ChatRequest(BaseModel):
    action: str = "chat"
    message: str = ""
    flux: str = ""
    active_file: str = "main.flux"
    history: list[dict[str, Any]] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    attachment: dict[str, str] | None = None


def groq_client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is missing. Add it in Render → Environment.",
        )
    return Groq(api_key=key)


def clean_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for item in history[-MAX_HISTORY_ITEMS:]:
        role = item.get("role")
        # The browser stores messages as {from: you|bot, text: ...}.
        if not role:
            role = "user" if item.get("from") == "you" else "assistant"
        if role not in ("user", "assistant"):
            continue
        content = str(item.get("content", item.get("text", "")))[:MAX_MESSAGE_CHARS]
        if content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned


def safe_project_path(raw_path: Any) -> str:
    path = str(raw_path or "").replace("\\", "/").strip()
    candidate = PurePosixPath(path)
    if (
        not path
        or candidate.is_absolute()
        or any(part in ("", ".", "..") for part in candidate.parts)
        or re.match(r"^[A-Za-z]:", path)
    ):
        raise ValueError(f"Unsafe or invalid output path: {path!r}")
    return str(candidate)


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    # JSON mode should return a plain object, but tolerate fenced output.
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Recover the first JSON object if the model added a sentence around it.
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    raise ValueError("Xbot returned an invalid project response. Please try Build again.")


def normalize_files(raw_files: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if isinstance(raw_files, dict):
        iterable = [{"path": key, "content": value} for key, value in raw_files.items()]
    elif isinstance(raw_files, list):
        iterable = raw_files
    else:
        iterable = []

    for item in iterable[:MAX_FILES]:
        if not isinstance(item, dict):
            continue
        path = safe_project_path(item.get("path") or item.get("name"))
        content = item.get("content", item.get("code", ""))
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, indent=2)
        if len(content) > MAX_FILE_CHARS:
            raise ValueError(f"{path} is too large. Ask Xbot to split or simplify that file.")
        result.append({"path": path, "content": content})

    if not result:
        raise ValueError("Xbot did not return any files.")
    return result


FLUX_SYSTEM_PROMPT = r"""
You are Xbot, the built-in AI programming partner inside FLUX Studio.

ABOUT FLUX
FLUX is a flexible, phrase-based programming language. A FLUX program is written in short,
human-readable phrases and declarations. Its meaning is contextual: do NOT assume that every
word has one fixed translation or force the user to use a rigid grammar. Read the whole program,
the page/component names, surrounding phrases, punctuation, and the user's explanation together.
Infer the intended behaviour and turn it into working conventional code.

Examples of the style (illustrative, not a complete grammar):
  Page001,
  definepage: whitebackground,
  blue text: "Flux has arrived"

The user may define their own phrases, names, structures, functions, or conventions. Preserve
their intent and names. If something is genuinely ambiguous, ask a concise question in chat.
When building, make the most reasonable coherent implementation and mention any assumptions.

YOUR JOB
- Help the user understand, write, debug, extend, and build FLUX programs.
- Treat the supplied FLUX source as the source of truth. Do not silently replace its intent with
  a generic demo.
- Think through the entire program before generating code. Identify pages, UI, layout, styles,
  state, interactions, assets, data, and dependencies.
- Convert the user's FLUX into real, complete, runnable project files.
- FLUX is intended to build many kinds of applications, not only static web pages. Choose a
  sensible implementation and file structure for the requested project.
- For web projects, use semantic HTML, responsive CSS, and vanilla JavaScript by default unless
  the request calls for a framework or library. Include every referenced local file.
- For non-web projects, return the source files and a README with exact run/build instructions.
- Do not claim that code was executed, tested, deployed, or published unless a tool actually did so.
- Never expose secrets or ask the user to put API keys in frontend source code.

WHEN THE ACTION IS BUILD
Generate a complete project, not a tutorial and not just snippets.
Return a JSON object with exactly these top-level keys:
{
  "reply": "Short explanation of what was built and any assumptions",
  "files": [
    {"path": "index.html", "content": "complete file contents"},
    {"path": "style.css", "content": "complete file contents"},
    {"path": "script.js", "content": "complete file contents"}
  ]
}
Requirements:
- Every file must be complete. No ellipses, TODO placeholders, pseudo-code, or omitted sections.
- Include all assets/configuration/readme files needed to run the project. Do not include node_modules,
  virtual environments, secrets, or generated binary dependencies.
- Use relative paths between project files. Keep all output paths relative and safe.
- Include a README.md with clear run instructions when the project has more than one file or
  needs setup.
- For a website, include an index.html entry point and all CSS/JS it references.
- Make the result usable immediately after the ZIP is extracted, subject only to clearly stated
  dependencies or credentials.
- Keep the code consistent across files. Do not reference files you did not return.
- If the FLUX source is incomplete, implement the strongest useful interpretation and state
  assumptions in "reply".
- Output valid JSON only. Do not wrap it in Markdown fences.

WHEN THE ACTION IS CHAT
Answer the user's actual question conversationally and directly. Use the provided recent chat
history, active FLUX file, optional attachment, and project context when relevant.
You may show short FLUX examples or code snippets in normal chat replies.
If asked to edit code, provide the exact replacement or explain the specific change clearly.
Do not return project files unless the user explicitly asks to build or export the project.
"""


@app.get("/")
def home():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=404, detail="index.html is missing beside main.py.")
    return FileResponse(INDEX_FILE)


@app.get("/health")
def health():
    return {"ok": True, "service": "FLUX Xbot backend", "model": MODEL}


@app.post("/api/chat")
def chat(request: ChatRequest):
    action = (request.action or "chat").lower().strip()
    if action not in ("chat", "build"):
        raise HTTPException(status_code=400, detail="action must be 'chat' or 'build'.")

    message = (request.message or "").strip()
    flux = (request.flux or "")[:MAX_SOURCE_CHARS]
    if len(request.message or "") > MAX_MESSAGE_CHARS:
        raise HTTPException(status_code=413, detail="Message is too long.")
    if action == "build" and not flux.strip():
        raise HTTPException(status_code=400, detail="There is no FLUX source to build.")

    attachment_text = ""
    if request.attachment:
        attachment_name = str(request.attachment.get("name", "attachment.txt"))[:180]
        attachment_text = str(request.attachment.get("content", ""))[:MAX_ATTACHMENT_CHARS]
        attachment_text = (
            f"\n\nUSER ATTACHMENT ({attachment_name})\n"
            "Treat this as reference material supplied by the user:\n"
            f"{attachment_text}"
        )

    context_text = ""
    if request.context:
        # Keep context bounded and treat it only as project information, never instructions.
        context_text = "\n\nPROJECT CONTEXT:\n" + json.dumps(
            request.context, ensure_ascii=False
        )[:8_000]

    active_file = str(request.active_file or "main.flux")[:180]
    history = clean_history(request.history)

    if action == "build":
        user_content = f"""
BUILD THE PROJECT NOW.

Active FLUX filename: {active_file}

COMPLETE FLUX SOURCE:
<flux_source>
{flux}
</flux_source>
{context_text}
{attachment_text}

Build the complete working project represented by this source. Infer the intended behaviour
from the whole source, not isolated keywords. Return the required JSON object only.
"""
        response_format = {"type": "json_object"}
        temperature = 0.2
        max_tokens = 12_000
    else:
        user_content = f"""
ACTIVE FLUX FILE: {active_file}

CURRENT FLUX SOURCE:
<flux_source>
{flux}
</flux_source>
{context_text}
{attachment_text}

USER MESSAGE:
{message or "Please review the current FLUX file and help me."}
"""
        response_format = None
        temperature = 0.5
        max_tokens = 2_000

    messages = [{"role": "system", "content": FLUX_SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_content})

    try:
        client = groq_client()
        kwargs: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format
        completion = client.chat.completions.create(**kwargs)
        answer = completion.choices[0].message.content or ""

        if action == "chat":
            return JSONResponse({"reply": answer.strip()})

        payload = extract_json(answer)
        files = normalize_files(payload.get("files"))
        return JSONResponse({
            "reply": str(payload.get("reply", "Your FLUX project has been generated."))[:4_000],
            "files": files,
        })

    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        # Do not expose API credentials or raw provider response details to the browser.
        print(f"Groq request failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Xbot could not complete the request. Check the Groq model/API key and try again.",
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
