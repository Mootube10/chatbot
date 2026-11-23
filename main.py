import os
import json
import sqlite3
import uuid
import base64
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx

app = FastAPI(title="Autonomous AI Case Triage & Account Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = os.getenv("DB_PATH", "cases.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
ZENDESK_SUBDOMAIN = os.getenv("ZENDESK_SUBDOMAIN", "").strip()
ZENDESK_EMAIL = os.getenv("ZENDESK_EMAIL", "").strip()
ZENDESK_API_TOKEN = os.getenv("ZENDESK_API_TOKEN", "").strip()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_accounts (
            id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            company TEXT,
            account_tier TEXT DEFAULT 'Standard',
            created_at TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS support_cases (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            case_type TEXT NOT NULL,
            assigned_team TEXT NOT NULL,
            subject TEXT NOT NULL,
            summary TEXT NOT NULL,
            status TEXT NOT NULL,
            external_platform TEXT,
            external_ticket_id TEXT,
            external_ticket_url TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES user_accounts(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS case_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id TEXT NOT NULL,
            sender_role TEXT NOT NULL,
            sender_name TEXT NOT NULL,
            message TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(case_id) REFERENCES support_cases(id)
        )
    """)
    conn.commit()
    conn.close()


init_db()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    history: List[dict] = []


class ReplyRequest(BaseModel):
    sender_role: str
    sender_name: str
    message: str


async def push_to_zendesk_api(full_name: str, email: str, subject: str, summary: str, transcript: str) -> dict:
    if not (ZENDESK_SUBDOMAIN and ZENDESK_EMAIL and ZENDESK_API_TOKEN):
        mock_id = f"ZD-{uuid.uuid4().hex[:6].upper()}"
        return {
            "mode": "simulation",
            "ticket_id": mock_id,
            "ticket_url": f"https://demo-company.zendesk.com/agent/tickets/{mock_id}",
            "handoff_url": f"https://demo-company.zendesk.com/hc/en-us?prefill_email={email}&case_ref={mock_id}"
        }

    url = f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/api/v2/tickets.json"
    auth_str = f"{ZENDESK_EMAIL}/token:{ZENDESK_API_TOKEN}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    payload = {
        "ticket": {
            "subject": f"[AI Pre-Triaged] {subject}",
            "comment": {
                "body": (
                    f"AI Triage Summary:\n{summary}\n\n"
                    f"--- Full Pre-Collected Chat Transcript ---\n{transcript}"
                ),
                "public": False
            },
            "requester": {
                "name": full_name,
                "email": email
            },
            "tags": ["ai_triaged", "general_knowledge_handoff"],
            "priority": "normal"
        }
    }

    headers = {
        "Authorization": f"Basic {b64_auth}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code in (200, 201):
            data = resp.json().get("ticket", {})
            tid = str(data.get("id", "UNKNOWN"))
            return {
                "mode": "live",
                "ticket_id": tid,
                "ticket_url": f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/agent/tickets/{tid}",
                "handoff_url": f"https://{ZENDESK_SUBDOMAIN}.zendesk.com/hc/en-us/requests/{tid}"
            }
        else:
            raise HTTPException(status_code=502, detail=f"Zendesk API Error: {resp.text}")


async def run_ai_triage(history: List[dict], latest_message: str) -> dict:
    system_prompt = """
    You are an autonomous AI Intake & Triage Specialist for a B2B software support system.
    Your job is to collect 4 details from the user before taking action:
    1. Full Name
    2. Email Address
    3. Company Name (optional, default to 'Individual' if not given)
    4. Their exact issue or request

    Once you have their Name, Email, and Issue, classify the request into one of two categories:
    - "TECHNICAL": Bugs, API failures, account setup/provisioning errors, system outages, integrations, or technical configuration. (These bypass Customer Service completely and go straight to Engineering/Technical Teams attached to their user account).
    - "GENERAL": Billing inquiries, general product questions, pricing, password resets, or general policy questions. (These transfer straight to a Zendesk Customer Service Advisor with all details pre-filled).

    Respond ONLY with valid JSON in this exact schema:
    {
      "ready_to_execute": true or false,
      "reply_to_user": "Friendly conversational response asking for missing info OR confirming action taken",
      "full_name": "extracted or null",
      "email": "extracted or null",
      "company": "extracted or 'Individual'",
      "case_type": "TECHNICAL" or "GENERAL" or null,
      "assigned_team": "Engineering Team" or "Technical Support" or "Customer Service Advisor" or null,
      "subject": "Short 5-word case title or null",
      "summary": "Detailed summary of what the user needs or null"
    }
    """

    if not GEMINI_API_KEY:
        full_text = " ".join([m.get("content", "") for m in history] + [latest_message])
        words = full_text.split()
        email = next((w.strip(".,!?") for w in words if "@" in w and "." in w), None)
        if not email:
            return {
                "ready_to_execute": False,
                "reply_to_user": "Hello! I can create your account and route your case straight to the right team. Could you please share your **Full Name**, **Email Address**, and describe what you need help with?",
                "full_name": None,
                "email": None,
                "company": "Individual",
                "case_type": None,
                "assigned_team": None,
                "subject": None,
                "summary": None
            }
        is_tech = any(k in full_text.lower() for k in ["bug", "api", "error", "broken", "code", "integration", "server", "account", "technical", "crash"])
        return {
            "ready_to_execute": True,
            "reply_to_user": (
                "I have provisioned your account and routed this technical case directly to our Engineering Team—bypassing Customer Service completely."
                if is_tech else
                "I have collected all your details and created a pre-triaged ticket directly with a Zendesk Customer Service Advisor so you won't have to repeat anything."
            ),
            "full_name": email.split("@")[0].replace(".", " ").title(),
            "email": email,
            "company": "Demo Corp",
            "case_type": "TECHNICAL" if is_tech else "GENERAL",
            "assigned_team": "Engineering Team" if is_tech else "Customer Service Advisor",
            "subject": "Technical Integration & Account Issue" if is_tech else "General Account & Product Inquiry",
            "summary": f"User reported: {latest_message}"
        }

    conversation_text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in history])
    conversation_text += f"\nuser: {latest_message}"

    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    payload = {
        "contents": [{
            "parts": [{"text": f"{system_prompt}\n\nConversation so far:\n{conversation_text}"}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(gemini_url, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Gemini API Error: {resp.text}")
        raw_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw_text)


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    ai_result = await run_ai_triage(req.history, req.message)

    if not ai_result.get("ready_to_execute"):
        return {"status": "collecting", "reply": ai_result["reply_to_user"]}

    conn = get_db()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    email = ai_result["email"].lower().strip()
    full_name = ai_result.get("full_name") or email.split("@")[0]
    company = ai_result.get("company") or "Individual"

    cur.execute("SELECT * FROM user_accounts WHERE email = ?", (email,))
    existing_user = cur.fetchone()
    if existing_user:
        user_id = existing_user["id"]
        account_created = False
    else:
        user_id = f"ACC-{uuid.uuid4().hex[:8].upper()}"
        cur.execute(
            "INSERT INTO user_accounts (id, full_name, email, company, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, full_name, email, company, now)
        )
        account_created = True

    case_id = f"CASE-{uuid.uuid4().hex[:6].upper()}"
    case_type = ai_result.get("case_type", "TECHNICAL")
    assigned_team = ai_result.get("assigned_team", "Engineering Team")
    subject = ai_result.get("subject", "Support Request")
    summary = ai_result.get("summary", req.message)

    transcript = "\n".join([f"{m.get('role')}: {m.get('content')}" for m in req.history] + [f"user: {req.message}"])

    ext_platform = None
    ext_ticket_id = None
    ext_ticket_url = None
    handoff_url = None

    if case_type == "GENERAL":
        zd_res = await push_to_zendesk_api(full_name, email, subject, summary, transcript)
        ext_platform = f"Zendesk ({zd_res['mode']})"
        ext_ticket_id = zd_res["ticket_id"]
        ext_ticket_url = zd_res["ticket_url"]
        handoff_url = zd_res["handoff_url"]
        status = "Transferred to Zendesk Advisor"
    else:
        ext_platform = "Internal Specialist Queue"
        status = "Open - Assigned to Engineering"

    cur.execute("""
        INSERT INTO support_cases (
            id, user_id, case_type, assigned_team, subject, summary,
            status, external_platform, external_ticket_id, external_ticket_url, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        case_id, user_id, case_type, assigned_team, subject, summary,
        status, ext_platform, ext_ticket_id, ext_ticket_url, now
    ))

    cur.execute(
        "INSERT INTO case_messages (case_id, sender_role, sender_name, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (case_id, "customer", full_name, req.message, now)
    )
    cur.execute(
        "INSERT INTO case_messages (case_id, sender_role, sender_name, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (case_id, "system", "AI Triage Engine", f"Account {user_id} verified. Routed to {assigned_team}. Summary: {summary}", now)
    )

    conn.commit()
    conn.close()

    return {
        "status": "executed",
        "reply": ai_result["reply_to_user"],
        "account_id": user_id,
        "account_created": account_created,
        "case_id": case_id,
        "case_type": case_type,
        "assigned_team": assigned_team,
        "portal_url": f"/portal?email={email}",
        "zendesk_handoff_url": handoff_url,
        "zendesk_ticket_id": ext_ticket_id
    }


@app.get("/api/cases")
def list_all_cases():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        SELECT c.*, u.full_name, u.email, u.company
        FROM support_cases c
        JOIN user_accounts u ON c.user_id = u.id
        ORDER BY c.created_at DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        cur.execute("SELECT * FROM case_messages WHERE case_id = ? ORDER BY id ASC", (r["id"],))
        r["messages"] = [dict(m) for m in cur.fetchall()]
    conn.close()
    return rows


@app.get("/api/portal/{email}")
def user_portal_data(email: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM user_accounts WHERE email = ?", (email.lower().strip(),))
    user = cur.fetchone()
    if not user:
        conn.close()
        raise HTTPException(status_code=404, detail="No account found for this email.")
    cur.execute("SELECT * FROM support_cases WHERE user_id = ? ORDER BY created_at DESC", (user["id"],))
    cases = [dict(c) for c in cur.fetchall()]
    for c in cases:
        cur.execute("SELECT * FROM case_messages WHERE case_id = ? ORDER BY id ASC", (c["id"],))
        c["messages"] = [dict(m) for m in cur.fetchall()]
    conn.close()
    return {"account": dict(user), "cases": cases}


@app.post("/api/cases/{case_id}/reply")
def post_case_reply(case_id: str, req: ReplyRequest):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM support_cases WHERE id = ?", (case_id,))
    if not cur.fetchone():
        conn.close()
        raise HTTPException(status_code=404, detail="Case not found")
    now = datetime.utcnow().isoformat()
    cur.execute(
        "INSERT INTO case_messages (case_id, sender_role, sender_name, message, created_at) VALUES (?, ?, ?, ?, ?)",
        (case_id, req.sender_role, req.sender_name, req.message, now)
    )
    if req.sender_role == "engineer":
        cur.execute("UPDATE support_cases SET status = 'Answered by Specialist' WHERE id = ?", (case_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


def read_html(filename: str) -> str:
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    return f"<h1>{filename} not found</h1>"


@app.get("/", response_class=HTMLResponse)
def serve_chatbot():
    return read_html("index.html")


@app.get("/portal", response_class=HTMLResponse)
def serve_portal():
    return read_html("portal.html")


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard():
    return read_html("dashboard.html")
