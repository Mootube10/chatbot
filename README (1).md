# Autonomous AI Case Triage & Account Provisioning System

## Included Files
1. `main.py` - FastAPI backend, Gemini AI triage engine, SQLite database, and live Zendesk Tickets REST API integration.
2. `index.html` - Customer-facing AI intake chatbot.
3. `portal.html` - User account portal where customers view attached cases and message Engineering/Technical teams directly.
4. `dashboard.html` - Engineering & Specialist team console (bypasses Customer Service for technical issues).
5. `requirements.txt` - Python dependencies for deployment.

## How to Run Locally or Deploy (Render / Railway)
1. Install dependencies:
   `pip install -r requirements.txt`
2. Set your environment variables (optional for offline testing, required for live AI & Zendesk):
   - `GEMINI_API_KEY` = Your free key from Google AI Studio
   - `ZENDESK_SUBDOMAIN` = Your Zendesk subdomain (e.g., `acme` for `acme.zendesk.com`)
   - `ZENDESK_EMAIL` = Your Zendesk admin email
   - `ZENDESK_API_TOKEN` = Your Zendesk API token
3. Start the server:
   `uvicorn main:app --host 0.0.0.0 --port 8000`
