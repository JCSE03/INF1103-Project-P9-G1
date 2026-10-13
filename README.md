# INF1103-P9-TEAM1: Helpdesk Ticket Prioritisation and Routing System (HTPRS)

## Problem Statement
In a large organisation such as a bank, the IT helpdesk receives a high volume of tickets that must be vetted and routed to the right team. As more systems (including AI tools) are introduced, the number of things that can go wrong keeps growing.

Vetting and routing every ticket by hand takes significant staff time. As ticket volume grows, this slows the IT team down, delays urgent incidents and hurts its KPIs.

Our IT helpdesk routing system:
1. Takes the requester's name, contact number, a ticket title and a free-text description of the IT issue, written in plain language.
2. Screens the text before it reaches the AI, rejecting prompt-injection attempts and non-helpdesk requests, blocking attempts to obtain privileged credentials, and flagging any mention of privileged accounts for identity verification.
3. Removes sensitive data such as passwords, emails, NRIC numbers and card numbers, so they are never sent to the AI. The requester's name and contact number are never sent to the AI either.
4. Sends the title and description to an AI (Google Gemini), which identifies the issue's category, severity and number of people affected, writes a short summary and gives a confidence score.
5. Checks the AI's response against a fixed format, retrying if it is malformed or the service is unavailable.
6. Calculates a priority score from a fixed scoring rubric and assigns a priority level from P1 (most urgent) to P5 (least urgent), with security incidents affecting multiple users always raised to P1.
7. Routes the ticket to the correct department: Cyber, HR, Finance or Infra.
8. Flags the ticket for human review when the AI is unsure (confidence below 60%) or a privileged account is involved.
9. Saves the processed ticket to a JSON file, storing only the AI's summary and never the user's raw description.
10. Shows a helpdesk dashboard with each department's queue, most urgent first, where staff mark tickets resolved or reopen them.

## Project Structure

| File | Role |
| --- | --- |
| `main.py` | Entry point: starts the web app, runs the tests or the demo |
| `io_manager.py` | Web routes, form validation and all console output |
| `ai_manager.py` | Security checks, redaction, Gemini call, response validation |
| `logic_manager.py` | Priority score, P1-P5, department routing, review flag, queue order |
| `data_manager.py` | Saves and loads `data/tickets.json`, query and resolve functions |
| `index.html` | Ticket submission form |
| `helpdesk.html` | Helpdesk dashboard (queues, resolve / reopen) |
| `test_*.py` | Offline tests for each manager (no API key needed) |
| `dummy_ai.py` | Demo run of the full pipeline with a fake Gemini |

## Setup

1. Clone the repo and pull the latest `main`
2. Copy the env template: `cp .env.example .env`
3. Get your own Gemini API key from https://aistudio.google.com/apikey (use your own Google account, don't share keys) and paste it into `.env`
4. Install dependencies: `pip install -r requirements.txt`

**Never commit `.env`**. It's gitignored on purpose. If `git status` ever shows `.env`, don't run `git add .` blindly; check first.

The free tier allows about 20 Gemini requests per day per key, so use the offline tests and the demo for day-to-day work.

## Running

| Command | What it does |
| --- | --- |
| `python main.py` | Starts the web app and opens the ticket form and the helpdesk dashboard |
| `python main.py --test` | Runs all offline tests (no API key or internet needed) |
| `python main.py --test --live` | Offline tests, then a few real Gemini calls to check the prompt (uses quota) |
| `python main.py --demo` | Runs the full pipeline with a fake Gemini and prints the results |
| `python main.py --no-browser` | Starts the web app without opening a browser |
| `python main.py --port 8000` | Uses a different port |

Once the app is running:
- Ticket form: http://localhost:5000
- Helpdesk dashboard: http://localhost:5000/helpdesk
- One department's queue: http://localhost:5000/departments/Infra (or Cyber, HR, Finance)

Tickets are saved in `data/tickets.json`, which is gitignored. Delete it to start with an empty queue.

## Docker

Docker Desktop must be running ("Engine running").

```bash
docker build -t htprs .
docker run --rm htprs python main.py --test
docker run --rm --env-file .env -p 5000:5000 -v "${PWD}/data:/app/data" htprs
```

1. Builds the image.
2. Runs all offline tests inside the container. No API key is needed.
3. Starts the web app. Open http://localhost:5000 and http://localhost:5000/helpdesk yourself (no browser opens from inside Docker). Press Ctrl+C to stop.

The `-v` option keeps tickets in your project's `data` folder, so they survive restarts and are shared with `python main.py`. Without it, the queue starts empty every time. In Command Prompt, use `-v "%cd%/data:/app/data"` instead.

## Team Workflow

1. Start from an up-to-date `main`: `git checkout main` then `git pull origin main`
2. Create a branch: `git checkout -b feature/short-description`
3. Commit small, focused changes with clear messages
4. Run `python main.py --test` before pushing
5. Push the branch and open a pull request; merge after a teammate approves
6. Delete the branch after merging

Don't upload files through the GitHub website: uploads ignore `.gitignore`, so `.env`, `data/` and `__pycache__/` can end up in the repo.