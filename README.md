# INF1103-P9-TEAM1

## Problem Statement
 In a large organization like a financial institute or bank, the IT helpdesk team recieves a high volume of tickets which has to be vetted and routed to the appropriate team for remedy. 
 Especially in this current technology age where AI is incorporated into more and more things, there is a trend of a growing surface area for issues to arise.

 This would mean that meaningful time has to be taken from a manpower to vet and route the tickets to the appropriate department. As the number of tickets grows larger and larger, this time taken can siginificantly affect the efficiently of the IT team which wil result in lower KPIs being met.

 Our team has decided to increase value for the shareholders by creating an IT helpdesk routing system which will:
  1. Takes a free-text description from an internal bank staff member, written in plain language, of the IT issue they are facing.
  2. Screens the description before it reaches the AI, rejecting prompt-injection attempts and non-helpdesk requests, blocking attempts to obtain privileged credentials, and flagging any mention of privileged accounts for identity verification.
  3. Removes sensitive data such as passwords, emails, NRIC numbers and card numbers, so they are never sent to the AI.
  4. Sends the description to an AI (Google Gemini), which identifies the issue's category, severity and number of people affected, writes a short summary and gives a confidence score.
  5. Checks the AI's response against a fixed format, retrying if it is malformed or the service is unavailable.
  6. Calculates a priority score from a fixed scoring rubric and assigns a priority level from P1 (most urgent) to P5 (least urgent), with security incidents affecting multiple users always raised to P1.
  7. Routes the ticket to the correct department: Cyber, HR, Finance or Infra.
  8. Flags the ticket for human review when the AI is unsure (confidence below 60%) or a privileged account is involved.
  9. Saves the processed ticket to a file, storing only the AI's summary and never the user's raw description.
  10. Displays a ticket queue, filterable by department and ordered by priority, so the most urgent tickets are handled first.

 
## Setup

1. Clone the repo and pull the latest `main` (or your feature branch)
2. Copy the env template: `cp .env.example .env`
3. Get your own Gemini API key from https://aistudio.google.com/apikey (use your own Google account, don't share keys) and paste it into `.env`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python main.py` (or `python io_manager.py`, whichever is the entry point)

**Never commit `.env`**. It's gitignored on purpose. If `git status` ever shows `.env`, don't run `git add .` blindly; check first.

The free tier allows about 20 Gemini requests per day per key. Use the offline tests (`python test_ai.py`) for day-to-day work, since they make no API calls.


