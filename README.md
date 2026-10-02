# INF1103-P9-TEAM1

# Logic Manager (Johanan & Shaun)
### Based on the AI Mananger, the AI will return a JSON containing things such as:

#### Severity:
```
Low (1) | Medium (2) | High (3) | Critical (4)
```
#### Affected Scope:
```
Individual (1) | Multiple People (2) | Department (3) | Organization (4)
```
#### Confidence Score:
```
0.00 (No Confidence) - 1.00 (Full Confidence)
```
#### Department:
```
Cyber | HR | Finance | Infra
```

#### In order to calculate the priority score, we will use the following fomulae
```
Severity Score * Affected Scope Score = Priority Score
P1: 15-16
P2: 13-14
P3: 8-12
P4: 4-7
P5: 1-3
```
#### The logic manager will insert the tickets into a queue system and send it to the respective departments

#### If the confidence score is below a threshold of 0.5, we will include a note at the top of the ticket that says 
```
Notice: Confidence Score < 0.5 | Please re-route if needed
```
 
## Setup

1. Clone the repo and pull the latest `main` (or your feature branch)
2. Copy the env template: `cp .env.example .env`
3. Get your own Gemini API key from https://aistudio.google.com/apikey (use your own Google account, don't share keys) and paste it into `.env`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `python main.py` (or `python io_manager.py`, whichever is the entry point)

**Never commit `.env`**. It's gitignored on purpose. If `git status` ever shows `.env`, don't run `git add .` blindly; check first.

The free tier allows about 20 Gemini requests per day per key. Use the offline tests (`python test_ai.py`) for day-to-day work, since they make no API calls.


