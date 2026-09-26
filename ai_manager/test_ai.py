from ai_manager import (
    build_prompt,
    call_api,
    parse_response,
    validate_response
)


sample_records = [
    {
        "ticket_title": "Forgot password",
        "problem_description": "I cannot log in and I think I forgot my password.",
        "affected_service": "Employee account",
        "affected_users": 1,
        "work_blocked": True,
        "workaround_available": False
    },

    {
        "ticket_title": "Branch WiFi down",
        "problem_description": "Staff in the whole branch cannot connect to the office WiFi.",
        "affected_service": "Office WiFi",
        "affected_users": 35,
        "work_blocked": True,
        "workaround_available": False
    },

    {
        "ticket_title": "Suspicious email",
        "problem_description": "I received an email asking me to click a link and verify my bank login details.",
        "affected_service": "Email",
        "affected_users": 1,
        "work_blocked": False,
        "workaround_available": True
    },

    {
        "ticket_title": "Core banking system unavailable",
        "problem_description": "Employees across multiple branches cannot access the core banking system.",
        "affected_service": "Core banking application",
        "affected_users": 200,
        "work_blocked": True,
        "workaround_available": False
    },

    {
        "ticket_title": "Laptop screen not working",
        "problem_description": "My laptop turns on but the screen remains black.",
        "affected_service": "Laptop",
        "affected_users": 1,
        "work_blocked": True,
        "workaround_available": True
    }
]


for record in sample_records:

    print("\n==============================")
    print("TESTING:", record["ticket_title"])
    print("==============================")

    prompt = build_prompt(record)

    raw_response = call_api(prompt)

    if raw_response is None:
        print("Gemini API call failed.")
        continue

    parsed_response = parse_response(raw_response)

    print("AI RESPONSE:")
    print(parsed_response)

    if validate_response(parsed_response):
        print("Validation: VALID")
    else:
        print("Validation: INVALID")



print("\n--- Testing invalid responses ---")
bad_response = {
    "category": "account_access",
    "issue_type": "forgotten_credentials",
    "technical_severity": "SUPER_SERIOUS",
    "affected_scope": "individual",
    "summary": "User forgot password.",
    "confidence_score": 0.98
}

bad_response_2 = {
    "category": "account_access",
    "issue_type": "forgotten_credentials",
    "technical_severity": "low",
    "affected_scope": "individual",
    "summary": "User forgot password.",
    "confidence_score": 5
}

print(validate_response(bad_response))
print(validate_response(bad_response_2))