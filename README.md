<img width="4693" height="2648" alt="mermaid-diagram (1)" src="https://github.com/user-attachments/assets/a801338f-760b-4aef-9cb2-a47426c30216" />
# Munder Difflin Multi-Agent System Project

Welcome to the starter code repository for the **Munder Difflin Paper Company Multi-Agent System Project**! This repository contains the starter code and tools you will need to design, build, and test[...]

## Project Context

You’ve been hired as an AI consultant by Munder Difflin Paper Company, a fictional enterprise looking to modernize their workflows. They need a smart, modular **multi-agent system** to automate:

- **Inventory checks** and restocking decisions
- **Quote generation** for incoming sales inquiries
- **Order fulfillment** including supplier logistics and transactions

Your solution must use a maximum of **5 agents** and process inputs and outputs entirely via **text-based communication**.

This project challenges your ability to orchestrate agents using modern Python frameworks like `smolagents`, `pydantic-ai`, or `npcsh`, and combine that with real data tools like `sqlite3`, `pandas`, [...]

---

## What’s Included

From the `project.zip` starter archive, you will find:

- `project_starter.py`: The main Python script you will modify to implement your agent system
- `quotes.csv`: Historical quote data used for reference by quoting agents
- `quote_requests.csv`: Incoming customer requests used to build quoting logic
- `quote_requests_sample.csv`: A set of simulated test cases to evaluate your system

---

## Workspace Instructions

All the files have been provided in the VS Code workspace on the Udacity platform. Please install the agent orchestration framework of your choice.

## Local setup instructions

1. Install dependencies

Make sure you have Python 3.10 or later installed.

You can install all required packages using the provided requirements.txt file:

`pip install -r requirements.txt`

2. Create a `.env` file in the project root

Add your OpenAI-compatible API key and endpoint:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://openai.vocareum.com/v1
MODEL_ID=gpt-4o-mini
```

`OPENAI_BASE_URL` defaults to OpenAI's API URL; set it to the URL for any other provider, including the Vocareum endpoint shown above. `MODEL_ID` defaults to `gpt-4o-mini`. The API key is required only for the live agent run; the offline tests do not call an LLM.

## How to Run the Project

Run the offline business-rule tests:

```sh
python -m unittest discover -s tests -v
```

Run the full, live sample set:

```sh
python project_starter.py
```

The live run resets `munder_difflin.db`, processes every row in `quote_requests_sample.csv` in date order, and writes `test_results.csv`. It makes live model calls and requires a valid API key and endpoint.

Output will include:

- Agent responses
- Cash and inventory updates
- Final financial report
- A `test_results.csv` file with all interaction logs

---

## Tips for Success

- Start by sketching a **flow diagram** to visualize agent responsibilities and interactions.
- Test individual agent tools before full orchestration.
- Always include **dates** in customer requests when passing data between agents.
- Ensure every quote includes **bulk discounts** and uses past data when available.
- Use the **exact item names** from the database to avoid transaction failures.

---

## Submission Checklist

Make sure to submit the following files:

1. Your completed `template.py` or `project_starter.py` with all agent logic
2. A **workflow diagram** describing your agent architecture and data flow
3. A README or design notes explaining how your system works
4. Outputs from your test run (like `test_results.csv`)

---
