# Beaver's Choice Paper Company – Multi-Agent System Report

Framework: **smolagents** (`ToolCallingAgent` + managed agents). Single source file: `project_starter.py`. Diagram: `workflow_diagram.png`.

## Architecture (4 agents, limit is 5)
| Agent | Responsibility | Tools |
|---|---|---|
| Orchestrator | Parses the request (items, quantities, date), delegates, writes the customer reply, hides internal costs/DB details | managed agents |
| Inventory agent | Maps customer wording to catalog names, reports stock vs. reorder threshold, reorders from supplier | `find_catalog_item`, `check_stock`, `list_inventory`, `restock_item` |
| Quoting agent | Prices orders using volume discounts and past quotes as precedent | `find_catalog_item`, `search_past_quotes`, `calculate_quote` |
| Ordering agent | Finalizes sales: restock if short, record transactions, report fulfilment date | `fulfill_order`, `get_finances` |

Flow: request → orchestrator → inventory (names/stock) → quoting → ordering (if the customer is buying) → orchestrator reply.

## Business rules (all enforced in tools, not left to the LLM)
- Fuzzy catalog matching (`difflib`); unknown items are never quoted or sold, and are reported.
- Pricing: list price with volume discounts: ≥200 units 4%, ≥500 7%, ≥1000 10%.
- Supplier cost is 70% of list price; delivery lead time uses `get_supplier_delivery_date`; stock-order cash is booked on the delivery date.
- Restock when stock is short or would fall below `min_stock_level` (order shortfall + 1.5× the top-up); refused if cash would drop below a $1,000 reserve.
- Sales are recorded on the first date the goods are available, so inventory/cash reports stay consistent with delivery timelines. Partial/unfulfillable lines are explicitly reported.
- All DB access uses the provided helpers (parameterized SQL).

## Configuration and running
`pip install -r requirements.txt`, create `.env` with `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `MODEL_ID`), then `python project_starter.py`. It resets the DB, runs every row of `quote_requests_sample.csv` in date order, prints cash/inventory after each request and writes `test_results.csv`.

## Fixes to the starter
`init_database()` was called without its engine argument, and `response` was undefined; both fixed.

## Testing status
The tools were exercised directly against a fresh database (quote, multi-line fulfilment with a supplier restock and delayed sale date, finance report, inventory list) and behaved as designed. The LLM-driven run over the sample CSV requires an API key, which was not available in the build environment, so `test_results.csv` has to be generated on your machine.

## Possible improvements
Parse delivery deadlines and flag impossible ones; per-customer pricing history; learned reorder points from sales velocity.
