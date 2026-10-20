# Beaver's Choice Paper Company – Multi-Agent System Report

Framework: **smolagents** (`ToolCallingAgent` + managed agents). Single source file: `project_starter.py`. Diagram: `workflow_diagram.png`.

## Architecture (4 agents, limit is 5)
| Agent | Responsibility | Tools |
|---|---|---|
| Orchestrator | Fallback only: handles a request if intake output can't be parsed; otherwise the pipeline in code coordinates the specialists | managed agents |
| Inventory agent | Reports stock vs. reorder threshold (read-only; cannot reorder) | `find_catalog_item`, `check_stock`, `list_inventory` |
| Quoting agent | Prices orders using volume discounts and past quotes as precedent | `find_catalog_item`, `search_past_quotes`, `calculate_quote` |
| Ordering agent | Finalizes sales: restock if short, record transactions, report fulfilment date | `fulfill_order`, `get_finances` |

Flow: request → **intake (LLM)** extracts `{requested wording, catalog name or null, quantity}` → **code** validates each catalog name, decides order vs. quote, and calls the specialist agents' tools (`check_stock` → `calculate_quote` → `fulfill_order` for orders) → code writes the customer reply. The LLM never decides whether a sale is recorded, and never writes prices or delivery dates. If intake fails to parse, the orchestrator agent handles the request as a fallback.

## Business rules (all enforced in tools, not left to the LLM)
- Catalog names are matched exactly (case-insensitive); `difflib` is used only to *suggest* names. The intake model may only pick a name from the catalog list, and code additionally requires the customer's wording to share a distinctive word with it. Anything else is reported as not supplied; nothing is substituted.
- Order vs. quote is decided in code: a request is an order unless it asks for a quote/estimate without any order wording. Quote-only requests never touch the database.
- Pricing: list price with volume discounts: ≥200 units 4%, ≥500 7%, ≥1000 10%.
- Supplier cost is 70% of list price; delivery lead time uses `get_supplier_delivery_date`; stock-order cash is booked on the delivery date.
- Restock when stock is short or would fall below `min_stock_level` (order shortfall + 1.5× the top-up); refused if cash would drop below a $1,000 reserve.
- Sales are recorded on the first date the goods are available, so inventory/cash reports stay consistent with delivery timelines. Partial/unfulfillable lines are explicitly reported.
- Delivery deadlines: the requested date ("by April 15, 2025"; the latest date mentioned; year defaults to the request year) is parsed in code and compared with each line's actual delivery date. Every line states whether it meets the deadline or is LATE (by how many days), and a closing note lists late items. Late orders are still recorded, and the customer is told. Quotes show an estimated delivery date the same way, computed read-only (no restock placed).
- When stock is short, the restock is the shortfall plus 1.5x the minimum level (the shortfall is not counted twice), which keeps orders in the shorter supplier lead-time tiers.
- All DB access uses the provided helpers (parameterized SQL).

## Configuration and running
`pip install -r requirements.txt`, create `.env` with `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`, `MODEL_ID`), then `python project_starter.py`. It resets the DB, runs every row of `quote_requests_sample.csv` in date order, prints cash/inventory after each request and writes `test_results.csv`.

## Fixes to the starter
`init_database()` was called without its engine argument, and `response` was undefined; both fixed.

## Testing status
The full sample set (20 requests) was run end to end against the Vocareum endpoint (`OPENAI_BASE_URL=https://openai.vocareum.com/v1`) with no errors and no intake fallbacks; every order request was routed to fulfilment and unknown items (e.g. balloons, streamers, A3 paper) were reported rather than substituted. `test_results.csv` holds the per-request responses and cash/inventory after each.

## Possible improvements
Offer partial or alternative-date fulfilment when a deadline can't be met (currently late orders are recorded and flagged); per-customer pricing history; learned reorder points from sales velocity.
