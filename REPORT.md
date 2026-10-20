# Beaver's Choice Paper Company – Multi-Agent System Report

Framework: **smolagents** (`ToolCallingAgent` with `managed_agents`). Single source file: `project_starter.py`. Diagram: `workflow_diagram.jpg`. Evaluation output: `test_results.csv`.

## 1. Business problem and objectives
Beaver's Choice receives free-text requests for paper products ("200 sheets of A4 glossy paper by April 15"). Staff have to work out which catalog items are meant, check stock, price the order, decide whether the supplier can deliver in time, and record the sale. The aim is a text-in, text-out multi-agent system (at most 5 agents) that automates this. It must:
- quote consistently using volume discounts and historical quotes;
- confirm only orders it can actually fulfil by the customer's date;
- restock sensibly without running cash below a safe level;
- keep the SQLite inventory and cash records consistent;
- explain clearly to the customer anything it cannot supply.

## 2. Architecture: 4 agents (limit 5)

| Agent | Responsibility | Tools → starter helpers |
|---|---|---|
| **Orchestrator** (`orchestrator`) | Receives the validated plan, delegates to the specialists in a fixed order (stock → price → order), and never contacts the ordering agent for quote-only requests | managed agents: `inventory_agent`, `quoting_agent`, `ordering_agent` |
| **Inventory agent** | Reports stock vs. `min_stock_level` (read-only) | `find_catalog_item`; `check_stock` → `get_stock_level()`; `list_inventory` → `get_all_inventory()` |
| **Quoting agent** | Prices the lines with volume discounts and checks historical quotes for precedent (read-only) | `find_catalog_item`; `search_past_quotes` → `search_quote_history()`; `calculate_quote` |
| **Sales & fulfilment agent** (`ordering_agent`) | Checks the deadline, restocks if needed, records the sale | `fulfill_order` → `get_supplier_delivery_date()`, `restock_item` (→ `get_cash_balance()`, `create_transaction()`), `create_transaction()`; `get_finances` → `generate_financial_report()` |

All seven required helpers are used on the request path. `generate_financial_report()` is also used by the evaluation loop after every request.

**Why four agents.** Each specialist has a clear, separate job: inventory and quoting only read data, while fulfilment is the only agent that writes. Keeping the writer separate means a quote can never change the database. A fifth agent (for example a separate restocking agent) was considered and rejected. Restocking only ever happens as part of fulfilling a sale, so splitting it out would add another hand-off for the LLM to get wrong without adding any capability.

**Request flow and hand-offs**
1. *Intake (LLM, temperature 0):* the model turns the request into `{customer wording, catalog name or null, quantity}` lines. It matches on product type and ignores quality, colour, finish or standard-size words ("A4 glossy paper" → Glossy paper). It returns null when the product or size isn't carried (A3 paper, balloons, tickets).
2. *Validation (code):* each catalog name must match exactly and share a distinctive word with the customer's wording, with a small synonym map such as "printer" → "copy". Anything else is "not carried". Code also decides order vs. quote, parses the customer's deadline, and adds together repeated lines for the same item.
3. *Delegation (orchestrator agent):* receives the validated plan and calls `inventory_agent`, then `quoting_agent`, then (orders only) `ordering_agent`.
4. *Enforcement (tools):* the tools write their results to a per-request ledger (`REQUEST`).
   - `fulfill_order` refuses quote-only requests and items not in the plan.
   - It always uses the validated quantity and processes each line once.
   - So an agent mistake cannot create an incorrect sale.
5. *Guards (code):* if an agent skipped a step, the same tool is run directly, and this is logged in the `agent_guards` column.
6. *Reply (code):* built from the tool results, so prices and dates are always exact. It never includes supplier costs, margins, cash figures or system errors; internal errors return a polite generic message.

**How the diagram matches the code.**
- The four boxes are the four `ToolCallingAgent` objects.
- The numbered arrows ①②③ are the orchestrator's delegation order.
- Each tool is listed next to the starter helper it calls.
- The database arrows show that only the fulfilment agent writes.
- The decision diamond is the per-line outcome inside `_fulfill_line`: FULFILL, DEFER or REJECT.

## 3. Framework and database
- **smolagents:** tools are defined with `@tool` (typed arguments and docstrings), agents with `ToolCallingAgent`, and the orchestrator uses `managed_agents`. The model is `OpenAIServerModel`, defaulting to `gpt-4o-mini`, with `temperature=0` so runs are reproducible, via the Vocareum OpenAI-compatible endpoint.
- **SQLite:** `munder_difflin.db` is created by the starter's `init_database()`. Stock and cash are derived from the `transactions` table: `stock_orders` add units and cost cash, `sales` remove units and add cash. The starter's assumptions are kept: transactions are dated, and stock and cash are always computed "as of" a date.

## 4. Business rules
- **Pricing:** list price × (1 − volume discount): 4% for ≥200 units, 7% for ≥500, 10% for ≥1000. The same function prices both quotes and sales, so discounts are always consistent.
- **Historical quotes:** `search_past_quotes` returns similar past requests and their explanations. These are used as precedent and context; the quoting agent decides when to consult them. In the sample data, past quotes also discounted bulk orders, which supports the tiered discounts above. The final price comes from the deterministic formula so it can be checked.
- **Stock and restocking:**
  - If stock is short, use stock already on order when it arrives in time. Otherwise order the shortfall plus 1.5 × `min_stock_level` from the supplier, at 70% of list price.
  - If a sale would take stock below the minimum level, top it up.
  - A restock is refused if it would take cash below a $1,000 reserve.
- **Delivery feasibility:** the earliest delivery date (from `get_supplier_delivery_date()` lead times) is compared with the customer's deadline *before* anything is written. If it is late, nothing is restocked or sold, and the customer is offered the earliest date instead.
- **Record dates:** sales are recorded on the date the goods are available, and stock orders on their delivery date.

## 5. Results from `test_results.csv` (all 20 sample requests)
Starting position: cash $45,059.70, inventory $4,940.30. Final report (as of 2025-04-24, the last delivery): **cash $45,831.43, inventory $4,676.60**.

The run was checked for integrity: the database was reset once, and its April transactions total exactly what the CSV attributes to the requests ($2,787.57 sales, $2,015.84 restocking).

| Outcome | Requests | Evidence |
|---|---|---|
| Fully fulfilled | **5** (#1, 4, 7, 10, 12) | e.g. #1: 200 Glossy paper $38.40 (4% off) + 100 Cardstock $15.00 + 100 Colored paper $10.00, all delivered 2025-04-01, before the 2025-04-15 deadline |
| Partially fulfilled | **8** (#2, 3, 5, 6, 8, 14, 19, 20) | Confirmed lines recorded, missing lines explained; e.g. #20: 5,000 Flyers $675.00 + 2,000 Poster paper $450.00 (10% off), "10,000 tickets" not carried |
| Unfulfilled | **7** (#9, 11, 13, 15, 16, 17, 18) | e.g. #13: "500 x A4 paper (earliest delivery 2025-04-12, after your 2025-04-10 deadline)… nothing has been charged"; #11: A3 glossy paper is not carried |
| Errors | 0 | No internal errors; no fallback replies |

- **Cash changes:** 13 requests booked transactions (non-zero `net_cash_effect`), and `cash_balance` changed at the request date in 8 rows. Total sales booked were $2,787.57 against $2,015.84 of restocking, a net cash gain of $771.73.
- **Unfulfilled requests left the books untouched:** all 7 have `sales_booked = 0` and `stock_purchases_booked = 0`.
- **Pricing checks:**
  - #3: 10,000 A4 paper at $0.05 with 10% off = $450.00.
  - #4: 500 Cardstock at $0.15 with 7% off = $69.75.
  - #12: 100 Paper napkins below the 200-unit tier = $2.00 with no discount.
- **Delivery checks:** every confirmed line's date is on or before the deadline. #19 confirmed 1,000 Cardstock for 2025-04-19 (deadline 2025-04-20) but deferred 2,000 Glossy paper, whose 7-day restock lands on 2025-04-22.
- **Agents did the work:**
  - Of the 19 requests with at least one valid item, the orchestrator delegated all 19 to `inventory_agent` and `quoting_agent`, and called `ordering_agent` in 14.
  - #11 had no item the catalog carries, so no agents were needed.
  - Guards were needed in 5 requests (#2, 6, 8, 9, 20): the ordering agent didn't complete every order line, so the code ran `fulfill_order` for the rest. The quote never had to be recalculated.
- **Offline tests:** 14 tests (`python3 -m unittest discover -s tests -v`) pass. They cover:
  - discount boundaries, exact catalog matching, synonyms and no substitutions;
  - quote-only requests never writing to the database;
  - orders placed through the agents, and late orders deferred with no database writes;
  - guards completing an order when the agents fail;
  - `fulfill_order` refusing anything outside the validated order;
  - restocking, pending deliveries and the cash reserve;
  - internal errors being hidden from customers.

## 6. Strengths and limitations shown by testing
**Strengths**
- No false confirmations: every "confirmed" line meets its deadline, and the 7 unfulfilled requests changed nothing.
- Discounts were applied consistently in every confirmed line.
- Products the catalog doesn't carry (balloons, streamers, tickets, A3 and A5 paper, poster board) were reported, never substituted.
- Customer replies contain no costs, margins or errors.

**Limitations**
- *Item matching is still somewhat conservative.* With the updated intake rules, "A4 glossy paper" (#1), "white printer paper" (#6) and "high-quality recycled cardstock" (#4) now match. But "A4 matte paper" (#8, #11), "colorful construction paper" (#6, #16), "500 reams of printer paper" (#3) and "biodegradable paper cups/plates" (#17) were still treated as not carried, although the catalog has Matte paper, Construction paper, Standard copy paper, Paper cups and Paper plates.
- *Deadlines are tight relative to supplier lead times.* Orders over 1,000 units need 7 days. 8 of the 20 requests had at least one deferred line. Six of the 7 unfulfilled requests had at least one line deferred for timing; #11 contained only items that aren't carried.
- *The ordering agent is not fully reliable.* It left at least one order line unfulfilled in 5 requests and the code had to complete it. The guards keep results correct, but LLM delegation alone is not dependable.
- *Some sales show up in `cash_balance` later.* Sales are dated on delivery, so `cash_balance` at the request date lags (it changed in 8 rows, against 13 requests that booked cash). The `sales_booked` and `net_cash_effect` columns attribute amounts to the right request.
- *Concurrent runs corrupt results.* Each run resets the shared `munder_difflin.db`. Two runs at once write into each other's database, as happened in earlier attempts, so the evaluation must run alone.

## 7. Improvements
1. **Deterministic item matching with confirmation.** Replace the LLM's free choice with a scored matcher: the product-type noun plus a synonym table ("printer paper" → Standard copy paper, "table napkins" → Paper napkins). Ask the customer to confirm near matches ("Did you mean Matte paper?") instead of rejecting them. In this run, items rejected in 6 requests could at least have been quoted (#3, 6, 8, 11, 16, 17).
2. **Deadline-aware partial fulfilment and earlier reordering.**
   - When a deadline can't be met, offer what is in stock now and back-order the rest.
   - Set reorder points from sales velocity and supplier lead time, so popular items (A4 paper, Cardstock, Standard copy paper) are restocked before large orders arrive. Timing caused the deferrals in 8 requests.
3. **Make the ordering agent more reliable and isolate test runs.**
   - Give the agent a single-line `fulfill_order_line` tool with a structured return, and track the `agent_guards` rate (5 of 20 here; the target is 0).
   - Run each evaluation against its own temporary database file, so concurrent runs can't corrupt each other.

## 8. Configuration and running
```bash
pip install -r requirements.txt
```
Create `.env` with `OPENAI_API_KEY=...` and `OPENAI_BASE_URL=https://openai.vocareum.com/v1` (optionally `MODEL_ID`). Then:
```bash
python project_starter.py
```
This resets the database, runs every row of `quote_requests_sample.csv` in date order, and writes `test_results.csv` with the columns `status`, `agent_guards`, `sales_booked`, `stock_purchases_booked`, `net_cash_effect`, `cash_balance`, `inventory_value` and `response`.

Fixes to the starter: `init_database()` was called without its engine argument, and `response` was undefined in the test loop.
