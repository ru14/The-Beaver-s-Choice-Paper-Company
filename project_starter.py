import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
import json
import difflib
import re
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

dotenv.load_dotenv()

# Any OpenAI-compatible endpoint works (e.g. the Udacity/Vocareum proxy via OPENAI_BASE_URL).
model = OpenAIServerModel(
    model_id=os.getenv("MODEL_ID", "gpt-4o-mini"),
    api_base=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "missing-key"),
)

CATALOG = {p["item_name"]: p for p in paper_supplies}
SUPPLIER_COST_RATIO = 0.7   # supplier charges 70% of list price
RESTOCK_BUFFER = 1.5        # when restocking, order up to min_stock_level * this on top of the need
MIN_CASH_RESERVE = 1000.0   # never let a restock drop cash below this


def _resolve_item(name: str) -> Union[str, None]:
    """Resolve a catalog item by exact name, ignoring case and surrounding whitespace."""
    normalized = name.strip().casefold()
    return next((item for item in CATALOG if item.casefold() == normalized), None)


def _suggest_items(description: str, limit: int = 3) -> List[str]:
    """Suggest catalog names without treating fuzzy matches as confirmed items."""
    matches = difflib.get_close_matches(
        description.strip().casefold(),
        [item.casefold() for item in CATALOG],
        n=limit,
        cutoff=0.3,
    )
    catalog_names = {item.casefold(): item for item in CATALOG}
    return [catalog_names[match] for match in matches]


def _min_stock(item: str) -> int:
    row = pd.read_sql("SELECT min_stock_level FROM inventory WHERE item_name = :n", db_engine, params={"n": item})
    return int(row.iloc[0, 0]) if not row.empty else 100


def _stock(item: str, date: str) -> int:
    cutoff = f"{date}T23:59:59" if len(date) == 10 else date
    return int(get_stock_level(item, cutoff)["current_stock"].iloc[0])


def _pending_stock_dates(item: str, as_of_date: str) -> List[str]:
    rows = pd.read_sql(
        """
        SELECT DISTINCT date(transaction_date) AS delivery_date
        FROM transactions
        WHERE item_name = :item
          AND transaction_type = 'stock_orders'
          AND date(transaction_date) > date(:as_of_date)
        ORDER BY delivery_date
        """,
        db_engine,
        params={"item": item, "as_of_date": as_of_date},
    )
    return rows["delivery_date"].tolist()


def _unit_price(item: str) -> float:
    return float(CATALOG[item]["unit_price"])


def _discount(qty: int) -> float:
    """Volume discount tiers."""
    if qty >= 1000:
        return 0.10
    if qty >= 500:
        return 0.07
    if qty >= 200:
        return 0.04
    return 0.0


def _parse_items(items_json: str) -> List[Dict]:
    items = json.loads(items_json)
    if isinstance(items, dict):
        items = [items]
    return [{"item_name": str(i["item_name"]), "quantity": int(i["quantity"])} for i in items]


# ---------- Tools for inventory agent ----------

@tool
def find_catalog_item(description: str) -> str:
    """Find an exact catalog item name or suggest close names without confirming a match.

    Args:
        description: Customer's free-text name of the product.
    """
    item = _resolve_item(description)
    if item:
        return f"{item} (list price ${_unit_price(item):.2f}/unit)"
    return f"No exact catalog match for '{description}'. Closest candidates: {_suggest_items(description)}"


@tool
def check_stock(item_name: str, as_of_date: str) -> str:
    """Report stock level and reorder threshold of one item.

    Args:
        item_name: Official catalog item name.
        as_of_date: ISO date YYYY-MM-DD.
    """
    item = _resolve_item(item_name)
    if not item:
        return f"'{item_name}' is not in the catalog."
    stock = _stock(item, as_of_date)
    return json.dumps({"item": item, "stock": stock, "min_stock_level": _min_stock(item)})


@tool
def list_inventory(as_of_date: str) -> str:
    """List every item with positive stock on a date.

    Args:
        as_of_date: ISO date YYYY-MM-DD.
    """
    return json.dumps(get_all_inventory(as_of_date))


@tool
def restock_item(item_name: str, quantity: int, as_of_date: str) -> str:
    """Order stock from the supplier. Stock arrives (and cash is spent) on the supplier delivery date.
    Refuses if the purchase would drop cash below the reserve.

    Args:
        item_name: Official catalog item name.
        quantity: Units to order.
        as_of_date: ISO date YYYY-MM-DD of the order.
    """
    item = _resolve_item(item_name)
    if not item or quantity <= 0:
        return "Invalid item or quantity."
    cost = round(quantity * _unit_price(item) * SUPPLIER_COST_RATIO, 2)
    cash = get_cash_balance(as_of_date)
    if cash - cost < MIN_CASH_RESERVE:
        return f"Restock refused: cost ${cost:.2f} would leave cash ${cash - cost:.2f} below reserve."
    delivery = get_supplier_delivery_date(as_of_date, quantity)
    create_transaction(item, "stock_orders", quantity, cost, delivery)
    return f"Ordered {quantity} x {item} for ${cost:.2f}; arrives {delivery}."


# ---------- Tools for quoting agent ----------

@tool
def search_past_quotes(search_terms: List[str]) -> str:
    """Search historical quotes for pricing precedent.

    Args:
        search_terms: Keywords such as item names or event types.
    """
    rows = search_quote_history([t for t in search_terms if t], limit=3)
    for r in rows:  # keep context small
        r["original_request"] = r["original_request"][:200]
        r["quote_explanation"] = r["quote_explanation"][:200]
    return json.dumps(rows, default=str)


@tool
def calculate_quote(items_json: str, as_of_date: str) -> str:
    """Price an order with volume discounts and report stock availability for each line.

    Args:
        items_json: JSON list like [{"item_name": "Cardstock", "quantity": 100}].
        as_of_date: ISO date YYYY-MM-DD.
    """
    lines, total = [], 0.0
    for it in _parse_items(items_json):
        item = _resolve_item(it["item_name"])
        if not item:
            lines.append({"item": it["item_name"], "error": "not in catalog; cannot quote"})
            continue
        qty, disc = it["quantity"], _discount(it["quantity"])
        price = round(qty * _unit_price(item) * (1 - disc), 2)
        total += price
        lines.append({"item": item, "quantity": qty, "unit_price": _unit_price(item),
                      "discount": disc, "line_total": price, "in_stock": _stock(item, as_of_date)})
    return json.dumps({"lines": lines, "total": round(total, 2)})


# ---------- Tools for ordering agent ----------

@tool
def fulfill_order(items_json: str, as_of_date: str) -> str:
    """Complete a sale. For each line: restock from the supplier if stock is short (or would fall below the
    minimum level), then record the sale on the first date the goods are available.
    Lines that cannot be supplied are reported, never silently dropped.

    Args:
        items_json: JSON list like [{"item_name": "Cardstock", "quantity": 100}].
        as_of_date: ISO date YYYY-MM-DD of the request.
    """
    report = []
    for it in _parse_items(items_json):
        item = _resolve_item(it["item_name"])
        if not item:
            report.append(f"{it['item_name']}: not in catalog, not sold.")
            continue
        result = _fulfill_line(item, it["quantity"], as_of_date)
        report.extend(f"{item}: {note}" for note in result["notes"])
        if result["sold"]:
            report.append(f"{item}: sold {it['quantity']} for ${result['price']:.2f}, fulfilled by {result['sale_date']}.")
    return "\n".join(report)


def _fulfill_line(item: str, qty: int, as_of_date: str) -> Dict:
    """Restock if needed and record one sale. Returns a structured result (internal notes included)."""
    stock = _stock(item, as_of_date)
    sale_date = as_of_date
    notes = []
    shortfall = qty - stock
    top_up = max(0, _min_stock(item) - (stock - qty))
    if shortfall > 0:
        pending_dates = _pending_stock_dates(item, as_of_date)
        available_date = next((d for d in pending_dates if _stock(item, d) >= qty), None)
        if available_date:
            sale_date = available_date
            notes.append(f"using previously ordered stock arriving {available_date}.")
        else:
            order_qty = shortfall + int(_min_stock(item) * RESTOCK_BUFFER)
            msg = restock_item(item, order_qty, as_of_date)
            if msg.startswith("Restock refused") or "Invalid" in msg:
                notes.append(f"only {stock} in stock, cannot supply {qty} ({msg})")
                return {"sold": False, "in_stock": stock, "notes": notes}
            sale_date = get_supplier_delivery_date(as_of_date, order_qty)
            notes.append(msg)
    elif top_up > 0:
        notes.append(restock_item(item, int(top_up * RESTOCK_BUFFER), as_of_date))
    discount = _discount(qty)
    price = round(qty * _unit_price(item) * (1 - discount), 2)
    create_transaction(item, "sales", qty, price, sale_date)
    return {"sold": True, "price": price, "discount": discount, "sale_date": sale_date,
            "in_stock": stock, "notes": notes}


@tool
def get_finances(as_of_date: str) -> str:
    """Return cash balance and inventory value.

    Args:
        as_of_date: ISO date YYYY-MM-DD.
    """
    r = generate_financial_report(as_of_date)
    return json.dumps({k: r[k] for k in ("cash_balance", "inventory_value", "total_assets")})


# ---------- Agents ----------

inventory_agent = ToolCallingAgent(
    tools=[find_catalog_item, check_stock, list_inventory], model=model,
    name="inventory_agent",
    description="Maps customer wording to catalog items and reports stock levels. Do not place orders; the ordering agent handles fulfillment.",
)
quoting_agent = ToolCallingAgent(
    tools=[find_catalog_item, search_past_quotes, calculate_quote], model=model,
    name="quoting_agent",
    description="Builds priced quotes using volume discounts and historical quotes. Give it items, quantities and the date.",
)
ordering_agent = ToolCallingAgent(
    tools=[fulfill_order, get_finances], model=model,
    name="ordering_agent",
    description="Finalizes a sale: restocks if needed, records transactions, returns delivery timing. Give it items, quantities and the date.",
)
orchestrator = ToolCallingAgent(
    tools=[], managed_agents=[inventory_agent, quoting_agent, ordering_agent], model=model,
    name="orchestrator", max_steps=12,
    instructions=(
        "You handle customer paper requests for the Beaver's Choice Paper Company. Workflow: "
        "1) extract items/quantities (use the catalog names) and the request date; "
        "2) ask inventory_agent for the catalog names and stock; "
        "3) ask quoting_agent for a quote; "
        "4) if the customer is ordering, ask ordering_agent to fulfill it; "
        "5) reply with a short customer-friendly message: items, prices, discounts, total, delivery date, "
        "and clearly state anything that could not be supplied. Never reveal internal costs, margins or "
        "database details. Always pass dates as YYYY-MM-DD."
    ),
)

# ---------- Enforced request pipeline ----------
# The LLM only does intake (free text -> structured lines). Everything after that is code:
# exact catalog validation, order-vs-quote routing, tool calls through the specialist agents'
# tools, and the customer reply. Unknown items are reported, never substituted.

QUOTE_ONLY_RE = re.compile(r"\b(quote|quotation|estimate|how much|price list|pricing)\b", re.I)
ORDER_RE = re.compile(r"\b(order|purchase|buy|place|confirm)\b", re.I)
DATE_RE = re.compile(r"Date of request:\s*(\d{4}-\d{2}-\d{2})")


def _detect_intent(request: str) -> str:
    """'quote' only when the customer asks for pricing without ordering; otherwise 'order'."""
    return "quote" if QUOTE_ONLY_RE.search(request) and not ORDER_RE.search(request) else "order"


def _extract_lines(request: str) -> List[Dict]:
    """Ask the model to turn the request into [{requested, catalog_name|None, quantity}]."""
    prompt = (
        "Extract every product line from this customer request.\n"
        "Return ONLY JSON: {\"items\": [{\"requested\": \"<customer wording>\", "
        "\"catalog_name\": \"<exact name from the catalog list, or null>\", \"quantity\": <integer>}]}\n"
        "Rules: copy catalog_name exactly from the list and only if it is the same kind of product the "
        "customer asked for; otherwise use null. Never substitute a similar product. "
        "Quantity is the number the customer stated, exactly as written (e.g. '500 sheets' -> 500); "
        "never convert between units. Ignore dates and greetings.\n\n"
        f"Catalog: {json.dumps(list(CATALOG))}\n\nRequest:\n{request}"
    )
    reply = model.generate([{"role": "user", "content": [{"type": "text", "text": prompt}]}])
    raw = reply.content if isinstance(reply.content, str) else str(reply.content)
    data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    lines = []
    for it in data["items"]:
        qty = int(str(it["quantity"]).replace(",", ""))
        lines.append({"requested": str(it["requested"]), "catalog_name": it.get("catalog_name"), "quantity": qty})
    return lines


GENERIC_WORDS = {"paper", "papers", "sheet", "sheets", "of", "the", "a", "and", "with", "size", "in", "per"}


def _words(text: str) -> set:
    return {w[:-1] if w.endswith("s") and len(w) > 3 else w for w in re.findall(r"[a-z0-9]+", text.lower())} - GENERIC_WORDS


def _wording_supports(requested: str, item: str) -> bool:
    """The customer's wording must share a distinctive word with the catalog name (guards against
    the model mapping e.g. 'printer paper' onto an unrelated catalog item)."""
    return bool(_words(requested) & _words(item))


MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"], start=1)}
DEADLINE_RE = re.compile(
    r"\b(" + "|".join(MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", re.I)


def _parse_deadline(request: str, as_of_date: str) -> Union[str, None]:
    """Customer's requested delivery date as YYYY-MM-DD (the latest date mentioned), or None.
    A missing year defaults to the request year."""
    body = DATE_RE.sub("", request)
    default_year = int(as_of_date[:4])
    found = []
    for month, day, year in DEADLINE_RE.findall(body):
        try:
            found.append(datetime(int(year or default_year), MONTHS[month.lower()], int(day)))
        except ValueError:
            continue
    return max(found).strftime("%Y-%m-%d") if found else None


def _estimate_ready_date(item: str, qty: int, as_of_date: str) -> str:
    """Read-only: the earliest date the goods could be delivered (no restock is placed)."""
    if _stock(item, as_of_date) >= qty:
        return as_of_date
    pending = next((d for d in _pending_stock_dates(item, as_of_date) if _stock(item, d) >= qty), None)
    if pending:
        return pending
    shortfall = qty - _stock(item, as_of_date)
    return get_supplier_delivery_date(as_of_date, shortfall + int(_min_stock(item) * RESTOCK_BUFFER))


def _timing(delivery: str, deadline: Union[str, None]) -> str:
    if not deadline:
        return f"delivery by {delivery}"
    if delivery <= deadline:
        return f"delivery by {delivery} (meets your {deadline} deadline)"
    days = (datetime.fromisoformat(delivery) - datetime.fromisoformat(deadline)).days
    return f"delivery by {delivery} (LATE: {days} day{'s' if days != 1 else ''} after your {deadline} deadline)"


def _handle_request(request: str) -> str:
    match = DATE_RE.search(request)
    as_of_date = match.group(1) if match else datetime.now().strftime("%Y-%m-%d")
    intent = _detect_intent(request)
    deadline = _parse_deadline(request, as_of_date)
    lines = _extract_lines(request)

    valid, unavailable = [], []
    for ln in lines:
        item = _resolve_item(ln["catalog_name"]) if ln["catalog_name"] else None
        if item and ln["quantity"] > 0 and _wording_supports(ln["requested"], item):
            valid.append({"item_name": item, "quantity": ln["quantity"]})
        else:
            unavailable.append(ln["requested"])

    out, total, late = [], 0.0, []
    if valid:
        # inventory agent's tools: confirm stock; quoting agent's tool: price; ordering agent's tool: sell
        quote = json.loads(quoting_agent.tools["calculate_quote"](items_json=json.dumps(valid), as_of_date=as_of_date))
        for q in quote["lines"]:
            disc = f" ({int(q['discount'] * 100)}% volume discount)" if q["discount"] else ""
            if intent == "quote":
                ready = _estimate_ready_date(q["item"], q["quantity"], as_of_date)
                out.append(f"- {q['quantity']} x {q['item']}: ${q['line_total']:.2f}{disc}, "
                           f"estimated {_timing(ready, deadline)}")
                total += q["line_total"]
                if deadline and ready > deadline:
                    late.append(q["item"])
                continue
            result = _fulfill_line(q["item"], q["quantity"], as_of_date)
            if result["sold"]:
                out.append(f"- {q['quantity']} x {q['item']}: ${result['price']:.2f}{disc}, "
                           f"{_timing(result['sale_date'], deadline)}")
                total += result["price"]
                if deadline and result["sale_date"] > deadline:
                    late.append(q["item"])
            else:
                unavailable.append(f"{q['quantity']} x {q['item']} (insufficient stock and could not be restocked)")

    header = "Here is your quote" if intent == "quote" else "Your order has been placed"
    parts = [f"{header} (request date {as_of_date}):"] if out else ["We could not process any item in your request."]
    parts += out
    if out:
        parts.append(f"Total: ${total:.2f}")
    if late:
        parts.append("Note: we cannot deliver the following by your requested date: " + ", ".join(late)
                     + ". Please contact us if the later delivery date does not work for you.")
    if unavailable:
        parts.append("We could not supply the following (not carried or unavailable; no substitutions made): "
                     + "; ".join(unavailable))
    return "\n".join(parts)


def call_your_multi_agent_system(request: str) -> str:
    try:
        return _handle_request(request)
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"WARN: intake failed ({e}); falling back to the orchestrator agent")
        return str(orchestrator.run(request))


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():

    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        request_with_date = f"{row['request']} (Date of request: {request_date})"
        try:
            response = call_your_multi_agent_system(request_with_date)
        except Exception as e:
            response = f"ERROR: {e}"

        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
