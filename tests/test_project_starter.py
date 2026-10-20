import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, text

import project_starter as app


class StubModel:
    def __init__(self, lines):
        self.lines = lines

    def generate(self, messages):
        class Reply:
            content = json.dumps({"items": self.lines})

        return Reply()


class ProjectStarterTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.original_engine = app.db_engine
        self.engine = create_engine(
            f"sqlite:///{Path(self.tempdir.name) / 'test.db'}"
        )
        app.db_engine = self.engine
        app.init_database(self.engine)

        app.pd.DataFrame(
            [{
                "item_name": "Cardstock",
                "category": "paper",
                "unit_price": 0.15,
                "current_stock": 300,
                "min_stock_level": 100,
            }]
        ).to_sql("inventory", self.engine, if_exists="replace", index=False)
        with self.engine.begin() as connection:
            connection.execute(text("DELETE FROM transactions"))
        app.create_transaction(None, "sales", 0, 50000, "2025-01-01T00:00:00")
        app.create_transaction("Cardstock", "stock_orders", 300, 45, "2025-01-01T00:00:00")

    def tearDown(self):
        app.db_engine = self.original_engine
        self.engine.dispose()
        self.tempdir.cleanup()

    def _transactions(self, item=None, transaction_type=None):
        query = "SELECT * FROM transactions WHERE 1=1"
        params = {}
        if item is not None:
            query += " AND item_name = :item"
            params["item"] = item
        if transaction_type is not None:
            query += " AND transaction_type = :transaction_type"
            params["transaction_type"] = transaction_type
        return app.pd.read_sql(query, self.engine, params=params)

    def _run_request(self, text, lines):
        with patch.object(app, "model", StubModel(lines)):
            return app.call_your_multi_agent_system(text)

    def test_discount_tiers_at_boundaries(self):
        expected = {199: 0, 200: 0.04, 499: 0.04, 500: 0.07, 999: 0.07, 1000: 0.10}
        for quantity, discount in expected.items():
            with self.subTest(quantity=quantity):
                self.assertEqual(app._discount(quantity), discount)

    def test_catalog_requires_exact_name_and_rejects_unrelated_mapping(self):
        self.assertEqual(app._resolve_item(" cardstock "), "Cardstock")
        self.assertIsNone(app._resolve_item("A5 paper"))
        self.assertFalse(app._wording_supports("printer paper", "Poster paper"))
        self.assertTrue(app._wording_supports("A4 glossy paper", "Glossy paper"))

    def test_order_quote_intent_and_deadline_parsing(self):
        self.assertEqual(app._detect_intent("Could you send a quote?"), "quote")
        self.assertEqual(app._detect_intent("Please place an order and include a quote"), "order")
        self.assertEqual(
            app._parse_deadline(
                "Please deliver by April 15th, 2025. (Date of request: 2025-04-01)",
                "2025-04-01",
            ),
            "2025-04-15",
        )
        self.assertIsNone(app._parse_deadline("by February 31, 2025", "2025-02-01"))
        self.assertIn("LATE: 2 days", app._timing("2025-04-17", "2025-04-15"))
        self.assertIn("meets your", app._timing("2025-04-15", "2025-04-15"))

    def test_quote_is_read_only_and_includes_deadline_estimate(self):
        before = len(self._transactions())
        response = self._run_request(
            "Please quote 500 Cardstock by April 15, 2025 (Date of request: 2025-04-08)",
            [{"requested": "Cardstock", "catalog_name": "Cardstock", "quantity": 500}],
        )
        self.assertIn("Here is your quote", response)
        self.assertIn("estimated delivery by 2025-04-12", response)
        self.assertIn("meets your 2025-04-15 deadline", response)
        self.assertEqual(len(self._transactions()), before)

    def test_order_records_sale_and_flags_late_delivery(self):
        response = self._run_request(
            "Please order 50 Cardstock by April 7, 2025 (Date of request: 2025-04-08)",
            [{"requested": "Cardstock", "catalog_name": "Cardstock", "quantity": 50}],
        )
        self.assertIn("Your order has been placed", response)
        self.assertIn("LATE: 1 day", response)
        sales = self._transactions("Cardstock", "sales")
        self.assertEqual(int(sales.iloc[-1]["units"]), 50)
        self.assertEqual(sales.iloc[-1]["transaction_date"], "2025-04-08")

    def test_unknown_product_is_not_substituted(self):
        before = len(self._transactions(transaction_type="sales"))
        response = self._run_request(
            "Please order printer paper (Date of request: 2025-04-08)",
            [{"requested": "printer paper", "catalog_name": "Poster paper", "quantity": 100}],
        )
        self.assertIn("printer paper", response)
        self.assertIn("no substitutions made", response)
        self.assertEqual(len(self._transactions(transaction_type="sales")), before)

    def test_shortage_restock_sets_actual_sale_date(self):
        result = app._fulfill_line("Cardstock", 350, "2025-04-08")
        self.assertTrue(result["sold"])
        self.assertEqual(result["sale_date"], "2025-04-12")
        orders = self._transactions("Cardstock", "stock_orders")
        self.assertEqual(int(orders.iloc[-1]["units"]), 200)
        self.assertEqual(orders.iloc[-1]["transaction_date"], "2025-04-12")

    def test_pending_delivery_is_used_without_duplicate_restock(self):
        app.create_transaction("Cardstock", "stock_orders", 100, 0, "2025-04-10")
        before = len(self._transactions("Cardstock", "stock_orders"))
        result = app._fulfill_line("Cardstock", 350, "2025-04-08")
        self.assertTrue(result["sold"])
        self.assertEqual(result["sale_date"], "2025-04-10")
        self.assertEqual(len(self._transactions("Cardstock", "stock_orders")), before)

    def test_restock_is_refused_when_cash_reserve_would_be_breached(self):
        app.create_transaction("Cardstock", "stock_orders", 0, 49000, "2025-04-08")
        before = len(self._transactions("Cardstock", "sales"))
        result = app._fulfill_line("Cardstock", 350, "2025-04-08")
        self.assertFalse(result["sold"])
        self.assertIn("below reserve", result["notes"][0])
        self.assertEqual(len(self._transactions("Cardstock", "sales")), before)


if __name__ == "__main__":
    unittest.main()
