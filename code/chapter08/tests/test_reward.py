"""Tests for the chapter 8 verifiable reward function (section 8.5, RFT)."""
from chapter08.scripts.verifiable_reward import extract_json, invoice_reward

EXPECTED = {"invoice_number": "INV-2026-0413",
            "amount": "1240.00", "due_date": "2026-07-01"}


def test_correct_full_reward():
    r = ('{"invoice_number": "INV-2026-0413", "amount": "1240.00", '
         '"due_date": "2026-07-01"}')
    assert invoice_reward(r, EXPECTED) == 1.0


def test_fenced_json_full_reward():
    r = ('```json\n{"invoice_number": "INV-2026-0413", "amount": "1240.00", '
         '"due_date": "2026-07-01"}\n```')
    assert invoice_reward(r, EXPECTED) == 1.0


def test_partial_reward():
    r = ('{"invoice_number": "INV-2026-0413", "amount": "1240.00", '
         '"due_date": "2026-09-30"}')
    assert abs(invoice_reward(r, EXPECTED) - 2 / 3) < 1e-9


def test_not_json_is_zero():
    assert invoice_reward("Invoice INV-2026-0413, amount 1240.00.", EXPECTED) == 0.0


def test_extract_json_none_on_garbage():
    assert extract_json("no json here") is None
