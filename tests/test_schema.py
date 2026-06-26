import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import schema

def test_valid_min_doc():
    doc = {"meta": {"main_index": 4.0, "status": "ELEVATED", "generated_at": "x", "next_update": "y", "confidence": 0.5},
           "countries": {"Egypt": {"index": 4.0, "events": [], "confidence": 0.5}},
           "history": [{"timestamp": "x", "main_index": 4.0}],
           "forecast": [], "briefing": {"regional_summary_6h": {"headline": "h", "bullets": []}}}
    assert schema.validate(doc) == []

def test_catches_bad_index():
    doc = {"meta": {"main_index": 99, "status": "X", "generated_at": "", "next_update": "", "confidence": 2}}
    assert schema.validate(doc)
