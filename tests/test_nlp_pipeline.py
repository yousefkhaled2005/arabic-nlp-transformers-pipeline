"""
Unit Tests for Arabic NLP Pipeline
Copyright (c) 2018-2021 Yousef Khaled.
"""

import pytest
import torch
from src.preprocessing.arabic_normalizer import ArabicTextNormalizer
from src.models.transformer_classifier import ArabicTransformerClassifier


def test_arabic_normalizer():
    raw = "الخدمة مُمتاااااازة جِداً والأسعار مُناسِبَة!!!"
    clean = ArabicTextNormalizer.normalize_orthography(raw)
    assert "ممتااز" in clean or "ممتاز" in clean
    assert "ً" not in clean # Diacritics removed
    assert "أ" not in clean # Alef normalized to ا


def test_model_forward():
    model = ArabicTransformerClassifier(vocab_size=1000, hidden_dim=128, num_classes=3)
    dummy_input = torch.randint(1, 999, (2, 32))
    mask = torch.ones(2, 32)
    logits = model(dummy_input, attention_mask=mask)
    assert logits.shape == (2, 3)
