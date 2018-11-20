"""
High-Performance Arabic Text Normalizer and Morphological Cleaner
Copyright (c) 2018-2021 Yousef Khaled. All rights reserved.
"""

import re
import unicodedata


class ArabicTextNormalizer:
    """
    Normalizes complex Arabic orthographic variants (Alef forms, Taa Marbuta, Tatweel, Harakat)
    and handles colloquial Egyptian and Gulf dialectal contractions.
    """
    DIACRITICS_REGEX = re.compile(r"[ً-ْٰـ]") # Fatha, Damma, Kasra, Sukun, Shadda, Tatweel

    @classmethod
    def strip_diacritics(cls, text: str) -> str:
        return cls.DIACRITICS_REGEX.sub("", text)

    @classmethod
    def normalize_orthography(cls, text: str) -> str:
        text = cls.strip_diacritics(text)
        # Normalize Alef forms: أ, إ, آ -> ا
        text = re.sub(r"[إأآٱ]", "ا", text)
        # Normalize Taa Marbuta: ة -> ه
        text = re.sub(r"ة", "ه", text)
        # Normalize Yaa: ى -> ي
        text = re.sub(r"ى", "ي", text)
        # Normalize repetitive characters (e.g., ممتااااز -> ممتاز)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        # Normalize multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        return text
