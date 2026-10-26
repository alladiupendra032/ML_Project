"""
NLP Preprocessing Module
Handles text cleaning, tokenization, stop-word removal, and normalization.
"""

import re
import string
import nltk
import pandas as pd
from typing import List

# Download required NLTK data silently
def download_nltk_data():
    resources = ['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags and web-specific markup."""
    from bs4 import BeautifulSoup
    try:
        soup = BeautifulSoup(text, "lxml")
        return soup.get_text(separator=" ")
    except Exception:
        return re.sub(r'<[^>]+>', ' ', text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_special_characters(text: str) -> str:
    """Remove special characters, keeping letters, numbers, and spaces."""
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)  # remove numbers
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize multiple spaces to a single space."""
    return re.sub(r'\s+', ' ', text).strip()


def tokenize(text: str) -> List[str]:
    """Tokenize text into words."""
    try:
        return word_tokenize(text)
    except Exception:
        return text.split()


def remove_stopwords(tokens: List[str]) -> List[str]:
    """Remove stop words from token list."""
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatize tokens."""
    return [LEMMATIZER.lemmatize(t) for t in tokens]


def preprocess_text(text: str, remove_html: bool = True) -> str:
    """
    Full preprocessing pipeline:
    1. HTML removal
    2. URL removal
    3. Lowercase
    4. Special character removal
    5. Tokenization
    6. Stop-word removal
    7. Lemmatization
    8. Normalization
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    if remove_html:
        text = remove_html_tags(text)
    text = remove_urls(text)
    text = text.lower()
    text = remove_special_characters(text)
    text = normalize_whitespace(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply preprocessing to an entire DataFrame column."""
    df = df.copy()
    df['cleaned_text'] = df[text_col].apply(preprocess_text)
    # Remove empty rows after cleaning
    df = df[df['cleaned_text'].str.strip() != ""].reset_index(drop=True)
    # Remove duplicates
    df = df.drop_duplicates(subset=['cleaned_text']).reset_index(drop=True)
    return df


def get_preprocessing_steps() -> List[dict]:
    """Return list of preprocessing steps for display."""
    return [
        {"step": "1. HTML Removal", "description": "Remove HTML tags and web markup"},
        {"step": "2. URL Removal", "description": "Remove hyperlinks"},
        {"step": "3. Lowercasing", "description": "Convert all text to lowercase"},
        {"step": "4. Special Char Removal", "description": "Remove punctuation and numbers"},
        {"step": "5. Tokenization", "description": "Split text into individual words"},
        {"step": "6. Stop-word Removal", "description": "Remove common words (the, is, at…)"},
        {"step": "7. Lemmatization", "description": "Reduce words to their base form"},
        {"step": "8. Normalization", "description": "Remove extra whitespace, clean up"},
    ]
