"""
Sentiment Analysis - Secondary Heuristic

This module provides sentiment analysis to support the core classification.
This is a secondary heuristic, NOT a core thesis component.

Purpose: Detect positive/negative sentiment to adjust highlighting confidence
"""

from textblob import TextBlob


def get_sentiment_polarity(text):
    """
    Calculate sentiment polarity of text.
    
    Args:
        text (str): Input text
    
    Returns:
        float: Polarity score [-1.0, 1.0] where:
            - Negative values indicate negative sentiment
            - Positive values indicate positive sentiment
            - 0.0 is neutral
    """
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0


def is_negative_sentiment(text, threshold=-0.1):
    """
    Check if text has negative sentiment below threshold.
    
    Args:
        text (str): Input text
        threshold (float): Sentiment threshold (default: -0.1)
    
    Returns:
        bool: True if sentiment is negative, False otherwise
    """
    return get_sentiment_polarity(text) < threshold


def is_positive_sentiment(text, threshold=0.15):
    """
    Check if text has positive sentiment above threshold.
    
    Args:
        text (str): Input text
        threshold (float): Sentiment threshold (default: 0.15)
    
    Returns:
        bool: True if sentiment is positive, False otherwise
    """
    return get_sentiment_polarity(text) > threshold
