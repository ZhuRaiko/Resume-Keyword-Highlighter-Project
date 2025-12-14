################################################################################
# Module: sentiment.py
#
# What this module does:
#   - Provides a small, secondary sentiment heuristic based on TextBlob.
#   - Supplies utilities to compute sentiment polarity and boolean helpers
#     for negative/positive thresholds used by downstream heuristics.
#
# Why this module is necessary in the overall system:
#   - Sentiment signals are used as secondary checks to adjust highlighting
#     or scoring confidence (for example, to de-emphasize soft-skill matches
#     that appear in negatively-worded contexts).
#
# How this module connects to other parts of the NLP / ML pipeline:
#   - Called by `highlight.py` and `scoring.py` when contextual sentiment can
#     help decide whether to accept or reject ambiguous matches. It is not a
#     primary classifier but a lightweight auxiliary heuristic.
#
"""
Sentiment Analysis - Secondary Heuristic

This module provides sentiment analysis to support the core classification.
This is a secondary heuristic, NOT a core thesis component.

Purpose: Detect positive/negative sentiment to adjust highlighting confidence
"""

from textblob import TextBlob


# Function: get_sentiment_polarity
# What this function does:
#   - Computes a sentiment polarity score for the input text using TextBlob.
# Why this function exists:
#   - Exposes a single, numeric polarity value callers can use to decide if a
#     sentence/context is positive, negative, or neutral relative to thresholds.
# Inputs expected:
#   - text (str): Text string (sentence or short paragraph) to analyze.
# Returns / side effects:
#   - Returns a float polarity in [-1.0, 1.0]. On errors, returns 0.0.
# How it contributes to the larger NLP / ML system:
#   - Provides a numeric feature used by heuristics in highlighting and
#     scoring to reduce false positive highlights under negative contexts.
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


# Function: is_negative_sentiment
# What this function does:
#   - Returns whether the input text's polarity is below a negative threshold.
# Why this function exists:
#   - Provides a simple boolean check for callers who only need a negative
#     / not-negative decision rather than the raw polarity value.
# Inputs expected:
#   - text (str): Input text to analyze.
#   - threshold (float): Numeric threshold; default -0.1.
# Returns / side effects:
#   - Returns True if polarity < threshold, otherwise False.
# How it contributes to the larger NLP / ML system:
#   - Used by highlight and scoring heuristics to suppress matches appearing
#     in negatively-worded sentences.
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


# Function: is_positive_sentiment
# What this function does:
#   - Returns whether the input text's polarity is above a positive threshold.
# Why this function exists:
#   - Provides a simple boolean check to detect positively-worded contexts.
# Inputs expected:
#   - text (str): Input text to analyze.
#   - threshold (float): Numeric threshold; default 0.15.
# Returns / side effects:
#   - Returns True if polarity > threshold, otherwise False.
# How it contributes to the larger NLP / ML system:
#   - Used where positive context should boost confidence in soft-skill or
#     recruiter-keyword matches.
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
