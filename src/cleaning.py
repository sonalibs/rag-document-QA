import re


def clean_text(text):
    text = re.sub(r"==+\s*(.*?)\s*==+", r"\1", text)
    text = re.sub(r"\{\\displaystyle[^}]*\}", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()