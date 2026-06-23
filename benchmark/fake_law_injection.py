import re
import random

def inject_fake_law(question: str) -> str:
    """
    Swaps real section or article references in a question for fake ones.
    E.g., 'Section 10' -> 'Section 999Z'
    """
    fake_sections = ["999Z", "888X", "1000B", "404NotFound"]
    fake_articles = ["420B", "500X", "399Z", "999A"]
    
    # Replace sections
    question = re.sub(
        r"Section\s+\d+[A-Z]*(?:\(\d+\))?", 
        lambda m: f"Section {random.choice(fake_sections)}", 
        question, 
        flags=re.IGNORECASE
    )
    
    # Replace articles
    question = re.sub(
        r"Article\s+\d+[A-Z]*", 
        lambda m: f"Article {random.choice(fake_articles)}", 
        question, 
        flags=re.IGNORECASE
    )
    
    return question
