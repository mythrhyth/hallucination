"""
Deterministic Sentiment-Flip Model (Standalone JSON Output)

- Detects sentiment using HuggingFace ("sentiment-analysis").
- Replaces sentiment-bearing words with antonyms (via WordNet).
- Falls back to negation or mild polarity flip templates.
- Outputs modified reasoning steps in JSON format with updated step_labels.

Run:
    python injectionModel/injection_sentiment_flip.py
"""

import re
import random
import json
import logging
from transformers import pipeline

# Attempt to import NLTK WordNet utils
try:
    import nltk
    from nltk.corpus import wordnet as wn
except Exception:
    nltk = None
    wn = None

logging.basicConfig(level=logging.INFO)


def ensure_wordnet():
    """Ensure WordNet data is available."""
    global nltk, wn
    if nltk is None:
        import nltk as _nltk
        nltk = _nltk
    try:
        wn.synsets("good")
    except Exception:
        nltk.download("wordnet")
        nltk.download("omw-1.4")
    finally:
        from nltk.corpus import wordnet as _wn
        wn = _wn


def find_antonym(word: str) -> str:
    """Return an antonym for the given word if found."""
    if wn is None:
        return None
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            for ant in lemma.antonyms():
                return ant.name().replace("_", " ")
    return None


def simple_negate_sentence(sent: str) -> str:
    """
    Flip polarity using simple negation heuristics.
    Example: "It is good." → "It is not good."
    """
    s = sent.strip()
    m = re.search(r'\b(am|is|are|was|were)\b\s+(.*)', s, flags=re.I)
    if m:
        verb = m.group(1)
        rest = m.group(2)
        if re.match(r'not\s+', rest, flags=re.I):
            rest2 = re.sub(r'not\s+', '', rest, count=1, flags=re.I)
            return f"{s.split(verb, 1)[0].strip()} {verb} {rest2}"
        else:
            return f"{s.split(verb, 1)[0].strip()} {verb} not {rest}"
    return f"Contrary to that, {s[0].lower() + s[1:]}" if s else s


class SentimentFlipModel:
    """Core sentiment flipping model."""

    def __init__(self, use_wordnet: bool = True):
        self.analyzer = pipeline("sentiment-analysis")
        self.use_wordnet = use_wordnet
        if use_wordnet:
            try:
                ensure_wordnet()
            except Exception as e:
                logging.warning(f"WordNet unavailable: {e}")
                self.use_wordnet = False

    def flip_step(self, step: str) -> str:
        """Flip sentiment of one reasoning step."""
        try:
            label = self.analyzer(step)[0]
            sentiment = label["label"].upper()
        except Exception:
            sentiment = "NEUTRAL"

        # Try antonym replacement
        tokens = re.findall(r"\w+|\W+", step)
        changed = False
        new_tokens = []

        for tok in tokens:
            if re.match(r"^\w+$", tok):
                lower = tok.lower()
                if self.use_wordnet:
                    ant = find_antonym(lower)
                    if ant and ant != lower:
                        new_tokens.append(ant if tok.islower() else ant.capitalize())
                        changed = True
                        continue
            new_tokens.append(tok)

        if changed:
            return "".join(new_tokens)

        # Fallbacks
        if sentiment == "POSITIVE":
            return simple_negate_sentence(step)
        elif sentiment == "NEGATIVE":
            s = re.sub(r'\bnot\b', '', step, flags=re.I).strip()
            if s == step or len(s) < 3:
                return f"Overall, {step} seems beneficial rather than harmful."
            return s
        else:
            return f"In contrast, {step[0].lower() + step[1:]}" if step else step


def main():
    """Run model on one example JSON (no file I/O)."""
    example = {
        "context": "Identify the odd one out in a group of animals based on biological classification.",
        "question": "Which of the following is the odd one out? Lion, Tiger, Cheetah, Wolf",
        "reasoning_steps": [
            "Lion, Tiger, and Cheetah are felines (wild cats) belonging to the Felidae family.",
            "Wolf is a canine belonging to the Canidae family.",
            "Therefore, Wolf is different from the others in biological classification."
        ],
        "step_labels": [0, 0, 0],
        "final_answer": "Wolf",
        "final_answer_correct": 1
    }

    model = SentimentFlipModel()
    print("🚀 Running Sentiment Flip Model on example...")

    steps = example["reasoning_steps"]
    labels = [0] * len(steps)

    # Choose 1 random step by default (can be more)
    flip_indices = random.sample(range(len(steps)), k=1)

    new_steps = []
    for i, step in enumerate(steps):
        if i in flip_indices:
            flipped = model.flip_step(step)
            new_steps.append(flipped)
            labels[i] = 1
        else:
            new_steps.append(step)

    # Create modified JSON
    modified = example.copy()
    modified["reasoning_steps"] = new_steps
    modified["step_labels"] = labels
    modified["final_answer_correct"] = 0 if any(labels) else example["final_answer_correct"]

    # Print JSON output
    print("\n=== Modified JSON Output ===")
    print(json.dumps(modified, indent=2))


if __name__ == "__main__":
    main()
