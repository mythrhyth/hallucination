"""
Sentiment Injection Pipeline

- Wraps the deterministic Sentiment Flip model.
- Provides an `inject()` interface compatible with api_server.py.
- Flips sentiment in reasoning steps and returns modified JSON.
"""

import random
import json
import logging
from injectionModel.sentiment_flip_ import SentimentFlipModel

logging.basicConfig(level=logging.INFO)

class SentimentInjectionPipeline:
    """Pipeline wrapper for SentimentFlipModel."""

    def __init__(self):
        logging.info("Initializing SentimentInjectionPipeline...")
        self.model = SentimentFlipModel()

    def run(self, example_json: dict, num_steps_to_flip: int = 1) -> dict:
        """Run the sentiment flip operation."""
        example = example_json.copy()
        steps = example.get("reasoning_steps", [])
        if not steps:
            logging.warning("No reasoning steps found in input JSON.")
            return example

        step_labels = [0] * len(steps)
        flip_indices = random.sample(range(len(steps)), k=min(num_steps_to_flip, len(steps)))

        new_steps = []
        for i, step in enumerate(steps):
            if i in flip_indices:
                flipped = self.model.flip_step(step)
                new_steps.append(flipped)
                step_labels[i] = 1
            else:
                new_steps.append(step)

        modified = example.copy()
        modified["reasoning_steps"] = new_steps
        modified["step_labels"] = step_labels
        modified["final_answer_correct"] = 0 if any(step_labels) else example.get("final_answer_correct", 1)
        return modified

    # 👇 Add this to make it API-compatible
    def inject(self, data: dict, num_steps_to_alter: int = 1, pure: bool = False, **kwargs):
        """
        API-compatible wrapper method.
        """
        result = self.run(data, num_steps_to_flip=num_steps_to_alter)

        if pure:
            # Remove metadata fields for minimal JSON
            result.pop("context", None)
            result.pop("question", None)
            result.pop("final_answer", None)

        return result


# --- Standalone test ---
if __name__ == "__main__":
    sample = {
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

    pipeline = SentimentInjectionPipeline()
    output = pipeline.inject(sample)
    print(json.dumps(output, indent=2))
