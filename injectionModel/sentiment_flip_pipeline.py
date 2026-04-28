# injectionModel/sentiment_pipeline.py
import random
import json
from injectionModel.sentiment_flip_ import SentimentFlipModel

class SentimentInjectionPipeline:
    """
    Pipeline wrapper for the Sentiment Flip Model.
    This class provides a standardized interface for injecting hallucinations
    (sentiment-based modifications) into reasoning steps.
    """

    def __init__(self):
        self.model = SentimentFlipModel()

    def inject(self, data: dict, num_steps_to_alter: int = 1, pure: bool = False):
        """
        Perform sentiment-based injection on the given JSON data.

        Args:
            data (dict): Input JSON with reasoning steps and step_labels.
            num_steps_to_alter (int): Number of reasoning steps to modify.
            pure (bool): If True, removes metadata (only reasoning part returned).

        Returns:
            dict: Modified JSON with updated reasoning steps and step_labels.
        """
        example = data.copy()
        steps = example.get("reasoning_steps", [])
        labels = example.get("step_labels", [0] * len(steps))

        if not steps:
            raise ValueError("Input JSON missing 'reasoning_steps'.")

        # Ensure valid number of steps to alter
        k = min(num_steps_to_alter, len(steps))
        flip_indices = random.sample(range(len(steps)), k=k)

        new_steps = []
        for i, step in enumerate(steps):
            if i in flip_indices:
                flipped = self.model.flip_step(step)
                new_steps.append(flipped)
                labels[i] = 1
            else:
                new_steps.append(step)

        # Build output
        modified = example.copy()
        modified["reasoning_steps"] = new_steps
        modified["step_labels"] = labels
        modified["final_answer_correct"] = 0 if any(labels) else example.get("final_answer_correct", 1)

        # If pure mode, remove non-essential fields
        if pure:
            return {
                "context": modified.get("context", ""),
                "question": modified.get("question", ""),
                "reasoning_steps": modified["reasoning_steps"],
                "step_labels": modified["step_labels"]
            }

        return modified


# Optional CLI test
if __name__ == "__main__":
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

    pipe = SentimentInjectionPipeline()
    result = pipe.inject(example, num_steps_to_alter=1, pure=False)
    print(json.dumps(result, indent=2))
