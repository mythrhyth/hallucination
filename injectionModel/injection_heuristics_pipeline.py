import random
import json
import spacy
from .injection_heuristics import (
    rules,
    generate_all_injection_combinations,
)

class HeuristicInjectionPipeline:
    """
    Pipeline for heuristic hallucination injection.
    Supports 'single' and 'comprehensive' modes.
    """

    def __init__(self):
        print("Initializing Heuristic Injection Pipeline...")
        self.nlp = spacy.load("en_core_web_sm")

    def inject(self, data: dict, mode="single", pure=False):
        """
        Inject hallucinations based on heuristic rules.

        Args:
            data (dict): JSON-like input containing 'reasoning_steps'
            mode (str): 'single' → one random rule per step
                        'comprehensive' → all possible single-step combinations
            pure (bool): if True, remove metadata (_injection_info)

        Returns:
            dict | list: modified JSON or list of injected versions
        """
        reasoning_steps = list(data.get("reasoning_steps", []))
        num_steps = len(reasoning_steps)

        if num_steps == 0:
            return data

      
        if mode == "single":
            step_index = random.randint(0, num_steps - 1)
            rule_name, rule_func = random.choice(rules)

            modified_steps = reasoning_steps.copy()
            original_step = reasoning_steps[step_index]
            modified_steps[step_index] = rule_func(original_step)

            step_labels = [0] * num_steps
            step_labels[step_index] = 1

            modified_data = data.copy()
            modified_data["reasoning_steps"] = modified_steps
            modified_data["step_labels"] = step_labels
            modified_data["_injection_info"] = {
                "rule": rule_name,
                "step_index": step_index,
                "original": original_step,
                "modified": modified_steps[step_index],
            }

            if pure:
                modified_data.pop("_injection_info", None)
            return modified_data

        
        elif mode == "comprehensive":
            variants = []
            for step_index in range(num_steps):
                for rule_name, rule_func in rules:
                    modified_steps = reasoning_steps.copy()
                    original_step = reasoning_steps[step_index]
                    modified_steps[step_index] = rule_func(original_step)

                    step_labels = [0] * num_steps
                    step_labels[step_index] = 1

                    modified_data = data.copy()
                    modified_data["reasoning_steps"] = modified_steps
                    modified_data["step_labels"] = step_labels
                    modified_data["_injection_info"] = {
                        "rule": rule_name,
                        "step_index": step_index,
                        "original": original_step,
                        "modified": modified_steps[step_index],
                    }

                    if pure:
                        modified_data.pop("_injection_info", None)

                    variants.append(modified_data)
            return variants

        else:
            raise ValueError("Invalid mode. Use 'single' or 'comprehensive'.")
