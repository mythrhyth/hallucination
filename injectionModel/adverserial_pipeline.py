# detectionModel/injection_adversarial_pipeline.py
import random
import copy
from typing import List, Dict, Any, Union
from injectionModel.adverserial import AdversarialInjector


class AdversarialInjectionPipeline:
    """
    Pipeline wrapper for the AdversarialInjector.
    Provides a simple, model-like interface so your API/Streamlit can call:
        pipeline = AdversarialInjectionPipeline()
        out = pipeline.inject(data, mode="single", strategy="random", max_edits=1)
    """

    def __init__(self):
        print("⚔️ Initializing Adversarial Injection Pipeline...")
        self.injector = AdversarialInjector()

    def inject(
        self,
        data: Dict[str, Any],
        mode: str = "single",
        strategy: str = "random",
        max_edits: int = 1,
        num_variants: int = 5,
        pure: bool = False
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Apply adversarial injection to input JSON-like data.

        Args:
            data: dict with keys like "reasoning_steps", "final_answer", etc.
            mode: "single" -> apply one injection
                  "comprehensive" -> generate multiple variants (num_variants)
            strategy: 'random' | 'numbers_first' | 'operations_first'
            max_edits: max edits per step (int)
            num_variants: number of variants to produce in comprehensive mode
            pure: if True, remove any internal metadata keys (e.g., don't add extra fields)

        Returns:
            A single modified dict (mode='single') or list of modified dicts (mode='comprehensive')
        """
        if not isinstance(data, dict):
            raise ValueError("`data` must be a dictionary.")

        if mode not in ("single", "comprehensive"):
            raise ValueError("mode must be 'single' or 'comprehensive'")

        if mode == "single":
            # operate on a deep copy so original isn't mutated
            working = copy.deepcopy(data)
            modified = self.injector.apply_to_json_reasoning(
                json_data=working,
                strategy=strategy,
                max_edits_per_step=max_edits
            )
            if pure:
                self._strip_metadata(modified)
            return modified

        # comprehensive: produce several variants with randomized strategies / edits
        variants: List[Dict[str, Any]] = []
        for i in range(max(1, num_variants)):
            working = copy.deepcopy(data)
            chosen_strategy = random.choice(["random", "numbers_first", "operations_first", strategy])
            chosen_max_edits = max(1, min(3, int(max_edits if isinstance(max_edits, int) else 1)))
            modified = self.injector.apply_to_json_reasoning(
                json_data=working,
                strategy=chosen_strategy,
                max_edits_per_step=chosen_max_edits
            )
            if pure:
                self._strip_metadata(modified)
            variants.append(modified)

        return variants

    def generate_dataset(
        self,
        json_sources: List[Dict[str, Any]],
        num_samples: int = 200,
        max_edits_range: tuple = (1, 2),
        output_file: str = None
    ) -> List[Dict[str, Any]]:
        """
        Convenience method to generate many adversarial samples from a list of source JSONs.

        Args:
            json_sources: list of source JSON dicts
            num_samples: how many adversarial samples to generate
            max_edits_range: (min, max) edits per sample
            output_file: optional path to save JSON list to disk

        Returns:
            List of modified dicts
        """
        from random import randint, choice
        import json as _json

        dataset = []
        for _ in range(num_samples):
            source = copy.deepcopy(choice(json_sources))
            strategy = choice(["random", "numbers_first", "operations_first"])
            max_edits = randint(max_edits_range[0], max_edits_range[1])
            modified = self.injector.apply_to_json_reasoning(
                json_data=source,
                strategy=strategy,
                max_edits_per_step=max_edits
            )
            dataset.append(modified)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                _json.dump(dataset, f, indent=2, ensure_ascii=False)

        return dataset

    @staticmethod
    def _strip_metadata(d: Dict[str, Any]) -> None:
        """
        Remove any keys that are internal/metadata-like to produce a 'pure' sample.
        This mutates in-place.
        """
        # keys we tend to add that may be considered metadata
        for k in ["original_step", "modified_step", "hallucination_injected", "edit_type", "impact_score"]:
            d.pop(k, None)
        # If there are nested injection markers like 'step_labels' you may want to keep them,
        # so we only remove the obvious internal ones. Adjust as needed.
