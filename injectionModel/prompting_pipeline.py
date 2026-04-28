# injection_prompt_pipeline.py
import os
import json
import re
import random
import logging
from openai import OpenAI


API_KEY = "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"
MODEL_NAME = "deepseek/deepseek-chat-v3.1:free"

logging.basicConfig(level=logging.INFO)



def extract_json_from_text(text: str):
    """Find the first {...} JSON substring and safely parse it."""
    if not text:
        return None

    # Try direct load
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try regex extraction
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try a more lenient approach
    candidate2 = text.replace("\n", " ").replace("'", '"')
    start = candidate2.find("{")
    end = candidate2.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(candidate2[start:end+1])
        except Exception:
            pass
    return None


def fallback_inject(puzzle):
    """Fallback if API fails — make one wrong reasoning step."""
    out = json.loads(json.dumps(puzzle))  # deep copy
    steps = out.get("reasoning_steps", [])
    if not steps:
        return {"reasoning_steps": ["(fallback) no steps"], "step_labels": [1]}
    idx = 0 if len(steps) == 1 else len(steps)//2
    wrong_candidates = [
        "Reading is a physical sport.",
        "All listed items are non-physical activities.",
        "Whales, dolphins, and sharks all live on land.",
        "Therefore the final answer is Swimming."
    ]
    steps[idx] = random.choice(wrong_candidates)
    labels = [0]*len(steps)
    labels[idx] = 1
    return {"reasoning_steps": steps, "step_labels": labels}




class PromptInjectionPipeline:
    def __init__(self):
        if not API_KEY:
            raise ValueError("❌ No API key found — please set OPENROUTER_API_KEY.")
        self.client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
        logging.info("✅ Prompt Injection Pipeline initialized.")

    def _call_model_and_parse(self, puzzle, prompt, max_retries=2):
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.8,
                    extra_headers={
                        "HTTP-Referer": "https://www.kaggle.com",
                        "X-Title": "Reasoning Corruptor"
                    }
                )
            except Exception as e:
                logging.error(f"API call failed: {e}")
                return None, str(e)

            raw_text = getattr(resp.choices[0].message, "content", "")
            logging.info("RAW MODEL OUTPUT:\n" + raw_text)

            parsed = extract_json_from_text(raw_text)
            if parsed:
                return parsed, raw_text
            else:
                logging.warning(f"Parsing attempt {attempt+1} failed.")
        return None, raw_text

    def inject(self, data: dict, mode: str = "single", pure: bool = False):
        """
        Generate hallucination injection using LLM prompting.
        """
        try:
            puzzle = data

            # Create corruption prompt
            PROMPT = f"""
            You are a reasoning corrupter for dataset generation / robustness testing.

            Given the following JSON object, alter the reasoning steps so that at least one step is wrong.
            Return ONLY a JSON object (no commentary) with keys:
              - reasoning_steps: [list of steps, same length as input]
              - step_labels: [0 or 1 for each step; 1 means that step is intentionally wrong]

            Input JSON:
            {json.dumps(puzzle, indent=2)}

            Return only valid JSON.
            """

            parsed, raw = self._call_model_and_parse(puzzle, PROMPT)
            if parsed is None:
                logging.warning("Model output couldn't be parsed — using fallback.")
                parsed = fallback_inject(puzzle)
                logging.info(f"Fallback used. Raw output:\n{raw}")

            # Fix lengths if needed
            steps_out = parsed.get("reasoning_steps", [])
            labels_out = parsed.get("step_labels", [])
            orig_len = len(puzzle.get("reasoning_steps", []))

            if len(steps_out) != orig_len:
                steps_out = (steps_out + puzzle["reasoning_steps"][len(steps_out):])[:orig_len]

            if len(labels_out) != orig_len:
                labels = [0]*orig_len
                for i, (a,b) in enumerate(zip(steps_out, puzzle["reasoning_steps"])):
                    if a != b:
                        labels[i] = 1
                labels_out = labels

            # Final output
            final_output = {
                "id": puzzle.get("id", "unknown"),
                "context": puzzle.get("context", ""),
                "question": puzzle.get("question", ""),
                "reasoning_steps": steps_out,
                "step_labels": labels_out,
                "final_answer": puzzle.get("final_answer", ""),
                "final_answer_correct": puzzle.get("final_answer_correct", 1)
            }

            if pure:
                return final_output

            return {
                "input": puzzle,
                "output": final_output,
                "model_used": MODEL_NAME
            }

        except Exception as e:
            logging.error(f"Injection failed: {e}")
            return {"error": str(e)}



if __name__ == "__main__":
    pipeline = PromptInjectionPipeline()
    test_data = {
        "id": "puzzle_041",
        "context": "Identify the activity that is not physical.",
        "question": "Running, Swimming, Skating, Reading",
        "reasoning_steps": [
            "Running, Swimming, and Skating are physical activities.",
            "Reading is not a physical activity.",
            "Hence, Reading is different from the others."
        ],
        "step_labels": [0, 0, 0],
        "final_answer": "Reading",
        "final_answer_correct": 1
    }

    result = pipeline.inject(test_data)
    print(json.dumps(result, indent=2))
