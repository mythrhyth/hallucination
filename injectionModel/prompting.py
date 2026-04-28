# injection_model_prompt.py
import os
import json
import re
import random
import logging
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)

def get_api_key():
    """Get API key from environment variable or Kaggle secrets"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            api_key = user_secrets.get_secret("OPENROUTER_API_KEY")
        except ImportError:
            logging.warning("Kaggle secrets not available - using environment variable")
    return api_key

def extract_json_from_text(text: str):
    """Find first {...} JSON substring and load it."""
    if not text:
        return None
    # quick direct try
    try:
        return json.loads(text)
    except Exception:
        pass
    # find first balanced {...} using regex to capture from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # as a last resort, try to fix common issues (replace single quotes -> double)
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
    """Deterministic fallback: flip one middle step to an obviously wrong sentence."""
    out = json.loads(json.dumps(puzzle))  # deep copy
    steps = out.get("reasoning_steps", [])
    if not steps:
        return {"reasoning_steps": ["(fallback) no steps"], "step_labels": [1]}
    idx = 0 if len(steps) == 1 else len(steps)//2
    original = steps[idx]
    # create a clear wrong replacement
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

def call_model_and_parse(client, puzzle, prompt, max_retries=2):
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="deepseek/deepseek-chat-v3.1:free",
                messages=[{"role":"user","content":prompt}],
                temperature=0.8,
                extra_headers={
                    "HTTP-Referer": "https://www.kaggle.com",
                    "X-Title": "Reasoning Corruptor"
                }
            )
        except Exception as e:
            logging.error("API call failed: %s", e)
            # if it's a 404 about privacy, inform user and fallback
            msg = str(e)
            if "No endpoints found matching your data policy" in msg or "privacy" in msg:
                logging.error("OpenRouter privacy setting likely prevents free model access. "
                              "Enable public data usage at https://openrouter.ai/settings/privacy or use a paid model.")
            # try fallback
            return None, str(e)

        # Try to extract model text from common fields (various SDK versions)
        raw_text = None
        try:
            raw_text = resp.choices[0].message.content
        except Exception:
            try:
                raw_text = getattr(resp, "output_text", None)
            except Exception:
                raw_text = None

        # ensure it's a str
        raw_text = (raw_text or "")
        logging.info("RAW MODEL OUTPUT (start) =====")
        print(raw_text)
        logging.info("RAW MODEL OUTPUT (end) =====")

        parsed = extract_json_from_text(raw_text)
        if parsed:
            return parsed, raw_text
        else:
            logging.warning("Parsing attempt %d failed, retrying...", attempt+1)
    return None, raw_text

def main():
    # Initialize puzzle
    puzzle = {
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

    # Create prompt
    PROMPT = f"""
    You are a reasoning corrupter used for dataset generation / robustness testing.

    Given the following JSON object, alter the reasoning steps so that at least one step is wrong.
    Return ONLY a JSON object (no commentary) with keys:
      - reasoning_steps: [list of steps, same length as input]
      - step_labels: [0 or 1 for each step; 1 means that step is intentionally wrong]

    Input JSON:
    {json.dumps(puzzle, indent=2)}

    Return only valid JSON.
    """

    # Get API key and initialize client
    api_key = get_api_key()
    if not api_key:
        logging.error("No API key found. Please set OPENROUTER_API_KEY environment variable.")
        return

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Try model
    parsed, raw = call_model_and_parse(client, puzzle, PROMPT)
    if parsed is None:
        logging.warning("Model output couldn't be parsed as JSON — using deterministic fallback.")
        parsed = fallback_inject(puzzle)
        logging.info("Used fallback injection. Raw model output (for debugging):\n%s", raw)

    # Ensure lengths and types
    steps_out = parsed.get("reasoning_steps", [])
    labels_out = parsed.get("step_labels", [])
    # If model returned different length, pad/truncate to original length
    orig_len = len(puzzle["reasoning_steps"])
    if len(steps_out) != orig_len:
        if len(steps_out) < orig_len:
            steps_out = steps_out + puzzle["reasoning_steps"][len(steps_out):]
        else:
            steps_out = steps_out[:orig_len]
    if len(labels_out) != orig_len:
        # default all 0 then mark any index that differs from original as 1
        labels = [0]*orig_len
        for i, (a,b) in enumerate(zip(steps_out, puzzle["reasoning_steps"])):
            if a != b:
                labels[i] = 1
        labels_out = labels

    # Create final output
    final_output = {
        "id": puzzle["id"],
        "context": puzzle["context"],
        "question": puzzle["question"],
        "reasoning_steps": steps_out,
        "step_labels": labels_out,
        "final_answer": puzzle["final_answer"],
        "final_answer_correct": puzzle["final_answer_correct"]
    }

    # Print results
    print("\n🔹 Altered reasoning (final):")
    for i, (s,l) in enumerate(zip(final_output["reasoning_steps"], final_output["step_labels"])):
        print(f"  Step {i+1}: {s}  [label={l}]")

    print("\n🧠 Final JSON output:")
    print(json.dumps(final_output, indent=2))
    
    return final_output

if __name__ == "__main__":
    main()