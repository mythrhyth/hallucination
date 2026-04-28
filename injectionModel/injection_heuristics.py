# injection_model_heuristics_comprehensive.py
import random
import json
import re
import spacy

# Load the small English model once
nlp = spacy.load("en_core_web_sm")

def negation_flip(s):
    """Flips common English auxiliary and modal verbs between positive and negative forms."""
    pairs = [
        ("is not", "is"), ("isn't", "is"),
        ("are not", "are"), ("aren't", "are"),
        ("was not", "was"), ("wasn't", "was"),
        ("were not", "were"), ("weren't", "were"),
        ("will not", "will"), ("won't", "will"),
        ("cannot", "can"), ("can't", "can"),
        ("could not", "could"), ("couldn't", "could"),
        ("should not", "should"), ("shouldn't", "should"),
        ("would not", "would"), ("wouldn't", "would"),
        ("do not", "do"), ("don't", "do"),
        ("does not", "does"), ("doesn't", "does"),
        ("did not", "did"), ("didn't", "did"),
        ("have not", "have"), ("haven't", "have"),
        ("has not", "has"), ("hasn't", "has"),
        ("had not", "had"), ("hadn't", "had"),
        ("is", "is not"), ("are", "are not"), ("was", "was not"), ("were", "were not"),
        ("will", "will not"), ("can", "cannot"), ("could", "could not"), ("should", "should not"),
        ("would", "would not"), ("do", "do not"), ("does", "does not"), ("did", "did not"),
        ("have", "have not"), ("has", "has not"), ("had", "had not"),
    ]
    
    placeholders = {}
    temp_s = s

    for i, (pos, neg) in enumerate(pairs):
        placeholder = f"__PLACEHOLDER_{i}__"
        if f" {neg} " in f" {temp_s} ":
            temp_s = temp_s.replace(neg, placeholder)
            placeholders[placeholder] = pos
        elif f" {pos} " in f" {temp_s} ":
            temp_s = temp_s.replace(pos, placeholder)
            placeholders[placeholder] = neg

    final_s = temp_s
    for placeholder, final_word in placeholders.items():
        final_s = final_s.replace(placeholder, final_word)

    return final_s

def homophone_swap(s):
    """Finds and randomly swaps common English homophones in a sentence."""
    homophone_groups = [
        ("affect", "effect"), ("ale", "ail"), ("ate", "eight"), ("buy", "by", "bye"),
        ("cell", "sell"), ("cent", "scent", "sent"), ("coarse", "course"),
        ("complement", "compliment"), ("dear", "deer"), ("flour", "flower"),
        ("for", "four"), ("hear", "here"), ("hour", "our"), ("its", "it's"),
        ("knew", "new"), ("knight", "night"), ("know", "no"), ("lead", "led"),
        ("meat", "meet"), ("one", "won"), ("pair", "pear"), ("peace", "piece"),
        ("principal", "principle"), ("rain", "reign", "rein"), ("right", "write", "rite"),
        ("sea", "see"), ("son", "sun"), ("steal", "steel"), ("tail", "tale"),
        ("than", "then"), ("their", "there", "they're"), ("to", "too", "two"),
        ("wait", "weight"), ("weak", "week"), ("wear", "where"), ("weather", "whether"),
        ("which", "witch"), ("wood", "would"), ("your", "you're"),
    ]

    word_to_group = {word: group for group in homophone_groups for word in group}
    words = s.split()
    swapped_words = []

    for word in words:
        clean_word = re.sub(r'[^\w\s]', '', word).lower()
        if clean_word in word_to_group:
            group = word_to_group[clean_word]
            choices = [h for h in group if h != clean_word]
            if choices:
                swap_with = random.choice(choices)
                if word[0].isupper():
                    swap_with = swap_with.capitalize()
                punctuation = word[len(clean_word):]
                swapped_words.append(swap_with + punctuation)
            else:
                swapped_words.append(word)
        else:
            swapped_words.append(word)

    return " ".join(swapped_words)

def inject_ambiguous_modifier(s):
    """Injects a prepositional phrase into a sentence to create scope ambiguity."""
    modifiers = [
        "with a telescope", "on the hill", "in the park", "with a friend",
        "from a distance", "in a red car", "on the phone", "with a blue hat",
        "in the kitchen", "behind the building", "using binoculars",
    ]

    doc = nlp(s)
    root_verb = None
    dobj = None
    
    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token
        if token.dep_ == "dobj":
            dobj = token

    if dobj:
        first_part = doc[:dobj.i + 1].text
        last_part = doc[dobj.i + 1:].text
        return f"{first_part} {random.choice(modifiers)}{last_part}"

    return s

def add_contradiction_flip(s):
    """Adds a phrase that contradicts the main action."""
    if s.endswith("."):
        s = s[:-1]

    doc = nlp(s)
    root_verb = None
    subject = None

    for token in doc:
        if token.dep_ == "ROOT" and token.pos_ == "VERB":
            root_verb = token
            break

    if root_verb:
        for child in root_verb.children:
            if "subj" in child.dep_:
                subject = child
                break

    if root_verb and subject:
        verb_lemma = root_verb.lemma_
        contradiction = f", but {subject.text} did not {verb_lemma} at all."
        return s + contradiction
    else:
        return s + ", but the opposite happened."

def thematic_non_sequitur(s):
    """Finds a noun and connects it to an absurd, unrelated predicate."""
    if s.endswith("."):
        s = s[:-1]

    doc = nlp(s)
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]

    predicates = [
        "is a master of origami", "secretly dreams of being a cloud",
        "can solve a Rubik's Cube in under three seconds",
        "is known for its advanced calculus skills", "invented the color blue",
    ]

    if nouns:
        chosen_noun = random.choice(nouns)
        chosen_predicate = random.choice(predicates)
        return f"{s}, which is why that {chosen_noun} {chosen_predicate}."
    else:
        return s

def contradictory_quantifier(s):
    """Replaces words with their contradictory quantifiers."""
    replacements = {
        "some": "all", "all": "none", "any": "no", "many": "few", "much": "little",
        "most": "least", "more": "less", "always": "never", "every": "no",
        "everyone": "no one", "everything": "nothing", "everywhere": "nowhere",
        "sometimes": "never", "often": "rarely", "plenty": "scarcity",
        "several": "one", "numerous": "few"
    }

    words = s.split()
    modified_words = []
    for word in words:
        lower_word = word.lower()
        replacement = replacements.get(lower_word)
        if replacement:
            if word[0].isupper():
                modified_words.append(replacement.capitalize())
            else:
                modified_words.append(replacement)
        else:
            modified_words.append(word)

    return " ".join(modified_words)

# Available rules
rules = [
    ("contradiction", add_contradiction_flip),
    ("negation", negation_flip),
    ("ambiguity", inject_ambiguous_modifier),
    ("homophone", homophone_swap),
    ("non-sequitur", thematic_non_sequitur),
    ("quantifier", contradictory_quantifier)
]

def generate_all_injection_combinations(json_data):
    """
    Generate every possible single injection combination for the JSON data
    
    Args:
        json_data: Original JSON data
    
    Returns:
        List of all possible modified JSONs with single injections
    """
    all_combinations = []
    
    reasoning_steps = json_data.get("reasoning_steps", [])
    total_steps = len(reasoning_steps)
    
    if total_steps == 0:
        return all_combinations
    
    # Generate all combinations: for each step and each rule
    for step_index in range(total_steps):
        for rule_name, rule_func in rules:
            # Create a fresh copy for each combination
            modified_data = json_data.copy()
            reasoning_steps_copy = reasoning_steps.copy()
            
            # Apply the rule to the selected step
            original_step = reasoning_steps_copy[step_index]
            modified_step = rule_func(original_step)
            reasoning_steps_copy[step_index] = modified_step
            
            # Update step labels - set the hallucinated step to 1, others remain 0
            step_labels = [0] * total_steps
            step_labels[step_index] = 1
            
            # Update the JSON data
            modified_data["reasoning_steps"] = reasoning_steps_copy
            modified_data["step_labels"] = step_labels
            
            # Add metadata for tracking (optional - remove if you want pure output)
            modified_data["_injection_info"] = {
                "step_index": step_index,
                "rule_name": rule_name,
                "original_step": original_step,
                "modified_step": modified_step
            }
            
            all_combinations.append(modified_data)
    
    return all_combinations

def generate_comprehensive_dataset(json_sources, output_file="comprehensive_dataset.json"):
    """
    Generate comprehensive dataset with all possible single injections
    
    Args:
        json_sources: List of original JSON data samples
        output_file: Name of output JSON file
    
    Returns:
        List of all modified JSON samples
    """
    dataset = []
    
    for source_json in json_sources:
        combinations = generate_all_injection_combinations(source_json)
        dataset.extend(combinations)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} comprehensive samples and saved to {output_file}")
    
    # Print statistics
    rule_counts = {}
    step_counts = {}
    
    for sample in dataset:
        info = sample.get("_injection_info", {})
        rule = info.get("rule_name", "unknown")
        step = info.get("step_index", -1)
        
        rule_counts[rule] = rule_counts.get(rule, 0) + 1
        step_counts[step] = step_counts.get(step, 0) + 1
    
    print("\nRule distribution:")
    for rule, count in rule_counts.items():
        print(f"  {rule}: {count}")
    
    print("\nStep distribution:")
    for step, count in sorted(step_counts.items()):
        print(f"  Step {step}: {count}")
    
    return dataset

def demonstrate_comprehensive_injection():
    """Demonstrate comprehensive injection generation on sample JSON data"""
    
    sample_json = {
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

    print("Original JSON:")
    print(json.dumps(sample_json, indent=2))
    print("\n" + "="*60)
    
    # Generate all combinations
    all_combinations = generate_all_injection_combinations(sample_json)
    
    print(f"Generated {len(all_combinations)} unique injection combinations:")
    print(f"({len(sample_json['reasoning_steps'])} steps × {len(rules)} rules = {len(sample_json['reasoning_steps']) * len(rules)} combinations)")
    
    # Show a few examples
    print("\n" + "="*60)
    print("Sample injections:")
    
    for i, combination in enumerate(all_combinations[:6]):  # Show first 6 examples
        info = combination.get("_injection_info", {})
        print(f"\nExample {i+1}:")
        print(f"  Rule: {info.get('rule_name', 'unknown')}")
        print(f"  Step: {info.get('step_index', -1)}")
        print(f"  Original: {info.get('original_step', 'N/A')}")
        print(f"  Modified: {info.get('modified_step', 'N/A')}")
        print(f"  Step labels: {combination['step_labels']}")
    
    # Generate comprehensive dataset
    print("\n" + "="*60)
    print("Generating comprehensive dataset...")
    dataset = generate_comprehensive_dataset([sample_json], "comprehensive_dataset.json")
    
    return dataset

# If you want to generate dataset without metadata (pure JSON structure)
def generate_pure_dataset(json_sources, output_file="pure_dataset.json"):
    """
    Generate dataset without any metadata - pure JSON structure only
    """
    dataset = []
    
    for source_json in json_sources:
        combinations = generate_all_injection_combinations(source_json)
        
        # Remove metadata to keep pure structure
        for combination in combinations:
            pure_combination = combination.copy()
            if "_injection_info" in pure_combination:
                del pure_combination["_injection_info"]
            dataset.append(pure_combination)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} pure samples and saved to {output_file}")
    return dataset

if __name__ == "__main__":
    # Generate with metadata for demonstration
    dataset_with_metadata = demonstrate_comprehensive_injection()
    
    # Also generate pure dataset without metadata
    sample_json = {
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
    
    print("\n" + "="*60)
    print("Generating pure dataset (no metadata)...")
    pure_dataset = generate_pure_dataset([sample_json], "pure_dataset.json")
    
    # Show one pure sample
    if pure_dataset:
        print("\nSample pure output (no metadata):")
        print(json.dumps(pure_dataset[0], indent=2))