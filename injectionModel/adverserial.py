# injection_model_adversial.py
import random
import re
import json
from typing import List, Dict, Any

class AdversarialInjector:
    """
    Adversarial minimal-edit injection for JSON puzzle structures
    Small targeted edits designed to flip answers in reasoning steps
    """
    
    def __init__(self):
        self.edit_candidates = self._initialize_edit_candidates()
    
    def _initialize_edit_candidates(self) -> Dict[str, List]:
        """Initialize candidate edits for different types of content"""
        return {
            'numbers': {
                'small_changes': [1, 2, 3, 5, 10],
                'operations': ['+', '-', '*', '/'],
                'replacements': {
                    '0': ['1', '5', '10'],
                    '1': ['2', '5', '10'],
                    '2': ['1', '3', '10'],
                    '3': ['2', '4', '5'],
                    '4': ['3', '5', '8'],
                    '5': ['4', '6', '10'],
                    '6': ['5', '7', '12'],
                    '7': ['6', '8', '14'],
                    '8': ['7', '9', '16'],
                    '9': ['8', '10', '18'],
                    '10': ['5', '15', '20'],
                    '12.5': ['6.25', '25', '18.75'],
                    '31.25': ['15.625', '62.5', '25'],
                    '78.125': ['39.0625', '156.25', '65'],
                    '195.3125': ['97.65625', '390.625', '165'],
                    '50': ['25', '75', '100'],
                    '100': ['50', '150', '200']
                }
            },
            'units': {
                'replacements': {
                    'dollars': ['cents', 'euros', 'pounds'],
                    'miles': ['kilometers', 'feet', 'meters'],
                    'hours': ['minutes', 'seconds', 'days'],
                    'cm': ['mm', 'inches', 'meters'],
                    'square cm': ['square meters', 'square inches', 'square mm']
                }
            },
            'variables': {
                'replacements': {
                    'x': ['y', 'z', 'a', 'b'],
                    'y': ['x', 'z', 'a', 'b'],
                    'z': ['x', 'y', 'a', 'b']
                }
            },
            'operations': {
                'replacements': {
                    '+': ['-', '*'],
                    '-': ['+', '/'],
                    '*': ['+', '-'],
                    '/': ['*', '-'],
                    '=': ['≠', '>', '<'],
                    '×': ['+', '-', '÷'],
                    '÷': ['×', '+', '-']
                }
            },
            'word_numbers': {
                'replacements': {
                    'one': ['two', 'three', 'five'],
                    'two': ['one', 'three', 'four'],
                    'three': ['two', 'four', 'six'],
                    'four': ['three', 'five', 'seven'],
                    'five': ['four', 'six', 'eight'],
                    'six': ['five', 'seven', 'nine'],
                    'seven': ['six', 'eight', 'ten'],
                    'eight': ['seven', 'nine', 'eleven'],
                    'nine': ['eight', 'ten', 'twelve'],
                    'ten': ['nine', 'eleven', 'fifteen'],
                    'eleven': ['ten', 'twelve', 'fourteen'],
                    'twelve': ['eleven', 'thirteen', 'fifteen'],
                    'thirteen': ['twelve', 'fourteen', 'sixteen'],
                    'fourteen': ['thirteen', 'fifteen', 'seventeen'],
                    'fifteen': ['fourteen', 'sixteen', 'eighteen'],
                    'sixteen': ['fifteen', 'seventeen', 'nineteen'],
                    'seventeen': ['sixteen', 'eighteen', 'twenty'],
                    'eighteen': ['seventeen', 'nineteen', 'twenty-one'],
                    'nineteen': ['eighteen', 'twenty', 'twenty-two'],
                    'twenty': ['nineteen', 'twenty-one', 'twenty-five'],
                    'thirty': ['twenty', 'forty', 'fifty'],
                    'forty': ['thirty', 'fifty', 'sixty'],
                    'fifty': ['forty', 'sixty', 'seventy'],
                    'sixty': ['fifty', 'seventy', 'eighty'],
                    'seventy': ['sixty', 'eighty', 'ninety'],
                    'eighty': ['seventy', 'ninety', 'hundred'],
                    'ninety': ['eighty', 'hundred', 'hundred ten'],
                    'hundred': ['ninety', 'hundred ten', 'two hundred']
                }
            }
        }
    
    def find_numbers_in_text(self, text: str) -> List[Dict]:
        """Extract numbers and their contexts from text"""
        patterns = [
            (r'\b(\d+\.?\d*)\b', 'integer'),  # integers and decimals
            (r'\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)\b', 'word_number'),  # extended word numbers
            (r'\b(half|quarter|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b', 'fraction')  # fractions
        ]
        
        numbers = []
        for pattern, num_type in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                numbers.append({
                    'value': match.group(1),
                    'start': match.start(),
                    'end': match.end(),
                    'type': num_type,
                    'context': text[max(0, match.start()-10):min(len(text), match.end()+10)]
                })
        return numbers
    
    def find_units_in_text(self, text: str) -> List[Dict]:
        """Extract units and measurements from text"""
        units_pattern = r'\b(miles|dollars|hours|cm|square cm|meters|kilometers|inches|feet|pounds|euros|cents|minutes|seconds|days|weeks|months|years)\b'
        units = []
        
        matches = re.finditer(units_pattern, text, re.IGNORECASE)
        for match in matches:
            units.append({
                'value': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'context': text[max(0, match.start()-10):min(len(text), match.end()+10)]
            })
        return units
    
    def find_variables_in_text(self, text: str) -> List[Dict]:
        """Extract mathematical variables from text"""
        variables_pattern = r'\b([xyzabc])\b'
        variables = []
        
        matches = re.finditer(variables_pattern, text, re.IGNORECASE)
        for match in matches:
            variables.append({
                'value': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'context': text[max(0, match.start()-10):min(len(text), match.end()+10)]
            })
        return variables
    
    def find_operations_in_text(self, text: str) -> List[Dict]:
        """Extract mathematical operations from text"""
        operations_pattern = r'([+\-*/=≠><×÷])'
        operations = []
        
        matches = re.finditer(operations_pattern, text)
        for match in matches:
            operations.append({
                'value': match.group(1),
                'start': match.start(),
                'end': match.end(),
                'context': text[max(0, match.start()-10):min(len(text), match.end()+10)]
            })
        return operations
    
    def generate_number_edits(self, number_info: Dict) -> List[str]:
        """Generate adversarial edits for numbers"""
        edits = []
        value = number_info['value']
        num_type = number_info['type']
        
        if num_type == 'integer' or num_type == 'fraction':
            try:
                # Handle decimal numbers
                if '.' in value:
                    num_value = float(value)
                    # Small changes for decimals
                    for change in [0.5, 1.0, 2.0, 5.0]:
                        edits.append(f"{num_value + change:.2f}".rstrip('0').rstrip('.'))
                        edits.append(f"{num_value - change:.2f}".rstrip('0').rstrip('.'))
                        if num_value > change:
                            edits.append(f"{num_value / change:.2f}".rstrip('0').rstrip('.'))
                        edits.append(f"{num_value * change:.2f}".rstrip('0').rstrip('.'))
                else:
                    num_value = float(value)
                    # Small changes for integers
                    for change in self.edit_candidates['numbers']['small_changes']:
                        edits.append(str(int(num_value + change)))
                        edits.append(str(int(num_value - change)))
                        if num_value > change:
                            edits.append(str(int(num_value / change)))
                        edits.append(str(int(num_value * change)))
                
                # Direct replacements
                if value in self.edit_candidates['numbers']['replacements']:
                    edits.extend(self.edit_candidates['numbers']['replacements'][value])
                    
            except ValueError:
                pass
                
        elif num_type == 'word_number':
            # Word number replacements
            if value.lower() in self.edit_candidates['word_numbers']['replacements']:
                edits.extend(self.edit_candidates['word_numbers']['replacements'][value.lower()])
        
        return edits
    
    def generate_unit_edits(self, unit_info: Dict) -> List[str]:
        """Generate adversarial edits for units"""
        value = unit_info['value'].lower()
        edits = []
        
        for category, replacements in self.edit_candidates['units']['replacements'].items():
            if value == category:
                edits.extend(replacements)
                break
                
        return edits
    
    def generate_variable_edits(self, variable_info: Dict) -> List[str]:
        """Generate adversarial edits for variables"""
        value = variable_info['value']
        edits = []
        
        if value in self.edit_candidates['variables']['replacements']:
            edits.extend(self.edit_candidates['variables']['replacements'][value])
            
        return edits
    
    def generate_operation_edits(self, operation_info: Dict) -> List[str]:
        """Generate adversarial edits for operations"""
        value = operation_info['value']
        edits = []
        
        if value in self.edit_candidates['operations']['replacements']:
            edits.extend(self.edit_candidates['operations']['replacements'][value])
            
        return edits
    
    def apply_edit(self, text: str, target: Dict, new_value: str) -> str:
        """Apply a single edit to the text"""
        return text[:target['start']] + new_value + text[target['end']:]
    
    def adversarial_inject(self, text: str, strategy: str = 'random', max_edits: int = 1) -> str:
        """
        Apply adversarial minimal edits to text
        
        Args:
            text: Input text to modify
            strategy: 'random', 'numbers_first', 'operations_first'
            max_edits: Maximum number of edits to apply
        
        Returns:
            Modified text with adversarial edits
        """
        if max_edits <= 0:
            return text
            
        # Find all potential edit targets
        numbers = self.find_numbers_in_text(text)
        units = self.find_units_in_text(text)
        variables = self.find_variables_in_text(text)
        operations = self.find_operations_in_text(text)
        
        all_targets = []
        edits_applied = 0
        
        # Strategy-based target ordering
        if strategy == 'numbers_first':
            all_targets = numbers + operations + variables + units
        elif strategy == 'operations_first':
            all_targets = operations + numbers + variables + units
        else:  # random
            all_targets = numbers + units + variables + operations
            random.shuffle(all_targets)
        
        modified_text = text
        
        for target in all_targets:
            if edits_applied >= max_edits:
                break
                
            edits = []
            if target in numbers:
                edits = self.generate_number_edits(target)
            elif target in units:
                edits = self.generate_unit_edits(target)
            elif target in variables:
                edits = self.generate_variable_edits(target)
            elif target in operations:
                edits = self.generate_operation_edits(target)
            
            if edits:
                # Choose one edit randomly
                chosen_edit = random.choice(edits)
                modified_text = self.apply_edit(modified_text, target, chosen_edit)
                edits_applied += 1
        
        return modified_text
    
    def check_final_answer_correctness(self, original_json: Dict, modified_json: Dict) -> bool:
        """
        Check if the final answer in modified JSON is still correct
        Simple check based on numeric values
        """
        original_answer = original_json.get("final_answer", "")
        modified_answer = modified_json.get("final_answer", "")
        
        # Extract numeric values for comparison
        original_numbers = re.findall(r'\d+\.?\d*', original_answer)
        modified_numbers = re.findall(r'\d+\.?\d*', modified_answer)
        
        # If numbers changed, answer is likely incorrect
        return original_numbers == modified_numbers
    
    def apply_to_json_reasoning(self, json_data: Dict, strategy: str = 'random', max_edits_per_step: int = 1) -> Dict:
        """
        Apply adversarial edits to reasoning steps in JSON data with step labeling
        
        Args:
            json_data: Original JSON puzzle data
            strategy: Edit strategy
            max_edits_per_step: Maximum edits per step
        
        Returns:
            Modified JSON data with same structure, only reasoning_steps and step_labels modified
        """
        modified_data = json_data.copy()
        
        reasoning_steps = modified_data.get("reasoning_steps", [])
        total_steps = len(reasoning_steps)
        
        if total_steps == 0:
            return modified_data
        
        # Randomly choose which step to inject hallucination (can be any step including last)
        step_index = random.randint(0, total_steps - 1)
        
        # Apply adversarial injection to the selected step
        original_step = reasoning_steps[step_index]
        modified_step = self.adversarial_inject(original_step, strategy, max_edits_per_step)
        reasoning_steps[step_index] = modified_step
        
        # Update step labels - set the hallucinated step to 1, others remain 0
        step_labels = [0] * total_steps
        step_labels[step_index] = 1
        
        # Update the JSON data
        modified_data["reasoning_steps"] = reasoning_steps
        modified_data["step_labels"] = step_labels
        
        # Check if final answer might be affected and update final_answer_correct
        is_correct = self.check_final_answer_correctness(json_data, modified_data)
        modified_data["final_answer_correct"] = 1 if is_correct else 0
        
        return modified_data


def generate_adversarial_dataset(json_sources: List[Dict], num_samples: int = 200, output_file: str = "adversarial_dataset.json") -> List[Dict]:
    """
    Generate a dataset with adversarial injections and save to JSON file
    
    Args:
        json_sources: List of original JSON data samples
        num_samples: Number of adversarial samples to generate
        output_file: Name of output JSON file
    
    Returns:
        List of modified JSON samples
    """
    injector = AdversarialInjector()
    dataset = []
    
    for i in range(num_samples):
        source_json = random.choice(json_sources).copy()
        
        # Apply adversarial injection
        strategy = random.choice(['random', 'numbers_first', 'operations_first'])
        max_edits = random.randint(1, 2)  # 1-2 edits per step
        
        modified_json = injector.apply_to_json_reasoning(source_json, strategy, max_edits)
        dataset.append(modified_json)
    
    # Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} adversarial samples and saved to {output_file}")
    return dataset


# Example usage and demonstration
def demonstrate_adversarial_injection():
    """Demonstrate the adversarial injection on sample JSON data"""
    
    sample_json = {
        "context": "Series: 2, 5, 12.5, ?, 78.125, 195.3125",
        "question": "Find the missing number in the series.",
        "reasoning_steps": [
            "Observe ratios between consecutive terms to identify a multiplicative pattern.",
            "5 / 2 = 2.5 and 12.5 / 5 = 2.5, so terms are multiplied by 2.5 each step.",
            "Compute the fourth term by multiplying the third term 12.5 by 2.5: 12.5 × 2.5 = 31.25.",
            "Verify next terms: 31.25 × 2.5 = 78.125 and 78.125 × 2.5 = 195.3125, which match the series."
        ],
        "step_labels": [0, 0, 0, 0],
        "final_answer": "The missing number is 31.25.",
        "final_answer_correct": 1
    }
    
    injector = AdversarialInjector()
    
    print("Original JSON:")
    print(json.dumps(sample_json, indent=2))
    print("\n" + "="*60)
    
    # Generate multiple adversarial examples
    for i in range(3):
        strategy = random.choice(['random', 'numbers_first', 'operations_first'])
        modified = injector.apply_to_json_reasoning(sample_json, strategy, max_edits_per_step=1)
        
        print(f"\nAdversarial Example #{i+1} (Strategy: {strategy}):")
        print(json.dumps(modified, indent=2))
        print("-" * 40)
    
    # Generate dataset
    print("\n" + "="*60)
    print("Generating adversarial dataset...")
    dataset = generate_adversarial_dataset([sample_json], num_samples=5, output_file="sample_adversarial.json")
    
    # Show one sample from dataset
    if dataset:
        print("\nSample from generated dataset:")
        print(json.dumps(dataset[0], indent=2))


if __name__ == "__main__":
    demonstrate_adversarial_injection()