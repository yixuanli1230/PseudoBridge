import random
from typing import Dict, List

class StyleManager:
    def __init__(self):
        # =========================================================================
        # Core Style Dimensions (Defined based on user requirements)
        # =========================================================================
        self.dimensions = {
            "Programming Paradigm": [
                "Procedural (Step-by-step logic, use functions)",
                "Object-Oriented (Encapsulate state/logic in Classes)",
                "Functional (Pure functions, use map/filter/reduce, avoid state)",
                "Declarative (Focus on what to do, not how, e.g., list comprehensions)",
                "Hybrid (Mix of OOP and Functional)"
            ],
            "Language Features": [
                "Type Declaration: Explicit (Heavy use of Type Hints/Annotations)",
                "Type Declaration: Implicit (Duck typing, no type hints)",
                "Type Enforcement: Strict (Add runtime type checks/assertions)",
                "Type Enforcement: Lenient (Flexible types)"
            ],
            "Syntactic Structures": [
                "Verbose (Descriptive, multi-step, detailed comments)",
                "Concise (Brief, efficient logic)",
                "Chainable (Method chaining / Fluent interface style)",
                "Declarative (High-level abstractions)",
                "Minimalistic (Code golf style, shortest possible)"
            ],
            "Naming Conventions": [
                "snake_case (standard_python_style)",
                "camelCase (javaStyleNaming)",
                "PascalCase (ClassNamingStyle)",
                "UPPER_SNAKE_CASE (CONSTANT_STYLE)",
                "kebab-case (lisp-style-naming, unusual for Python but distinct)"
            ],
            "Error Handling": [
                "EAFP (Easier to Ask for Forgiveness - heavy try/except usage)",
                "LBYL (Look Before You Leap - heavy if/else pre-checks)",
                "Try-Catch (Standard exception handling block)",
                "Error Codes (Return -1 or None on failure instead of raising)",
                "Logging (Log errors instead of crashing)"
            ],
            "Memory Management": [
                "Manual (Explicit cleanup, e.g., 'del', 'close()', context managers)",
                "GC (Rely on Garbage Collection, standard Python behavior)",
                "Reference (Focus on object references and mutability)",
                "Unsafe (Simulated unsafe operations, minimally applicable in Python)"
            ]
        }

    def get_diverse_styles(self, num_variants: int = 4) -> List[Dict[str, str]]:
        """
        Randomly generate `num_variants` unique style combinations.
        """
        styles = []
        # To ensure diversity, simple random selection is used.
        # If strict mutual exclusivity is needed (e.g., one OOP, one FP), add logic here.
        for _ in range(num_variants):
            profile = {
                "paradigm": random.choice(self.dimensions["Programming Paradigm"]),
                "language_features": random.choice(self.dimensions["Language Features"]),
                "syntactic_structures": random.choice(self.dimensions["Syntactic Structures"]),
                "naming": random.choice(self.dimensions["Naming Conventions"]),
                "error_handling": random.choice(self.dimensions["Error Handling"]),
                "memory": random.choice(self.dimensions["Memory Management"])
            }
            styles.append(profile)
        return styles

    def format_style_prompt(self, style_profile: Dict[str, str]) -> str:
        """
        Convert the style configuration dictionary into a Prompt string for the Generator.
        """
        return f"""
Please implement the code following these specific STYLE GUIDELINES:

1. **Programming Paradigm**: {style_profile['paradigm']}
2. **Language Features**: {style_profile['language_features']}
3. **Syntactic Structures**: {style_profile['syntactic_structures']}
4. **Naming Conventions**: Use {style_profile['naming']} for all identifiers.
5. **Error Handling**: Use the {style_profile['error_handling']} strategy.
6. **Memory Management**: {style_profile['memory']}

Ensure the code remains functional and strictly adheres to the logic in the Pseudo-code, but changes its APPEARANCE and STRUCTURE based on the above guidelines.
"""

# # Test code (can be run directly to see the output)
# if __name__ == "__main__":
#     manager = StyleManager()
#     variants = manager.get_diverse_styles(2)
#     for i, v in enumerate(variants):
#         print(f"--- Variant {i+1} ---")
#         print(manager.format_style_prompt(v))