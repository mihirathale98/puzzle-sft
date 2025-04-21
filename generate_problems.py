import itertools
from fractions import Fraction
import random
from itertools import product
import json
import time
from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    APIError,
    BadRequestError,
    RateLimitError,
    Timeout,
)
import asyncio
import os
from tqdm import tqdm
import logging
from typing import List, Dict, Any

# ------------------- PROMPT TEMPLATES -------------------

def template_sum_condition(num_dice, sides, condition, target):
    return f"What is the probability that the sum of {num_dice} {sides}-sided dice is {condition} {target}?"

def template_at_least_one_value(num_dice, sides, target_value):
    return f"What is the probability that at least one die shows a {target_value} when rolling {num_dice} {sides}-sided dice?"

def template_all_different(num_dice, sides):
    return f"What is the probability that all {num_dice} dice show different values when rolling {sides}-sided dice?"

def template_at_least_two_same(num_dice, sides):
    return f"What is the probability that at least two dice show the same number when rolling {num_dice} {sides}-sided dice?"

def template_sum_in_range(num_dice, sides, min_sum, max_sum):
    return f"What is the probability that the sum is between {min_sum} and {max_sum}, inclusive, when rolling {num_dice} {sides}-sided dice?"

# ------------------- SOLVERS -------------------

def solve_sum_condition(num_dice, sides, condition, target):
    """
    Solve the probability that the sum of dice satisfies a condition (e.g., '>= 20')
    """
    total_outcomes = sides ** num_dice
    favorable_outcomes = 0

    # For small dice counts, we can enumerate all possibilities
    if num_dice <= 8:  # Adjust based on memory constraints
        for roll in product(range(1, sides + 1), repeat=num_dice):
            total = sum(roll)
            if condition == '=' and total == target:
                favorable_outcomes += 1
            elif condition == '>' and total > target:
                favorable_outcomes += 1
            elif condition == '<' and total < target:
                favorable_outcomes += 1
            elif condition == '>=' and total >= target:
                favorable_outcomes += 1
            elif condition == '<=' and total <= target:
                favorable_outcomes += 1
    else:
        # For larger dice counts, use sampling
        samples = 100000  # Adjust based on desired accuracy
        for _ in range(samples):
            roll = [random.randint(1, sides) for _ in range(num_dice)]
            total = sum(roll)
            if condition == '=' and total == target:
                favorable_outcomes += 1
            elif condition == '>' and total > target:
                favorable_outcomes += 1
            elif condition == '<' and total < target:
                favorable_outcomes += 1
            elif condition == '>=' and total >= target:
                favorable_outcomes += 1
            elif condition == '<=' and total <= target:
                favorable_outcomes += 1
        return Fraction(favorable_outcomes, samples)

    return Fraction(favorable_outcomes, total_outcomes)

def solve_at_least_one_value(num_dice, sides, target_value):
    """
    Calculate probability that at least one die shows a specific value
    """
    # Probability of NOT getting the target value on a single die
    p_not_target = (sides - 1) / sides
    # Probability of NOT getting the target value on any of the dice
    p_none = p_not_target ** num_dice
    # Probability of getting at least one target value
    p_at_least_one = 1 - p_none
    return Fraction(int(p_at_least_one * (sides ** num_dice)), sides ** num_dice)

def solve_all_different(num_dice, sides):
    """
    Calculate probability that all dice show different values
    """
    if num_dice > sides:
        return Fraction(0, 1)  # Impossible to have all different values
    
    # Number of ways to select num_dice different values from sides values
    favorable = 1
    for i in range(num_dice):
        favorable *= (sides - i)
    
    # Total number of possible outcomes
    total = sides ** num_dice
    
    return Fraction(favorable, total)

def solve_at_least_two_same(num_dice, sides):
    """
    Calculate probability that at least two dice show the same value
    """
    return Fraction(1) - solve_all_different(num_dice, sides)

def solve_sum_in_range(num_dice, sides, min_sum, max_sum):
    """
    Calculate probability that the sum of dice is within a range
    """
    # For small dice counts, we can enumerate all possibilities
    if num_dice <= 8:  # Adjust based on memory constraints
        all_rolls = itertools.product(range(1, sides + 1), repeat=num_dice)
        total, favorable = 0, 0
        for roll in all_rolls:
            s = sum(roll)
            if min_sum <= s <= max_sum:
                favorable += 1
            total += 1
        return Fraction(favorable, total)
    else:
        # For larger dice counts, use sampling
        samples = 100000  # Adjust based on desired accuracy
        favorable = 0
        for _ in range(samples):
            roll = [random.randint(1, sides) for _ in range(num_dice)]
            s = sum(roll)
            if min_sum <= s <= max_sum:
                favorable += 1
        return Fraction(favorable, samples)

# ------------------- GENERATE PROBLEMS -------------------

def generate_problems(target_count=1000):
    """
    Generate unique probability problems using the templates randomly
    """
    # Define parameter ranges
    num_dice_range = list(range(1, 7))  # 1 to 6 dice
    sides_range = [4, 6, 8, 10, 12, 20]  # Common dice sizes
    conditions = ['=', '>', '<', '>=', '<=']
    
    # Create a pool of all possible template types
    template_types = ["sum_condition", "at_least_one_value", "all_different", "at_least_two_same", "sum_in_range"]
    
    # Function to generate a random problem based on template type
    def generate_random_problem(template_type):
        num_dice = random.choice(num_dice_range)
        sides = random.choice(sides_range)
        
        if template_type == "sum_condition":
            condition = random.choice(conditions)
            min_possible = num_dice * 1
            max_possible = num_dice * sides
            
            # Choose target that makes sense for the condition
            if condition in ['=', '>=', '<=']:
                target = random.randint(min_possible, max_possible)
            elif condition == '>':
                target = random.randint(min_possible, max_possible - 1) if min_possible < max_possible else min_possible
            elif condition == '<':
                target = random.randint(min_possible + 1, max_possible) if min_possible < max_possible else max_possible
            
            return {
                "template": "sum_condition",
                "params": {
                    "num_dice": num_dice,
                    "sides": sides,
                    "condition": condition,
                    "target": target
                },
                "prompt": template_sum_condition(num_dice, sides, condition, target),
                "correct_answer": str(solve_sum_condition(num_dice, sides, condition, target))
            }
            
        elif template_type == "at_least_one_value":
            target_value = random.randint(1, sides)
            return {
                "template": "at_least_one_value",
                "params": {
                    "num_dice": num_dice,
                    "sides": sides,
                    "target_value": target_value
                },
                "prompt": template_at_least_one_value(num_dice, sides, target_value),
                "correct_answer": str(solve_at_least_one_value(num_dice, sides, target_value))
            }
            
        elif template_type == "all_different":
            # Only valid if num_dice <= sides
            if num_dice > sides:
                num_dice = random.randint(1, sides)
            
            return {
                "template": "all_different",
                "params": {
                    "num_dice": num_dice,
                    "sides": sides
                },
                "prompt": template_all_different(num_dice, sides),
                "correct_answer": str(solve_all_different(num_dice, sides))
            }
            
        elif template_type == "at_least_two_same":
            # Need at least 2 dice
            if num_dice < 2:
                num_dice = random.randint(2, 6)
                
            return {
                "template": "at_least_two_same",
                "params": {
                    "num_dice": num_dice,
                    "sides": sides
                },
                "prompt": template_at_least_two_same(num_dice, sides),
                "correct_answer": str(solve_at_least_two_same(num_dice, sides))
            }
            
        elif template_type == "sum_in_range":
            min_possible = num_dice * 1
            max_possible = num_dice * sides
            range_size = max_possible - min_possible
            
            # Generate a reasonable range
            range_width = random.randint(2, max(3, range_size // 2))
            min_sum = random.randint(min_possible, max_possible - range_width)
            max_sum = min_sum + range_width
            
            return {
                "template": "sum_in_range",
                "params": {
                    "num_dice": num_dice,
                    "sides": sides,
                    "min_sum": min_sum,
                    "max_sum": max_sum
                },
                "prompt": template_sum_in_range(num_dice, sides, min_sum, max_sum),
                "correct_answer": str(solve_sum_in_range(num_dice, sides, min_sum, max_sum))
            }
    
    # Generate problems randomly
    problems = []
    problem_prompts = set()  # To ensure uniqueness
        
    with tqdm(total=target_count, desc="Generating problems") as pbar:
        while len(problems) < target_count:
            # Choose a random template type
            template_type = random.choice(template_types)
            
            # Generate a problem of that type
            problem = generate_random_problem(template_type)
            
            # Check if this prompt is unique
            if problem["prompt"] not in problem_prompts:
                problems.append(problem)
                problem_prompts.add(problem["prompt"])
                
                # Update the progress bar
                pbar.update(1)
    
    return problems


def is_answer_correct(model_answer, correct_answer):
    """
    Check if the model's answer matches the correct answer by comparing numerical values.
    Handles fractions, decimals, percentages, and LaTeX boxed expressions.
    """
    import re
    from fractions import Fraction
    
    # Convert correct_answer to a Fraction object
    try:
        if '/' in correct_answer:
            num, denom = correct_answer.split('/')
            correct_fraction = Fraction(int(num), int(denom))
        else:
            correct_fraction = Fraction(correct_answer)
        correct_float = float(correct_fraction)
    except:
        return False  # Can't parse the correct answer
    
    # Simple exact match check
    if correct_answer in model_answer:
        return True
    
    # Check for boxed LaTeX expressions first (common in formatted answers)
    boxed_pattern = r'\\boxed\{(?:\\d?frac\{(\d+)\}\{(\d+)\}|(\d+)/(\d+))\}'
    boxed_matches = re.findall(boxed_pattern, model_answer)
    
    for match in boxed_matches:
        # Process non-empty groups in the match
        for i in range(0, len(match), 2):
            if i+1 < len(match) and match[i] and match[i+1]:
                try:
                    model_fraction = Fraction(int(match[i]), int(match[i+1]))
                    if model_fraction == correct_fraction or abs(float(model_fraction) - correct_float) < 0.0001:
                        return True
                except:
                    continue
    
    # Standard fraction patterns
    fraction_pattern = r'(?:(?:the answer|probability|result|final answer) is|probability of|equals|=)\s*(\d+)/(\d+)|(\d+)\s*/\s*(\d+)|\\frac\{(\d+)\}\{(\d+)\}'
    matches = re.findall(fraction_pattern, model_answer)
    
    for match in matches:
        # Process non-empty groups in the match
        for i in range(0, len(match), 2):
            if i+1 < len(match) and match[i] and match[i+1]:
                try:
                    model_fraction = Fraction(int(match[i]), int(match[i+1]))
                    if model_fraction == correct_fraction or abs(float(model_fraction) - correct_float) < 0.0001:
                        return True
                except:
                    continue
    
    # Check for decimals
    for decimal_match in re.findall(r'(\d+\.\d+)', model_answer):
        try:
            if abs(float(decimal_match) - correct_float) < 0.001:
                return True
        except:
            continue
    
    # Check for percentages
    for percent_match in re.findall(r'(\d+(?:\.\d+)?)%', model_answer):
        try:
            if abs(float(percent_match) / 100 - correct_float) < 0.001:
                return True
        except:
            continue
    
    # Special cases for 1 and 0
    if (correct_float == 1.0 and any(phrase in model_answer.lower() for phrase in 
                                    ["probability is 1", "probability of 1", "= 1"])):
        return True
    
    if (correct_float == 0.0 and any(phrase in model_answer.lower() for phrase in 
                                    ["probability is 0", "probability of 0", "= 0"])):
        return True
    
    return False



# ------------------- MAIN ASYNC FUNCTION -------------------
def main_async(target_count=2000, output_file="probability_dataset_5k.json"):
    """
    Async main function to generate, solve, and validate problems in parallel using the enhanced API client
    """
    
    # Check if the output file exists and load existing data
    dataset = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            try:
                dataset = json.load(f)
                print(f"Loaded {len(dataset)} existing samples")
            except json.JSONDecodeError:
                print("Error loading existing file, starting fresh")
                dataset = []
    
    # Generate problems
    remaining_count = target_count - len(dataset)
    if remaining_count <= 0:
        print(f"Already have {len(dataset)} samples, target reached")
        return
    
    print(f"Generating {remaining_count} more samples...")
    problems = generate_problems(remaining_count)

    with open("problems_5k.json", "w") as f:
        json.dump(problems, f, indent=2)
    
    return None


if __name__ == "__main__":
    main_async()