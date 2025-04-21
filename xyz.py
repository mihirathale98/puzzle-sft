import json

with open("problems.json", "r") as f:
    problems = json.load(f)

prompts = [problem["prompt"] for problem in problems]

print(prompts)