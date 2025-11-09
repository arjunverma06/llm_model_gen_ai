# STEP 1: Generate ONLY questions (one per line)
prompt_template = """
You are an assistant that generates interview-style questions strictly from the provided text.

TEXT:
------------
{text}
------------

INSTRUCTIONS:
- Output ONLY questions, one per line.
- Do NOT include numbering, answers, explanations, options, or any extra text.
- Keep each question concise and complete.
- 15–20 questions is ideal.
- English only.

OUTPUT:
"""

# STEP 2: (refine step) Keep SAME strict format: questions only, one per line
refine_template = """
We already have some questions: 
{existing_answer}

We have additional context:
------------
{text}
------------

INSTRUCTIONS:
- Refine or add questions ONLY if needed based on the new context.
- Output ONLY questions, one per line.
- Do NOT include numbering, answers, explanations, options, or any extra text.
- English only.

OUTPUT:
"""
