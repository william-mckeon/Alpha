"""Conservative, auditable identity normalization; code and quotes are untouched."""
import re

SELF = re.compile(r"\b(?:I am|I'm|My name is) (?:Claude(?: Code)?|ChatGPT)\b")


def normalize(text):
    changes = []
    result = []
    fenced = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(('```', '~~~')):
            fenced = not fenced
            result.append(line)
            continue
        if fenced or line.lstrip().startswith(('>', '"', "'")):
            result.append(line)
            continue
        # Inline code and quoted text must not be rewritten.
        parts = re.split(r'(`[^`]*`|"[^"\n]*"|\'[^\'\n]*\')', line)
        for i in range(0, len(parts), 2):
            def replace(match):
                changes.append({'before': match.group(), 'after': 'I am Arcus'})
                return 'I am Arcus'
            parts[i] = SELF.sub(replace, parts[i])
        result.append(''.join(parts))
    return ''.join(result), changes
