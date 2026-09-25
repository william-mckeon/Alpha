"""Shared role-safe serialization; content cannot inject structural role markers."""
import json


def header(role):
    if role not in ('system', 'user', 'assistant', 'tool'):
        raise ValueError('Unknown conversation role')
    return '\n<' + role + '>\n'


def content(item):
    text = item['content']
    if not isinstance(text, str):
        raise ValueError('Conversation content must be text')
    # Escaping only role delimiters preserves JSON tool-call syntax and ordinary text.
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def message(item):
    return header(item['role']) + content(item) + '\n</' + item['role'] + '>\n'


def pack(messages, tokenizer, budget):
    """Keep system and initial task plus the newest complete assistant/tool groups."""
    if not messages:
        raise ValueError('Empty conversation')
    prefix = []
    rest = list(messages)
    while rest and rest[0]['role'] in ('system', 'user'):
        prefix.append(rest.pop(0))
    groups = []
    for item in rest:
        if item['role'] == 'tool' and groups:
            groups[-1].append(item)
        else:
            groups.append([item])
    selected = []
    def encode(items):
        return tokenizer.encode(''.join(message(m) for m in items) + header('assistant'))
    if len(encode(prefix)) > budget:
        raise ValueError('Task and tool definitions exceed context budget')
    for group in reversed(groups):
        candidate = group + selected
        if len(encode(prefix + candidate)) > budget:
            break
        selected = candidate
    # Never silently drop the most recent observation and repeat a stale action.
    if groups and not selected:
        raise ValueError('Latest complete action and observation exceed context budget')
    packed = prefix + selected
    return packed, encode(packed)
