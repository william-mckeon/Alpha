import unittest
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
from arcus3.chat_model import ArcusChatModel
from arcus3.tools import SCHEMAS

class ChatTests(unittest.TestCase):
    def test_leading_call_with_untrusted_prose(self):
        raw='{"name":"echo","arguments":{"text":"hello"}}\nThe tool returned something.'
        answer=ArcusChatModel(backend=lambda m:{'response':raw}).bind_tools(SCHEMAS).invoke('echo')
        self.assertEqual(answer.content,raw);self.assertEqual(len(answer.tool_calls),1)
        self.assertIsNotNone(answer.response_metadata['protocol_warning'])
        ambiguous=raw.split('\n')[0]+'\n{"name":"echo","arguments":{"text":"other"}}'
        answer=ArcusChatModel(backend=lambda m:{'response':ambiguous}).bind_tools(SCHEMAS).invoke('echo')
        self.assertFalse(answer.tool_calls);self.assertTrue(answer.invalid_tool_calls)
    def test_invoke_and_bind(self):
        seen=[]
        def generate(messages):
            seen.extend(messages)
            return {'response':'{"name":"echo","arguments":{"text":"hello"}}','usage':{'input_tokens':3,'output_tokens':4,'total_tokens':7}}
        model=ArcusChatModel(backend=generate)
        bound=model.bind_tools(SCHEMAS)
        answer=bound.invoke([HumanMessage(content='Echo hello')])
        self.assertEqual(answer.tool_calls[0]['args'],{'text':'hello'})
        self.assertEqual(answer.usage_metadata['total_tokens'],7)
        self.assertFalse(model.tools)
        self.assertIn('Available tools:',seen[0]['content'])
        seen.clear()
        bound.invoke([answer,ToolMessage(content='hello',tool_call_id=answer.tool_calls[0]['id'])])
        self.assertIn(answer.tool_calls[0]['id'],seen[-1]['content'])
    def test_reject_arbitrary_bindings_and_malformed_json(self):
        model=ArcusChatModel(backend=lambda m:{'response':'{"name":'})
        with self.assertRaises(ValueError):model.bind_tools([{'name':'shell'}])
        self.assertTrue(model.bind_tools(SCHEMAS).invoke('test').invalid_tool_calls)
        other=ArcusChatModel(backend=lambda m:{'response':'{"name":"calculate","arguments":{"operation":"add","a":1,"b":1}}'})
        self.assertTrue(other.bind_tools(SCHEMAS[:1]).invoke('test').invalid_tool_calls)
