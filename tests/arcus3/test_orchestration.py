import unittest
from arcus3.orchestration import invoke_messages

class OrchestrationTests(unittest.TestCase):
    def test_messages_and_output_are_preserved(self):
        messages=[{'role':'user','content':'Hi'}]
        seen=[]
        def generate(value):
            seen.append(value)
            return {'text':'Hello', 'tool_calls':[]}
        result=invoke_messages(generate,messages)
        self.assertEqual(seen,[messages])
        self.assertEqual(result,{'text':'Hello','tool_calls':[]})
    def test_error_propagates(self):
        def fail(_): raise RuntimeError('donor failed')
        with self.assertRaisesRegex(RuntimeError,'donor failed'):
            invoke_messages(fail,[{'role':'user','content':'Hi'}])
class ApplicationTests(unittest.TestCase):
    def test_roundtrip_and_budget(self):
        from arcus3.chat_model import ArcusChatModel
        from arcus3.tools import SCHEMAS
        from arcus3.orchestration import application_turn
        from langchain_core.messages import HumanMessage
        seen=[]
        def generate(messages):
            seen.append(messages)
            return {'response':'42' if messages[-1]['role']=='tool' else '{"name":"calculate","arguments":{"operation":"add","a":17,"b":25}}'}
        chat=ArcusChatModel(backend=generate).bind_tools(SCHEMAS)
        result=application_turn(chat,[HumanMessage(content='add')],lambda:None)
        self.assertEqual(result['status'],'complete');self.assertEqual(result['tool_calls'],1)
        self.assertEqual(result['messages'][-1].content,'42')
        looping=ArcusChatModel(backend=lambda m:{'response':'{"name":"echo","arguments":{"text":"x"}}'}).bind_tools(SCHEMAS)
        result=application_turn(looping,[HumanMessage(content='loop')],lambda:None,max_model_calls=4,max_tool_calls=1)
        self.assertEqual(result['status'],'tool_budget_exhausted');self.assertEqual(result['tool_calls'],1)
    def test_error_observation_and_cancel(self):
        from arcus3.chat_model import ArcusChatModel
        from arcus3.tools import SCHEMAS
        from arcus3.orchestration import application_turn
        from langchain_core.messages import HumanMessage
        seen=[]
        def generate(messages):
            seen.append(messages)
            return {'response':'Cannot divide by zero.' if messages[-1]['role']=='tool' else '{"name":"calculate","arguments":{"operation":"divide","a":1,"b":0}}'}
        chat=ArcusChatModel(backend=generate).bind_tools(SCHEMAS)
        result=application_turn(chat,[HumanMessage(content='divide')],lambda:None)
        self.assertEqual(result['status'],'complete');self.assertIn('Division by zero',seen[-1][-1]['content'])
        def cancel():raise RuntimeError('paused')
        with self.assertRaisesRegex(RuntimeError,'paused'):application_turn(chat,[],cancel)
