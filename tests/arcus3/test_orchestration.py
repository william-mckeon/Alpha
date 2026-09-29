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
