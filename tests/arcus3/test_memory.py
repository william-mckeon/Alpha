import tempfile
import unittest
from pathlib import Path
from langchain_core.messages import HumanMessage,AIMessage
from arcus3.memory import ConversationMemory

class MemoryTests(unittest.TestCase):
    def test_isolation_and_turn_trimming(self):
        memory=ConversationMemory(max_turns=1)
        memory.add('a',[HumanMessage(content='one'),AIMessage(content='first')])
        memory.add('a',[HumanMessage(content='two'),AIMessage(content='second')])
        self.assertEqual(len(memory.get('a')),2);self.assertEqual(memory.get('a')[0].content,'two')
        self.assertEqual(memory.get('b'),[])
        with self.assertRaises(ValueError):memory.get('../escape')
    def test_opt_in_persistence_and_delete(self):
        with tempfile.TemporaryDirectory() as root:
            memory=ConversationMemory(directory=root)
            memory.add('a',[HumanMessage(content='remember')])
            self.assertEqual(ConversationMemory(directory=root).get('a')[0].content,'remember')
            memory.delete('a');self.assertFalse((Path(root)/'a.json').exists());self.assertEqual(memory.get('a'),[])
    def test_oversized_turn_dropped_whole(self):
        memory=ConversationMemory(max_chars=100)
        memory.add('a',[HumanMessage(content='x'*1000)])
        self.assertEqual(memory.get('a'),[])
