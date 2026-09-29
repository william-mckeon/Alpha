import unittest
from arcus3.tools import dispatch

class ToolsTests(unittest.TestCase):
    def test_real_results(self):
        self.assertEqual(dispatch('echo',{'text':'actual'}),{'ok':True,'result':'actual'})
        self.assertEqual(dispatch('calculate',{'operation':'add','a':17,'b':25})['result'],42)
    def test_rejections_and_execution_error(self):
        for name,args in [('shell',{'command':'whoami'}),('echo',{'text':4}),('echo',{'text':'x','extra':1}),
                          ('calculate',{'operation':'divide','a':1,'b':0}),
                          ('calculate',{'operation':'add','a':True,'b':1}),
                          ('calculate',{'operation':'add','a':float('nan'),'b':1})]:
            self.assertFalse(dispatch(name,args)['ok'])
