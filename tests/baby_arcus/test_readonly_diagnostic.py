import unittest
from scripts.diagnose_alpha_container import validate_request, validate_heads, RELEASE


class DiagnosticTests(unittest.TestCase):
    def test_readonly_scope_and_expiry(self):
        scope={'image_id':'sha256:'+'a'*64,'host_fingerprint':'b'*64}
        request={'schema':'alpha-readonly-diagnostic-v1','user_authorized_readonly':True,
                 'scope':scope,'stage':'model','checkpoint_sha256':RELEASE,
                 'training_updates':0,'created_at':1000}
        self.assertEqual(validate_request(request,scope,1100),request)
        self.assertEqual(validate_request({**request,'capacity':.8},scope,1100)['capacity'],.8)
        for change in ({'training_updates':1},{'stage':'train'},{'user_authorized_readonly':False},
                       {'created_at':0},{'checkpoint_sha256':'other'},{'scope':{}},
                       {'capacity':0},{'capacity':1.1},{'capacity':float('nan')},{'capacity':True}):
            with self.assertRaises(ValueError): validate_request({**request,**change},scope,1100)

    def test_expected_joint_masks_are_not_numerical_failures(self):
        import torch
        from baby_arcus.body_dynamics import JOINTS
        from baby_arcus.body_vocabulary import mask
        row={'senses':{'joint_positions':{key:0 for key in JOINTS}}}
        allowed=torch.tensor([mask(row['senses'])])
        logits=torch.zeros(allowed.shape).masked_fill(~allowed,-torch.inf)
        self.assertGreater(validate_heads({'body':logits},row)['body'],0)
        broken=logits.clone(); broken[0,0]=torch.nan
        with self.assertRaises(RuntimeError): validate_heads({'body':broken},row)
        broken=logits.clone(); broken[~allowed]=torch.inf
        with self.assertRaises(RuntimeError): validate_heads({'body':broken},row)
        with self.assertRaises(RuntimeError): validate_heads({'rest':torch.tensor([[-torch.inf]])},row)


if __name__=='__main__':unittest.main()
