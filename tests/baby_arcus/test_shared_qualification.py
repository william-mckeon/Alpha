import json,tempfile,unittest
from pathlib import Path
from baby_arcus.shared_qualification import compile_report,verify_evidence,GATES,LIVE_CHECKS,source_snapshot

def passing_records():
    records={
        'integration':{'integration':True,'diagnostics':{'authenticated_http':True,'one_core':True,'original_body_unchanged':True,
            'gradients':{key:True for key in ('rgb','body_sensations','text','internal','pixels','gaze','future_body','future_rgb','action_quality','memory')}}},
        'posture':{'settings':{'episodes':200},'results':{key:{'parent':200,'shared':200} for key in ('standing','lying','sitting')}},
        'retention':{'episodes':200,'approach':{'parent':200,'shared':200},'language_examples':200,'language_nll':{'parent':8.,'shared':8.1}},
        'transfer':{'split':'confirmation','examples_per_condition':300,'accuracy':{'commands:full':1.,'rest:full':1.,'color_reference:full':1.,
            'color_reference:no_rgb':.5,'color_reference:no_hearing':.5,'color_reference:shuffled_rgb':.5},'per_command':{key:1. for key in json.loads(GATES.read_text())['required_commands']}},
        'pixels':{'split':'confirmation','examples':1000,'count_accuracy':.96,'ball_iou':.82,'surface_color_accuracy':1.,'blank_count_accuracy':.6},
        'live':{'live':True,'recovery':True,'actions_passed':True,'hearing_passed':True,'checks':dict.fromkeys(LIVE_CHECKS,True)},
        'recovery':{'device':'cuda','recovery':True,'losses':[1.,1.],'mismatched_tensors':[]},
        'curiosity':{'split':'confirmation','examples':384,'scenes':32,'depth_capacity':.25,
            'prediction':{'body':.01,'body_persistence':.03,'body_shuffled':.02,'rgb':.01,'rgb_persistence':.03},
            'exploration':{'learned':{'discoveries':5},'random':{'discoveries':3},'no_action':{'discoveries':1}}}}
    return {key:dict(value,runtime_sources=source_snapshot()) for key,value in records.items()}

class QualificationTests(unittest.TestCase):
    def test_manifest_binds_backbone_and_sensory_dependencies(self):
        sources=source_snapshot()
        for path in ('arcus/model.py','arcus/moe.py','arcus/backbone.py','baby_arcus/language_stream.py','baby_arcus/body_visual.py'):
            self.assertIn(path,sources)

    def test_measured_report_survives_relocation_but_not_evidence_changes(self):
        import shutil
        manifest={'generation':'a'*32,'sha256':'b'*64,'depth_capacity':.25}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'source';root.mkdir();paths={}
            for name,value in passing_records().items():
                path=root/(name+'.json');path.write_text(json.dumps(dict(value,candidate=manifest)));paths[name]=path
            report=compile_report(root,manifest,paths);self.assertTrue(report['qualified'])
            self.assertTrue(all(not Path(item['path']).is_absolute() for item in report['evidence'].values()))
            moved=Path(folder)/'moved';shutil.copytree(root,moved);verify_evidence(report,moved)
            path=moved/'transfer.json';path.write_text(path.read_text()+'\n')
            with self.assertRaises(ValueError):verify_evidence(report,moved)
    def test_missing_or_mismatched_evidence_cannot_qualify(self):
        manifest={'generation':'a'*32,'sha256':'b'*64}
        with self.assertRaises(ValueError):compile_report(None,manifest,{})
        with self.assertRaises(ValueError):verify_evidence({'integration':True,'retention':True,'cross_modal':True,'live':True})
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'wrong.json';path.write_text(json.dumps({'candidate':{'generation':'c'*32,'sha256':'b'*64}}))
            with self.assertRaises(ValueError):compile_report(root,manifest,{name:path for name in ('integration','posture','retention','transfer','pixels','live')})

    def test_validation_scores_are_not_confirmation(self):
        manifest={'generation':'a'*32,'sha256':'b'*64}
        records=passing_records();records['transfer']['split']='validation'
        with tempfile.TemporaryDirectory() as root:
            paths={}
            for name,value in records.items():
                path=Path(root)/(name+'.json');path.write_text(json.dumps(dict(value,candidate=manifest)));paths[name]=path
            report=compile_report(root,manifest,paths)
            self.assertTrue(report['integration']);self.assertTrue(report['retention']);self.assertTrue(report['live'])
            self.assertFalse(report['qualified'])
            with self.assertRaises(ValueError):verify_evidence(report)
