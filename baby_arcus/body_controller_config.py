"""Select a qualified body adapter; never activate unqualified larger weights."""
import json
from pathlib import Path

def configuration(project):
    project=Path(project)
    approach=project/'runs/arcus_approach_v2'
    if (approach/'qualification/report.json').is_file():
        result=json.loads((approach/'qualification/report.json').read_text())
        manifest=json.loads((approach/'manifest.json').read_text())
        if (result.get('gate_passed') is True and result.get('checkpoint_unchanged') is True
            and result.get('old_skills_preserved') is True
            and result.get('checkpoint_sha256')==manifest.get('checkpoint_sha256')):
            return {'checkpoint':approach/'postures.pt','expected_hash':manifest['checkpoint_sha256'],
                    'goals':('standing','lying','sitting','approach'),'label':f"Arcus interaction model · {manifest['parameters']:,} parameters"}
    sitting=project/"runs/arcus_sitting"
    if (sitting/"qualification/report.json").is_file():
        result=json.loads((sitting/"qualification/report.json").read_text())
        manifest=json.loads((sitting/"manifest.json").read_text())
        if (result.get("gate_passed") is True and result.get("checkpoint_unchanged") is True
            and manifest.get("old_skills_preserved") is True and manifest.get("trunk_weights_preserved") is True
            and result.get("checkpoint_sha256")==manifest.get("checkpoint_sha256")):
            return {"checkpoint":sitting/"postures.pt","expected_hash":manifest["checkpoint_sha256"],
                    "goals":("standing","lying","sitting"),"label":f'Arcus learned postures · {manifest["parameters"]:,} parameters'}
    postures=project/"runs/arcus_postures_v2"
    if (postures/"qualification/report.json").is_file():
        result=json.loads((postures/"qualification/report.json").read_text())
        manifest=json.loads((postures/"manifest.json").read_text())
        if (result.get("gate_passed") is True and result.get("checkpoint_unchanged") is True
            and manifest.get("standing_weights_preserved") is True and manifest.get("trunk_weights_preserved") is True
            and result.get("checkpoint_sha256")==manifest.get("checkpoint_sha256")):
            return {"checkpoint":postures/"postures.pt","expected_hash":manifest["checkpoint_sha256"],
                    "goals":("standing","lying"),"label":f'Arcus learned postures · {manifest["parameters"]:,} parameters'}
    root=project/"runs/arcus_large_body"
    report=root/"qualification/report.json"
    if report.is_file():
        result=json.loads(report.read_text())
        manifest=json.loads((root/"manifest.json").read_text())
        if (result.get("gate_passed") is True and result.get("checkpoint_unchanged") is True
            and manifest.get("trunk_weights_preserved") is True
            and result.get("checkpoint_sha256")==manifest.get("checkpoint_sha256")):
            return {"checkpoint":root/"standing.pt","expected_hash":manifest["checkpoint_sha256"],
                    "label":f'Arcus large body adapter · {manifest["parameters"]:,} parameters'}
    return {"checkpoint":project/"runs/arcus_standing_pilot/standing.pt",
            "label":"Standing pilot · 48,351 parameters"}
