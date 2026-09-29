"""Whole-turn bounded history. Disk persistence is explicit and session-isolated."""
import json
import re
from pathlib import Path
from langchain_core.messages import messages_to_dict, messages_from_dict


class ConversationMemory:
    def __init__(self, max_turns=4, max_chars=12000, directory=None):
        if not 1<=max_turns<=8 or not 100<=max_chars<=24000: raise ValueError('Invalid memory limits')
        self.max_turns,self.max_chars=max_turns,max_chars
        self.directory=Path(directory) if directory is not None else None
        self.sessions={}
        if self.directory: self.directory.mkdir(parents=True,exist_ok=True)

    def _key(self,session):
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}',session): raise ValueError('Invalid session ID')
        return session

    def _turns(self,session):
        session=self._key(session)
        if session not in self.sessions:
            path=self.directory/(session+'.json') if self.directory else None
            if path and path.exists():
                if path.stat().st_size>200000: raise ValueError('Persisted memory exceeds budget')
                self.sessions[session]=json.loads(path.read_text())
                turns=self.sessions[session]
                if not isinstance(turns,list) or any(not isinstance(turn,list) for turn in turns):
                    raise ValueError('Invalid saved history')
                while turns and (len(turns)>self.max_turns or len(json.dumps(turns))>self.max_chars): turns.pop(0)
            else: self.sessions[session]=[]
        return self.sessions[session]

    def get(self,session):
        turns=self._turns(session)
        return messages_from_dict([message for turn in turns for message in turn])

    def add(self,session,messages):
        turns=list(self._turns(session))+[messages_to_dict(messages)]
        while turns and (len(turns)>self.max_turns or len(json.dumps(turns))>self.max_chars): turns.pop(0)
        self.sessions[session]=turns
        if self.directory:
            path=self.directory/(session+'.json');tmp=path.with_suffix('.pending')
            tmp.write_text(json.dumps(turns));tmp.replace(path)

    def delete(self,session):
        self._key(session);self.sessions.pop(session,None)
        if self.directory: (self.directory/(session+'.json')).unlink(missing_ok=True)
