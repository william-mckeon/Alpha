"""Nonblocking host bridge to the separate language process or authenticated service.

This module intentionally uses only the standard library and existing HTTP transport.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

from baby_arcus.contracts import ContractError


class LanguageRuntime:
    def __init__(self,app,python,project,config='configs/baby_arcus/language.json'):
        self.app=app;self.python=Path(python);self.project=Path(project);self.config=config
        self.info={'status':'stopped','reason':'Enable language learning to begin the bounded pilot',
                   'message_receipts':{},'expressions':[]}
        self.enabled=False;self.closing=False;self.busy=False;self.next_tick=0
        self.commands=queue.Queue(maxsize=8);self.thread=None;self.process=None
        self.guard=threading.RLock();self.revision=0
        settings=json.loads((self.project/config).read_text(encoding='utf-8'))
        self.interval=settings['decision_interval_seconds']
        self.log_root=self.project/settings['language_root']
        pointer=self.log_root/'active.json'
        if pointer.exists() and not os.environ.get('ARCUS_LANGUAGE_URL'):
            saved=json.loads(pointer.read_text(encoding='utf-8'))
            self.info.update(saved.get('summary',{
                'training_tokens':saved['training_tokens'],'decisions':saved['decisions'],
                'remaining_training_tokens':max(0,settings['session_training_tokens']-saved['training_tokens'])}))
            self.info.update(status='stopped',reason='Saved language progress; enable to continue')

    def snapshot(self):
        with self.guard:return deepcopy({**self.info,'enabled':self.enabled,'busy':self.busy})

    def control(self,action):
        if action not in ('start','stop','pause','resume','restart'):raise ContractError('Unknown language control')
        with self.guard:
            if action=='stop':
                self.enabled=False;self.revision+=1
                self.info['reason']='Stopping after the current transaction' if self.busy else 'Stopped by you'
                self.info['status']='stopping' if self.busy else 'stopped'
                return self.snapshot()
            if self.closing:raise ContractError('Language host is closing')
            if action=='start':
                if self.enabled:return self.snapshot()
                self.enabled=True
                self.info.update(status='ready' if self.thread and self.thread.is_alive() else 'loading',
                                 reason='Language pilot enabled')
            elif not self.enabled:raise ContractError('Enable the language pilot before changing its listening stream')
            self.ensure_worker()
            if action!='start':
                try:self.commands.put_nowait((self.revision,{'op':'control','action':action}))
                except queue.Full:raise ContractError('Language control queue full; wait for current transaction')
            return self.snapshot()

    def ensure_worker(self):
        if not self.thread or not self.thread.is_alive():
            self.info['status']='loading'
            self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def on_tick(self):
        # Called under the playroom lock; never waits for GPU or disk work.
        with self.guard:
            if not self.enabled or self.busy or self.closing or time.monotonic()<self.next_tick:return
            if self.app.world.paused or self.app.world.body.sleep_state!='awake':return
            if self.info['status'] not in ('ready','running'):return
            if self.info.get('remaining_training_tokens',1)<=0 or self.info.get('decisions',0)>=256:
                self.enabled=False;self.info.update(status='budget_complete',reason='Bounded pilot completed; review before more training')
                return
            self.app.conversation.release(True)
            receipts=self.info.get('message_receipts',{})
            messages=[{k:r[k] for k in ('request_id','sender','text')} for r in self.app.conversation.rows
                      if r['status']=='available' and receipts.get(r['request_id'],{}).get('status') not in ('trained','received_only')][:4]
            try:self.commands.put_nowait((self.revision,{'op':'step','messages':messages}))
            except queue.Full:return
            self.busy=True;self.next_tick=time.monotonic()+self.interval

    def _run(self):
        process=None;stderr=None
        try:
            url=os.environ.get('ARCUS_LANGUAGE_URL')
            if url:
                from baby_arcus.transport import Client
                token=os.environ.get('ARCUS_LANGUAGE_TOKEN')
                if not token:raise ValueError('Remote language worker needs ARCUS_LANGUAGE_TOKEN')
                client=Client(url,token,timeout=180,attempts=1)
                receive=lambda request:client.request('POST','/v1/language',request)
                initial=receive({'op':'status'})
            else:
                self.log_root.mkdir(parents=True,exist_ok=True)
                stderr=(self.log_root/'worker.stderr.log').open('a',encoding='utf-8')
                process=subprocess.Popen([str(self.python),'-u','-m','baby_arcus.services.language_worker','--config',self.config],
                    cwd=self.project,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
                    text=True,encoding='utf-8',bufsize=1,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                self.process=process;replies=queue.Queue()
                def reader():
                    for line in process.stdout:replies.put(line)
                    replies.put(None)
                threading.Thread(target=reader,daemon=True).start()
                def response():
                    line=replies.get(timeout=180)
                    if line is None:raise RuntimeError('Language worker exited; inspect worker.stderr.log')
                    data=json.loads(line)
                    if data.get('status')=='error':raise RuntimeError(data.get('error','Language worker failed'))
                    return data
                initial=response()
                def receive(request):
                    process.stdin.write(json.dumps(request,ensure_ascii=True)+'\n');process.stdin.flush()
                    return response()
            with self.guard:self.info=initial
            while not self.closing:
                try:revision,request=self.commands.get(timeout=.5)
                except queue.Empty:continue
                with self.guard:
                    if revision!=self.revision:
                        self.busy=False
                        if not self.enabled:self.info.update(status='stopped',reason='Stopped by you')
                        continue
                    self.busy=True
                result=receive(request)
                with self.guard:
                    self.info=result;self.busy=False
                    if not self.enabled:self.info.update(status='stopped',reason='Stopped by you; last transaction saved')
                    self.next_tick=time.monotonic()+self.interval
        except Exception as exc:
            with self.guard:
                self.revision+=1
                self.enabled=False;self.busy=False;self.info.update(status='error',reason=str(exc))
        finally:
            if process:
                if process.poll() is None:
                    process.stdin.close()
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
                process.stdout.close()
            if stderr:stderr.close()

    def close(self):
        with self.guard:self.closing=True;self.enabled=False;self.revision+=1
        if self.thread:self.thread.join(timeout=5)
        if self.process and self.process.poll() is None:self.process.terminate()
