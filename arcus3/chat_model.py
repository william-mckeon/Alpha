"""LangChain adapter around an injected, read-only donor generation function."""
import json
from typing import Any, Callable
from uuid import uuid4
from pydantic import Field
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from arcus3.tools import SCHEMAS


class ArcusChatModel(BaseChatModel):
    backend: Callable = Field(exclude=True)
    tools: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self): return 'arcus3-local-donor'

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice not in (None,'auto') or kwargs:
            raise ValueError('Only automatic allowlisted tools are supported')
        if any(tool not in SCHEMAS for tool in tools) or len({t['function']['name'] for t in tools})!=len(tools):
            raise ValueError('Only exact approved tool schemas may be bound')
        return self.model_copy(update={'tools':list(tools)})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop or kwargs: raise ValueError('Unapproved generation overrides')
        serialized=[]
        for message in messages:
            if not isinstance(message.content,str): raise ValueError('Text messages only')
            role = ('system' if isinstance(message,SystemMessage) else 'user' if isinstance(message,HumanMessage)
                    else 'assistant' if isinstance(message,AIMessage) else 'tool' if isinstance(message,ToolMessage) else None)
            if role is None: raise ValueError('Unsupported message type')
            content=message.content
            if isinstance(message,AIMessage) and message.tool_calls:
                content=json.dumps({'tool_calls':[{'id':c['id'],'name':c['name'],'arguments':c['args']} for c in message.tool_calls]})
            if isinstance(message,ToolMessage):
                content=json.dumps({'tool_call_id':message.tool_call_id,'result':message.content})
            serialized.append({'role':role,'content':content})
        if self.tools:
            instruction=('Available tools: '+json.dumps(self.tools)+
                '\nWhen a tool is needed, return only {"name":"tool_name","arguments":{...}}. '
                'After a tool result, answer the user using that result; do not repeat the call unnecessarily.')
            if serialized and serialized[0]['role']=='system': serialized[0]={**serialized[0],'content':serialized[0]['content']+'\n'+instruction}
            else: serialized.insert(0,{'role':'system','content':instruction})
        result=self.backend(serialized)
        text=result['response'];calls=[];invalid=[];protocol_warning=None
        # Retain raw text even when parsed. Accept only the explicit application protocol.
        try:
            parsed,end=json.JSONDecoder().raw_decode(text.lstrip())
            if self.tools and isinstance(parsed,dict) and 'name' in parsed:
                if set(parsed)!={'name','arguments'} or not isinstance(parsed['name'],str) or not isinstance(parsed['arguments'],dict):
                    raise ValueError('Malformed tool call')
                if parsed['name'] not in {t['function']['name'] for t in self.tools}:
                    raise ValueError('Tool is not bound for this request')
                calls=[{'name':parsed['name'],'args':parsed['arguments'],'id':'call_'+uuid4().hex,'type':'tool_call'}]
                trailing=text.lstrip()[end:].strip()
                if trailing:
                    # Prose is not a tool observation. Execute the complete leading
                    # call and require a subsequent model turn using the real result.
                    if trailing.startswith(('{','[','<tool_call>')): raise ValueError('Ambiguous multiple calls')
                    protocol_warning='Trailing prose retained but not trusted as a tool result'
        except (ValueError,TypeError):
            calls=[]
            if text.lstrip().startswith('{') and self.tools:
                invalid=[{'name':None,'args':text,'id':None,'error':'Invalid tool JSON','type':'invalid_tool_call'}]
        usage=result.get('usage',{})
        message=AIMessage(content=text,tool_calls=calls,invalid_tool_calls=invalid,
            response_metadata={**{k:v for k,v in result.items() if k in ('seconds','truncated')},
                               'protocol_warning':protocol_warning},
            usage_metadata=usage or None)
        return ChatResult(generations=[ChatGeneration(message=message)])
