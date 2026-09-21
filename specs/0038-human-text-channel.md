# Human text channel
Status: implemented as input transport; language model consumption and generated replies are not implemented.

Humans send {request_id,sender,text}; sender is 'you' or 'wife', with server-selected display labels. These labels are local participant selections, not authenticated personal identity. Text must be nonblank and <=2000 characters. IDs are idempotent; changed content under an existing ID is rejected.

A durable atomic conversation file stores at most 500 messages. At capacity new messages are rejected explicitly; pending messages are never silently discarded. Existing IDs remain retryable. Export session includes the history. Messages have queued/available status and model_read=false. Awake eyes-closed input is available. Sleeping input queues until an awake read; waking does not claim model understanding.

The browser renders textContent and never interprets messages as HTML or executable commands. The restricted tool endpoint can read available messages, but cannot submit human messages. No generated Arcus replies are fabricated. Human and model credentials remain separate.

