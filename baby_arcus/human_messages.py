"""Human input validation; delivery does not claim language understanding."""
from baby_arcus.contracts import fields,identifier,ContractError
SENDERS={"you":"You","wife":"Your wife"}
def validate_message(value):
    fields(value,("request_id","sender","text"))
    identifier(value["request_id"])
    if not isinstance(value["sender"],str) or value["sender"] not in SENDERS: raise ContractError("Unknown sender")
    if not isinstance(value["text"],str) or not value["text"].strip() or len(value["text"])>2000:
        raise ContractError("Message must contain 1–2000 characters")

