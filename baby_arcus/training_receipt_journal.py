"""Portable compressed receipt blocks embedded in each immutable checkpoint.

No external sidecar is required to resume or copy a checkpoint. The loader restores
the original list API. Checksums/counts reject truncated or corrupt blocks.
"""
import hashlib
import json
import zlib

BLOCK_RECORDS=256
MAX_BLOCK_BYTES=16*1024*1024


def pack(receipts):
    def validate(value):
        if type(value) in (str,int,float,bool,type(None)):return
        if type(value) is list:
            for item in value:validate(item)
        elif type(value) is dict and all(type(key) is str for key in value):
            for item in value.values():validate(item)
        else:raise TypeError('Receipt value is not losslessly JSON encodable')
    validate(receipts)
    blocks=[]
    for start in range(0,len(receipts),BLOCK_RECORDS):
        rows=receipts[start:start+BLOCK_RECORDS]
        raw=json.dumps(rows,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()
        if len(raw)>MAX_BLOCK_BYTES:raise ValueError('Receipt block exceeds bounded journal size')
        blocks.append({'count':len(rows),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),
                       'payload':zlib.compress(raw,level=1)})
    return {'schema':'alpha-receipts-zlib-v1','count':len(receipts),'blocks':blocks}


def unpack(journal):
    if journal.get('schema')!='alpha-receipts-zlib-v1':raise ValueError('Unknown receipt encoding')
    records=[]
    for block in journal['blocks']:
        if not 0<block['bytes']<=MAX_BLOCK_BYTES:raise ValueError('Invalid receipt block length')
        reader=zlib.decompressobj()
        raw=reader.decompress(block['payload'],MAX_BLOCK_BYTES+1)
        if (len(raw)!=block['bytes'] or not reader.eof or reader.unused_data or reader.unconsumed_tail
                or hashlib.sha256(raw).hexdigest()!=block['sha256']):
            raise ValueError('Receipt journal integrity failure')
        rows=json.loads(raw)
        if not isinstance(rows,list) or len(rows)!=block['count'] or not 1<=len(rows)<=BLOCK_RECORDS:
            raise ValueError('Receipt journal count mismatch')
        records.extend(rows)
    if len(records)!=journal['count']:raise ValueError('Incomplete receipt journal')
    return records


def encode_progress(progress):
    result=dict(progress)
    if 'receipt_journal' in result:raise ValueError('Progress must be decoded before saving')
    # Small fixtures/old snapshots retain their original on-disk representation.
    if len(result.get('receipts',[]))>=BLOCK_RECORDS:
        try:journal=pack(result['receipts'])
        except (TypeError,ValueError):return result # Legacy non-JSON receipts remain exact.
        result.pop('receipts');result['receipt_journal']=journal
    return result


def decode_progress(progress):
    if 'receipt_journal' not in progress:return progress
    if 'receipts' in progress:raise ValueError('Ambiguous receipt history')
    result=dict(progress);result['receipts']=unpack(result.pop('receipt_journal'))
    return result
