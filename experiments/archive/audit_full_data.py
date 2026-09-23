"""Independent row-by-row audit of stored full-split tokenization against pinned source."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from run_full_lm import atomic_json,digest

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',default='data/wikitext103_bpe8k');ap.add_argument('--out',required=True)
    args=ap.parse_args();root=Path(args.data);m=json.loads((root/'manifest.json').read_text());tok=Tokenizer.from_file(str(root/'tokenizer.json'));report={}
    assert digest(root/'tokenizer.json')==m['tokenizer_sha256']
    for split,meta in m['splits'].items():
        stored=np.memmap(root/f'{split}.bin',dtype='<u2',mode='r');offset=0;rows=0;raw=hashlib.sha256();bytes_count=0
        assert digest(root/f'{split}.bin')==meta['token_sha256']
        for source in meta['source_files']:
            path=Path(hf_hub_download(m['dataset'],source['name'],repo_type='dataset',revision=m['revision']))
            assert digest(path)==source['sha256'];assert pq.ParquetFile(path).metadata.num_rows==source['rows']
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=['text']):
                texts=[x+'\n' for x in batch.column(0).to_pylist()]
                encodings=tok.encode_batch(texts)
                for text,encoding in zip(texts,encodings):
                    values=np.asarray(encoding.ids,dtype='<u2'); n=len(values)
                    assert np.array_equal(values,stored[offset:offset+n]),f'{split} token mismatch at row {rows}'
                    assert tok.decode(encoding.ids,skip_special_tokens=False)==text,f'{split} text loss at row {rows}'
                    raw.update(text.encode('utf-8'));bytes_count+=len(text.encode('utf-8'));offset+=n;rows+=1
        assert offset==len(stored)==meta['tokens'];assert rows==meta['rows']
        assert raw.hexdigest()==meta['normalized_text_sha256'];assert bytes_count==meta['utf8_bytes_with_newlines']
        report[split]={'verified_source_rows':rows,'verified_tokens':offset,'verified_utf8_bytes':bytes_count,
            'every_row_lossless_decode':True,'stored_tokens_match_source':True,'source_sha256_verified':True}
        print(json.dumps({'split':split,**report[split]}),flush=True)
    atomic_json(args.out,{'dataset_revision':m['revision'],'dataset_manifest_sha256':digest(root/'manifest.json'),
        'audit_source_sha256':digest(__file__),'splits':report,'passed':True})
if __name__=='__main__':main()
