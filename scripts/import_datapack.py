#!/usr/bin/env python3
"""Validate an annotated datapack and copy it to the local demo input folder."""
import argparse,json,math
from pathlib import Path

def validate_datapack(data):
    if not isinstance(data,dict) or not isinstance(data.get('datapackID'),str) or not data['datapackID'].strip():
        raise ValueError('A non-empty string datapackID is required')
    if not isinstance(data.get('chats'),dict):raise ValueError('chats must be an object')
    for chat in data['chats'].values():
        if not isinstance(chat,dict) or not isinstance(chat.get('content'),list):raise ValueError('Each chat needs a content array')
        for item in chat['content']:
            if not isinstance(item,dict) or item.get('type') not in ('message','separator'):raise ValueError('Content items must be messages or separators')
            if item['type']=='message':
                if not isinstance(item.get('body'),str):raise ValueError('Each message needs a string body')
                if 'prediction' in item:
                    score=item['prediction']
                    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:
                        raise ValueError('prediction must be a finite number between 0 and 1')
                if 'labels' in item and (not isinstance(item['labels'],list) or not all(isinstance(label,str) for label in item['labels'])):
                    raise ValueError('labels must be an array of strings')
    return data

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('--name',default='local-predictions.json',help='Local output filename')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parents[1]/'frontend/datapacks')
    parser.add_argument('--force',action='store_true',help='Allow replacing an existing local file')
    args=parser.parse_args()
    if Path(args.name).name != args.name or not args.name.endswith('.json') or args.name=='synthetic-demo.json':
        parser.error('--name must be a JSON filename, without directory components, and must not replace the synthetic example')
    try:
        data=validate_datapack(json.loads(args.source.read_text(encoding='utf-8')))
        args.output_dir.mkdir(parents=True,exist_ok=True)
        destination=args.output_dir/args.name
        with destination.open('w' if args.force else 'x',encoding='utf-8') as handle:
            json.dump(data,handle,indent=2,allow_nan=False)
            handle.write('\n')
    except (OSError,ValueError) as error:
        parser.exit(1,f'Import failed: {error}\n')
    print('Annotated datapack imported. Restart the demo to load the new file.')
if __name__=='__main__':main()
