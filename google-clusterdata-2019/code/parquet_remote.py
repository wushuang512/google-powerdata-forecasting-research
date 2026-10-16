"""Exact HTTP range reader. Cache every received range for local reproducibility."""
import io,json,hashlib,struct,time,urllib.request,threading
from pathlib import Path
import pyarrow.parquet as pq
import requests
ROOT=Path(__file__).resolve().parents[1]
LOCAL=threading.local()
def session():
    if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
    return LOCAL.session
class RemoteFile(io.RawIOBase):
    def __init__(self,cell,obj,cache=True):
        self.cell=cell;self.obj=obj;self.size=int(obj['size']);self.pos=0;self.bytes=0;self.requests=0;self.cache=cache;self.prefetched=[]
        self.url=f'https://storage.googleapis.com/clusterdata_2019_{cell}/{obj["name"]}';self.dir=ROOT/'column_cache'/cell/obj['name'].removesuffix('.parquet.gz');self.dir.mkdir(parents=True,exist_ok=True)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if self.pos<0:raise ValueError('negative seek')
        return self.pos
    def cached_cover(self,start,end):
        path=self.dir/f'{start}-{end}.bin'
        if path.exists() and path.stat().st_size==end-start:return path,start
        for f in self.dir.glob('*.bin'):
            lo,hi=map(int,f.stem.split('-'))
            if lo<=start and end<=hi and f.stat().st_size==hi-lo:return f,lo
        return None
    def range(self,start,end):
        if end<=start:return b''
        for lo,hi,data in self.prefetched:
            if lo<=start and end<=hi:return data[start-lo:end-lo]
        path=self.dir/f'{start}-{end}.bin'
        cached=self.cached_cover(start,end)
        if cached:
            f,lo=cached
            with f.open('rb') as fp:fp.seek(start-lo);return fp.read(end-start)
        for attempt in range(5):
            try:
                with session().get(self.url,headers={'Range':f'bytes={start}-{end-1}','Accept-Encoding':'identity'},timeout=(20,90)) as r:
                    if r.status_code!=206:raise RuntimeError(f'Range ignored: {r.status_code}')
                    data=r.content;assert len(data)==end-start,(start,end,len(data));assert r.headers['Content-Range'].startswith(f'bytes {start}-{end-1}/')
                    hashes=r.headers.get('X-Goog-Hash','')
                    if 'md5=' in hashes:assert self.obj['md5Hash'] in hashes,'Object MD5 differs from official inventory'
                self.bytes+=len(data);self.requests+=1
                if self.cache:
                    temporary=path.with_suffix('.tmp');temporary.write_bytes(data);temporary.replace(path)
                return data
            except Exception:
                if attempt==4:raise
                time.sleep(min(12,1.5*(attempt+1)))
    def prefetch_columns(self,meta,columns,gap=256*1024):
        spans=[]
        for j in range(meta.num_row_groups):
            rg=meta.row_group(j)
            for k in range(rg.num_columns):
                c=rg.column(k)
                if c.path_in_schema in columns:
                    start=min(c.data_page_offset,c.dictionary_page_offset or c.data_page_offset)
                    spans.append((start,start+c.total_compressed_size))
        merged=[]
        if all(self.cached_cover(start,end) for start,end in spans):
            self.prefetched=[(start,end,self.range(start,end)) for start,end in spans]
            return
        for start,end in sorted(spans):
            if merged and start-merged[-1][1]<=gap:merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
            else:merged.append((start,end))
        self.prefetched=[(start,end,self.range(start,end)) for start,end in merged]
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        end=min(self.size,self.pos+n);data=self.range(self.pos,end);self.pos=end;return data
    def readinto(self,b):
        d=self.read(len(b));b[:len(d)]=d;return len(d)
def footer(cell,obj):
    rf=RemoteFile(cell,obj);last=rf.range(rf.size-8,rf.size);assert last[4:]==b'PAR1';length=struct.unpack('<I',last[:4])[0];raw=rf.range(rf.size-8-length,rf.size-8)
    meta=pq.read_metadata(io.BytesIO(b'PAR1'+raw+last))
    groups=[]
    for j in range(meta.num_row_groups):
        rg=meta.row_group(j);cols=[]
        for k in range(rg.num_columns):
            c=rg.column(k);s=c.statistics
            def value(x):
                if isinstance(x,bytes):return x.decode('utf-8','replace')
                if hasattr(x,'item'):return x.item()
                return x
            cols.append(dict(name=c.path_in_schema,compressed=c.total_compressed_size,offset=c.data_page_offset,dictionary=c.dictionary_page_offset,stats=None if s is None else dict(min=value(s.min),max=value(s.max),nulls=s.null_count)))
        groups.append(dict(rows=rg.num_rows,columns=cols))
    return dict(object=obj,row_groups=groups,total_rows=meta.num_rows,footer_sha256=hashlib.sha256(raw+last).hexdigest(),footer_bytes=length+8,network_bytes=rf.bytes)
def main():
    a=json.loads((ROOT/'docs/gcs_a_objects.json').read_text());items=[x for x in a if x['name'].startswith('instance_usage-') and x['name'].endswith('.parquet.gz')];chosen=[min(items,key=lambda x:int(x['size'])),items[0],items[len(items)//2]];out=[]
    for o in chosen:
        z=footer('a',o);out.append(z);print(json.dumps(z),flush=True)
    (ROOT/'results/parquet_pilot.json').write_text(json.dumps(out,indent=2))
if __name__=='__main__':main()
