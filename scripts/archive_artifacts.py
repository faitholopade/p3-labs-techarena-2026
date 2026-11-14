"""Build and verify a ZIP privately before atomically exposing its final filename."""
import argparse,hashlib,json,os,tempfile,zipfile
from pathlib import Path


def archive(source,destination,exclude=()):
    source=Path(source).resolve();destination=Path(destination).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    files=[source] if source.is_file() else [p for p in sorted(source.rglob('*')) if p.is_file() and not any(part.startswith('.') for part in p.relative_to(source).parts) and not any(p.relative_to(source).as_posix()==x.rstrip('/') or p.relative_to(source).as_posix().startswith(x.rstrip('/')+'/') for x in exclude)]
    if not files:raise ValueError('No files to archive.')
    if source.is_dir() and destination.is_relative_to(source):raise ValueError('Archive must be outside its source directory.')
    fd,temp=tempfile.mkstemp(prefix='.building-',suffix='.zip',dir=destination.parent);os.close(fd)
    try:
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for file in files:
                z.write(file,file.name if source.is_file() else source.name+'/'+file.relative_to(source).as_posix())
        with zipfile.ZipFile(temp) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC validation failed.')
            names=z.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive members.')
        with open(temp,'rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        size=os.path.getsize(temp);os.replace(temp,destination)
    finally:
        if os.path.exists(temp):os.unlink(temp)
    return {'filename':destination.name,'sha256':digest,'bytes':size,'files':len(files),'zip_crc_valid':True,'atomic_publication':True}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('destination',type=Path)
    p.add_argument('--exclude',action='append',default=[]);p.add_argument('--report',type=Path)
    a=p.parse_args();result=archive(a.source,a.destination,a.exclude)
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
