"""Read bounded members from the pinned public SoccerNet tracking archives."""
import concurrent.futures
import configparser
import io
import struct
import time
from urllib.request import Request, urlopen
import zipfile
import zlib

class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size = url, size
        self.pos = 0
        self.bytes_read = 0
        self.cache_start = -1
        self.cache = b''
    def readable(self):
        return True
    def seekable(self):
        return True
    def tell(self):
        return self.pos
    def seek(self, offset, whence=0):
        start = (0, self.pos, self.size)[whence]
        if start + offset < 0:
            raise ValueError('Negative archive position')
        self.pos = start + offset
        return self.pos
    def read(self, count=-1):
        end = min(self.size, self.pos + count) if count >= 0 else self.size
        if end <= self.pos:
            return b''
        if end - self.pos > 12_000_000 or self.bytes_read + end - self.pos > 100_000_000:
            raise ValueError('Download budget exceeded')
        start = self.pos
        if self.cache_start <= start and end <= self.cache_start + len(self.cache):
            self.pos = end
            return self.cache[start-self.cache_start:end-self.cache_start]
        want = end
        if count <= 64:
            end = min(self.size, start + 400_000)
        request = Request(self.url, headers={'Range': f'bytes={start}-{end-1}', 'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=25) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.size}':
                raise ValueError('Unexpected HTTP range response')
            data = response.read(end - start + 1)
        if len(data) != end - start:
            raise ValueError('Incomplete HTTP response')
        self.bytes_read += len(data)
        self.cache_start, self.cache = start, data
        self.pos = want
        return data[:want-start]

REVISION = '3777f456d908570e10b19e7354744640e77c351e'
DATASET = 'SoccerNet/SN-Tracking-2023'
URL_ROOT = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}'
SIZES = {'train': 9575523385, 'test': 8711601474}


def load_entry(split, info):
    for attempt in range(3):
        try:
            stream = RemoteZip(f'{URL_ROOT}/{split}.zip', SIZES[split])
            stream.seek(info.header_offset)
            local = io.BytesIO(stream.read(info.compress_size + 1024))
            fields = struct.unpack('<4s5H3I2H', local.read(30))
            if fields[0] != b'PK\x03\x04' or fields[2] & 1:
                raise ValueError('Invalid ZIP member')
            local.seek(fields[-2] + fields[-1], 1)
            data = local.read(info.compress_size)
            if info.compress_type == zipfile.ZIP_DEFLATED:
                data = zlib.decompress(data, -15)
            elif info.compress_type != zipfile.ZIP_STORED:
                raise ValueError('Unknown ZIP compression')
            if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
                raise ValueError('ZIP integrity check failed')
            return data
        except (TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)


def parse_ini(data):
    ini = configparser.ConfigParser(interpolation=None)
    ini.read_string(data.decode('utf-8-sig'))
    return ini['Sequence']


def catalog(split):
    url = f'{URL_ROOT}/{split}.zip'
    with zipfile.ZipFile(RemoteZip(url, SIZES[split])) as archive:
        index = {i.filename: i for i in archive.infolist()}
    files = sorted(n for n in index if n.endswith('/gameinfo.ini'))
    def read(name):
        ini = parse_ini(load_entry(split, index[name]))
        ball_ids = {int(k.removeprefix('trackletid_')) for k,v in ini.items()
                    if k.startswith('trackletid_') and v.split(';')[0].strip().lower() == 'ball'}
        return dict(prefix=name.removesuffix('/gameinfo.ini'), name=ini['name'], game=ini['gameID'],
                    action=ini.get('actionClass', ''), ball_ids=ball_ids, official_split=split)
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        rows = [r for r in pool.map(read, files) if r['ball_ids']]
    print(split, len(rows), 'ball sequences; game IDs:', sorted({r['game'] for r in rows}), flush=True)
    return index, rows
