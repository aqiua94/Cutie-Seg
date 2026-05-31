import argparse
import csv
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def query_vram() -> str:
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            text=True,
        )
        return out.strip().splitlines()[0]
    except Exception as exc:
        return f'error:{exc}'


def query_ps(pattern: str) -> list[dict[str, str]]:
    out = subprocess.check_output(['pgrep', '-af', pattern], text=True)
    rows = []
    for line in out.splitlines():
        if 'monitor_training_resources.py' in line:
            continue
        pid = line.split(maxsplit=1)[0]
        try:
            ps = subprocess.check_output(
                ['ps', '-o', 'pid=,ppid=,rss=,vsz=,stat=,cmd=', '-p', pid],
                text=True,
            ).strip()
        except subprocess.CalledProcessError:
            continue
        parts = ps.split(maxsplit=5)
        if len(parts) < 6:
            continue
        rows.append({
            'pid': parts[0],
            'ppid': parts[1],
            'rss_kb': parts[2],
            'vsz_kb': parts[3],
            'stat': parts[4],
            'cmd': parts[5],
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('logs/creff_s480_l8_resource_trace.csv'))
    parser.add_argument('--pattern', default='train_compressed_davis_dataloader|resume_creff_train_until')
    parser.add_argument('--interval', type=float, default=5.0)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open('a', newline='') as f:
        fieldnames = ['time_utc', 'vram_mb', 'pid', 'ppid', 'rss_kb', 'vsz_kb', 'stat', 'cmd']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        while True:
            now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            vram = query_vram()
            rows = query_ps(args.pattern)
            if not rows:
                writer.writerow({
                    'time_utc': now,
                    'vram_mb': vram,
                    'pid': '',
                    'ppid': '',
                    'rss_kb': '',
                    'vsz_kb': '',
                    'stat': '',
                    'cmd': 'no matching process',
                })
            for row in rows:
                writer.writerow({'time_utc': now, 'vram_mb': vram, **row})
            f.flush()
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
