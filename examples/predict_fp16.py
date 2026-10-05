"""Run the three-part FP16 Laya model on RK3576 and print the complete API result."""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('state', help='Chinese instruction or other input text')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    sys.path.insert(0, str(root / 'bucket-study'))
    import torch
    from laya import Agent
    from api import Buckets

    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    agent = Agent('models/laya-multilingual', device='cpu', compile=False, fast=False)
    backend = Buckets(agent, 'split_fp16', flags=0)
    questions = {'route': {'type': 'choice', 'instructions': '判断语音指令的功能域',
        'criteria': {'日历安排': '日程与会议', '闹钟计时': '闹钟与倒计时',
                     '音量控制': '音量与静音', '音乐点播': '点歌与歌单',
                     '天气查询': '天气与气温', '交通出行': '导航与公交'}}}
    try:
        result = agent.predict(args.state, questions)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
