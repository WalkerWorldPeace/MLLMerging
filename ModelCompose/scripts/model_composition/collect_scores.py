import os
import re
import argparse
from pathlib import Path

answer_dir = '../../playground/data/eval/answers/'

def parse_scores(score_path: Path, name):
    if name in ['MUSIC-AVQA', 'MSRVTT-QA', 'seed_bench']:
        score = 'None'
        all_scores = dict()
        for fn in os.listdir(score_path):
            if str(score_path / fn).endswith('.txt'):
                try:
                    score_line = open(score_path / fn).read().strip().split('\n')[-1]
                    score_matched = re.match('.*Accuracy: (\d+\.\d+)%', score_line)
                    if score_matched:
                        score = score_matched.group(1)
                    score_matched2 = re.match('.*accuracy: (\d+\.\d+)%', score_line)
                    if not score_matched and score_matched2:
                        score = score_matched2.group(1)
                    all_scores[fn.replace('score_', '')[:-4]] = score
                except:
                    pass
        if len(all_scores) == 1:
            score = list(all_scores.items())[0]
        else:
            score = ' '.join([f'{v}({k})' for k,v in all_scores.items()])
    if name == 'MME':
        score = 'None'
        for fn in os.listdir(score_path):
            if str(score_path / fn).endswith('.txt'):
                try:
                    lines = open(score_path / fn).readlines()
                    for line in lines:
                        if "Perception ==========" in line:
                            perception_score = float(lines[lines.index(line) + 1].split(': ')[1])
                        elif "Cognition ==========" in line:
                            cognition_score = float(lines[lines.index(line) + 1].split(': ')[1])
                    score = f"{perception_score}+{cognition_score} = {perception_score+cognition_score}"
                except:
                    pass
    if name in ['VATEX', 'VALOR']:
        # print(score_path, )
        # if "multimodal-vicuna-7b-v1.5-vision+audio-beats-qformer-lr2e-5-ties-sum-locallora-v2-same" in str(score_path):
        #     print("hi")
        #     exit(0)
        score = 'None'
        for fn in os.listdir(score_path):
            if 'audio' in score_path.parts[-2] and 'vision' in score_path.parts[-2]:
                # print(fn)
                if 'v3' not in fn:
                    continue
            else: # UNI-MODAL
                if 'v3' not in fn:
                    continue
            if str(score_path / fn).endswith('.txt'):
                try:
                    # print(fn)
                    score_lines = open(score_path / fn).read().strip().split('\n')
                    for score_line in score_lines:
                        score_matched = re.match('.*CIDEr (\d+\.\d+)', score_line)
                        if score_matched:
                            score = round(100*float(score_matched.group(1)), 2)
                except:
                    pass
    return score

def collect_scores(names):
    all_scores = dict()
    for name in names:
        scores = dict()
        for dir in os.listdir(answer_dir):
            score_path = Path(answer_dir) / dir / name
            if os.path.exists(score_path):
                scores[dir] = parse_scores(score_path, name)
        all_scores[name] = scores

        
        table_str = "Model Name | Score\n" + "-"*60 + "\n"
        sorted_keys = sorted(list(all_scores[name].keys()))
        for key in sorted_keys:
            value = all_scores[name][key]
            table_str += f"{key} | {value}\n"
        
        print(table_str)    
        
    # import ipdb; ipdb.set_trace()

def main():
    parser = argparse.ArgumentParser(description='Collect evaluation results')
    parser.add_argument('names', nargs='+', help='List of evaluation datasets/benchmarks')
    args = parser.parse_args()

    collect_scores(args.names)

if __name__ == '__main__':
    main()