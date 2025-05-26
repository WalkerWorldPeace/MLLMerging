from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import argparse

def main(annotation_file, result_file, score_file):
    coco = COCO(annotation_file)
    cocoRes = coco.loadRes(result_file)

    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.evaluate()

    print("---Results---")
    with open(score_file, 'w') as f:
        for metric, score in cocoEval.eval.items():
            print(metric, score, file=f)
            print(metric, score)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COCO results and save scores to a file")
    parser.add_argument("--annotation_file", type=str, help="COCO annotation file path")
    parser.add_argument("--result_file", type=str, help="COCO result file path")
    parser.add_argument("--score_file", type=str, help="Output score file path")
    args = parser.parse_args()

    if args.annotation_file and args.result_file and args.score_file:
        main(args.annotation_file, args.result_file, args.score_file)
    else:
        print("Please provide annotation_file, result_file, and score_file paths.")
