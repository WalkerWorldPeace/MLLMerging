import json
import argparse

from m4c_evaluator import EvalAIAnswerProcessor

def calculate_accuracy(answers_file, output_file):
    processor = EvalAIAnswerProcessor()

    # Load data from the JSON files
    with open(answers_file, 'r') as answers_file:
        answers_data = json.load(answers_file)

    with open(output_file, 'r') as output_file:
        output_data = [json.loads(line) for line in output_file]

    # Create a dictionary to store answers by ID for easier access
    answers_by_id = {answer['id']: answer['conversations'][-1]['value'] for answer in answers_data}

    # Match question IDs and calculate accuracy
    correct_count = 0
    total_questions = len(output_data)

    for output_item in output_data:
        question_id = output_item['question_id']
        if question_id in answers_by_id:
            expected_answer = answers_by_id[question_id]
            output_answer = output_item['text']
            processed_ref = processor(expected_answer)
            processed_out = processor(output_answer)
            print(processed_ref, processed_out)
            if processed_ref == processed_out:
                correct_count += 1

    accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0

    print(f"Accuracy: {accuracy:.2f}%")

def find_mismatched_questions(answers_file, x, y, z):
    # Load data from the JSON files
    with open(answers_file, 'r') as answers_file:
        answers_data = json.load(answers_file)

    with open(x, 'r') as x_file:
        x_data = [json.loads(line) for line in x_file]

    with open(y, 'r') as y_file:
        y_data = [json.loads(line) for line in y_file]

    with open(z, 'r') as z_file:
        z_data = [json.loads(line) for line in z_file]

    # Create a dictionary to store answers by ID for easier access
    answers_by_id = {answer['id']: (answer['conversations'][0]['value'], answer['conversations'][-1]['value']) for answer in answers_data}

    mismatched_questions = []

    for x_item, y_item, z_item in zip(x_data, y_data, z_data):
        question_id = x_item['question_id']
        if question_id in answers_by_id:
            question, expected_answer = answers_by_id[question_id]
            x_answer = x_item['text']
            y_answer = y_item['text']
            z_answer = z_item['text']

            if expected_answer.lower() not in x_answer.lower() and \
               expected_answer.lower() not in y_answer.lower() and \
               expected_answer.lower() in z_answer.lower():
                mismatched_questions.append({'question': question, 'expected_answer': expected_answer,
                                             'x': x_answer, 'y': y_answer, 'z': z_answer})

    return mismatched_questions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate accuracy based on answers and output.')
    parser.add_argument('--answers', required=True, help='Path to answers.json')
    parser.add_argument('--output', required=False, help='Path to output.jsonl')
    parser.add_argument('--x', required=False, help='Path to output X.jsonl')
    parser.add_argument('--y', required=False, help='Path to output Y.jsonl')
    parser.add_argument('--z', required=False, help='Path to output Z.jsonl')
    args = parser.parse_args()

    if args.output is not None:
        calculate_accuracy(args.answers, args.output)
    else:
        mismatched_questions = find_mismatched_questions(args.answers, args.x, args.y, args.z)
        print("Mismatched Questions:")
        for question in mismatched_questions:
            print(f"[Question]: {question['question']}")
            print(f"[Expected Answer]: {question['expected_answer']}")
            print(f"[Vision Answer]: {question['x']}")
            print(f"[Audio  Answer]: {question['y']}")
            print(f"[Merged Answer]: {question['z']}")
            print()