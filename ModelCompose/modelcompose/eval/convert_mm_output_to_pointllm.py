import json
import argparse

def main(input_file, ann_file, output_file, prompt):
    data_list = []
    
    anns = json.load(open(ann_file))
    if 'label_name' not in anns[0]:
        # Objaverse
        anns = {x['object_id']: x['conversations'][1]['value'] for x in json.load(open(ann_file))}

        results = {
            'prompt': prompt
        }

        with open(input_file, 'r') as f:
            for line in f:
                line_data = json.loads(line.strip())  # Parse each line as JSON
                question_id = line_data.get("question_id", "")
                text = line_data.get("text", "")
                
                new_entry = {
                    "object_id": question_id,
                    "ground_truth": anns[question_id],
                    "model_output": text
                }

                data_list.append(new_entry)

        results["results"] = data_list
        # Write the data_list to the output JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Conversion completed. The output has been saved to {output_file}")
        
    else:
        # modelnet40
        anns = {x['id']: x for x in json.load(open(ann_file))}

        results = {
            'prompt': prompt
        }

        with open(input_file, 'r') as f:
            for line in f:
                line_data = json.loads(line.strip())  # Parse each line as JSON
                question_id = line_data.get("question_id", "")
                text = line_data.get("text", "")
                
                new_entry = {
                    "object_id": question_id,
                    "ground_truth": anns[question_id]['label'],
                    "model_output": text,
                    "label_name": anns[question_id]['label_name']
                }

                data_list.append(new_entry)

        results["results"] = data_list
        # Write the data_list to the output JSON file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Conversion completed. The output has been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL input file to JSON output file")
    parser.add_argument("--input_file", type=str, help="Input JSONL file path")
    parser.add_argument("--ann_file", type=str, help="Annotation JSON file path")
    parser.add_argument("--output_file", type=str, help="Output JSON file path")
    parser.add_argument("--prompt", type=str, help="prompt")
    args = parser.parse_args()

    if args.input_file and args.output_file:
        main(args.input_file, args.ann_file, args.output_file, args.prompt)
    else:
        print("Please provide both input and output file paths.")
