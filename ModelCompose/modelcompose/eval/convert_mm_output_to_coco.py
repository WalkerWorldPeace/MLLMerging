import json
import argparse

def main(input_file, output_file):
    data_list = []

    with open(input_file, 'r') as f:
        for line in f:
            line_data = json.loads(line.strip())  # Parse each line as JSON
            question_id = line_data.get("question_id", "")
            caption = line_data.get("text", "")

            # Create a dictionary in the desired format
            new_entry = {
                "image_id": question_id,
                "id": len(data_list),  # Use the current length of the list as the ID
                "caption": caption
            }

            data_list.append(new_entry)

    # Write the data_list to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(data_list, f, indent=4)

    print(f"Conversion completed. The output has been saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL input file to JSON output file")
    parser.add_argument("--input_file", type=str, help="Input JSONL file path")
    parser.add_argument("--output_file", type=str, help="Output JSON file path")
    args = parser.parse_args()

    if args.input_file and args.output_file:
        main(args.input_file, args.output_file)
    else:
        print("Please provide both input and output file paths.")
