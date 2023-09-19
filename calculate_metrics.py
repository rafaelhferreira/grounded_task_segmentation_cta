import json
from python_src.metrics import evaluate_over_dataset
import argparse
from python_src.models.model_utils import extract_checkpoint_name


class CalculateMetricsArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--predictions", type=str, required=True,
            help=f"Path to .tsv file with the model predictions for a split."
        )

        parser.add_argument(
            "--dataset_split", type=str, required=False, default="./data/test.tsv",
            help=f"Path to .tsv file with a dataset split."
        )

        parser.add_argument(
            "--output_file", type=str, required=False, default=None,
            help=f"Path to .json file to save the metrics results."
        )

        parser.add_argument(
            "--model_name", type=str, required=False, default=None,
            help=f"Name of the model. This is just to save the name of the model to the results output_file."
        )

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        return args


def main():
    arguments = CalculateMetricsArgsParser().parse()

    if arguments.model_name:
        checkpoint_name = extract_checkpoint_name(arguments.model_name)
    else:
        checkpoint_name = None

    metrics = evaluate_over_dataset(split_file_path=arguments.predictions, model_output_file=arguments.dataset_split,
                                    number_samples=None, output_file=arguments.output_file,
                                    model_name=arguments.model_name, checkpoint=checkpoint_name)

    print("Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()

'''
# example commands
python3 calculate_metrics.py --predictions ./results/bert-base-uncased-token/bert-base-uncased_test_predictions.tsv --output_file ./results/bert-base-uncased-token/token_level_results.json
'''
