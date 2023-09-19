import argparse
from python_src.models.model_utils import TextSegmentationModels, grouped
import json
from python_src.models import token_level_classifier, text_tiling, heuristic_baselines, cross_encoder


class TestingArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--config", default=None, type=str, required=False,
            help="The path to a configuration json file."
        )

        parser.add_argument(
            "--model_type", default=None, type=str, required=True,
            choices=list(TextSegmentationModels._member_map_.keys()),
            help=f"Model type to evaluate. "
                 f"It can be one of the following: {TextSegmentationModels._member_map_.keys()}"
        )

        parser.add_argument(
            "--model_location", default=None, type=str, required=False,
            help=f"Path to a folder containing a checkpoint of the model."
        )

        parser.add_argument(
            "--eval_number_samples", default=None, type=int, required=False,
            help=f"Number of samples to evaluate."
        )

        parser.add_argument(
            "--test_data_location", default="./data/test.tsv", type=str, required=False,
            help=f"Path to dataset to test."
        )

        parser.add_argument(
            "--suffix", default="_eval", type=str, required=False,
            help=f"Suffix used to differentiate between the original predictions and the ones at the moment."
        )

        self.parser = parser

    def parse(self):
        args, other = self.parser.parse_known_args()
        return args, other


def main():
    arguments, other_arguments = TestingArgsParser().parse()

    configuration = arguments.config
    model_type = arguments.model_type
    model_location = arguments.model_location

    if configuration is None:
        configuration = {}
    elif isinstance(configuration, str):
        with open(configuration) as f_open:
            configuration = json.load(f_open)
    else:
        raise ValueError(f"Format of the configuration provided: {type(configuration)} is not supported")

    # command line arguments are put over the configuration arguments
    # make sure name matches in the configuration (case sensitive) and values are able to be loaded by json.loads
    if other_arguments:
        for arg_name, value in grouped(other_arguments, 2):
            arg_name = arg_name.replace("--", "")
            try:
                configuration[arg_name] = json.loads(value)  # use this to load the correct type
            except json.decoder.JSONDecodeError:
                print(f"Not able to load {arg_name} with value {value}. Using str representation instead. "
                      f"If it is already a str ignore this warning.")
                print()
                configuration[arg_name] = value

    if model_type in [TextSegmentationModels.token_level.value, TextSegmentationModels.t5_encoder_token_level.value,
                      TextSegmentationModels.t5_encoder_decoder_token_level.value]:
        token_level_classifier.evaluate_model(configuration, model_location, arguments.eval_number_samples,
                                              arguments.test_data_location, arguments.suffix)
    elif model_type == TextSegmentationModels.cross_encoder.value:
        cross_encoder.evaluate_model(configuration, model_location, arguments.eval_number_samples,
                                     arguments.test_data_location, arguments.suffix)
    elif model_type == TextSegmentationModels.text_tiling.value:
        text_tiling.evaluate_model(configuration, arguments.eval_number_samples, arguments.test_data_location,
                                   arguments.suffix)
    elif model_type == TextSegmentationModels.heuristic.value:
        heuristic_baselines.evaluate_model(configuration, arguments.eval_number_samples, arguments.test_data_location,
                                           arguments.suffix)
    else:
        raise ValueError(f"Model type to use to train and evaluate. "
                         f"must be one of the following: {TextSegmentationModels._member_map_.keys()}. "
                         f"Received {model_type}.")


if __name__ == "__main__":
    main()


'''
# other baselines
python3 test_model.py --model_type heuristic --test_data_location "./data/test.tsv"

python3 test_model.py --model_type text_tiling --test_data_location "./data/test.tsv"
'''


'''
# example commands
python3 test_model.py --model_type token_level --config <config_location> --model_location <model_location>
'''
