import argparse
import json
from python_src.models import token_level_classifier, cross_encoder
from python_src.models.model_utils import TextSegmentationModels, grouped


class TrainingArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--config", default=None, type=str, required=False,
            help="The path to a configuration json file."
        )

        parser.add_argument(
            "--model_type", default=None, type=str, required=True,
            choices=list(TextSegmentationModels._member_map_.keys()),
            help=f"Model type to use to train and evaluate. "
                 f"It can be one of the following: {TextSegmentationModels._member_map_.keys()}"
        )

        self.parser = parser

    def parse(self):
        args, other = self.parser.parse_known_args()
        return args, other


def main():
    arguments, other_arguments = TrainingArgsParser().parse()

    configuration = arguments.config
    model_type = arguments.model_type

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
            arg_name = arg_name.replace("--", "")  # remove argument flag special tokens
            try:
                configuration[arg_name] = json.loads(value)  # use this to load the correct type
            except json.decoder.JSONDecodeError:
                print(f"Not able to load {arg_name} with value {value}. Using str representation instead. "
                      f"If it is already a str ignore this warning.")
                print()
                configuration[arg_name] = value

    if model_type in [TextSegmentationModels.token_level.value, TextSegmentationModels.t5_encoder_token_level.value,
                      TextSegmentationModels.t5_encoder_decoder_token_level.value]:
        token_level_classifier.train_eval_model(configuration)
    elif model_type == TextSegmentationModels.cross_encoder.value:
        cross_encoder.train_eval_model(configuration)
    else:
        raise ValueError(f"Model type to use to train and evaluate. "
                         f"must be one of the following: {TextSegmentationModels._member_map_.keys()}. "
                         f"Received {model_type}."
                         f"If using heuristic or text_tiling please use test_model.py script instead.")


if __name__ == '__main__':
    main()


# example commands
# python train_eval.py --model_type token_level --config ./run_configurations/token_level.json
