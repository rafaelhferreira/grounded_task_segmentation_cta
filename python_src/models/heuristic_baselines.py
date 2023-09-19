import os
import random
from tqdm import tqdm
from python_src.dataset.utils import load_dataset_split, get_spacy_nlp, write_dict_to_json_file, \
    write_configuration_to_file, class_attributes_to_dict, write_split_to_file
from python_src.metrics import evaluate_over_dataset
from typing import List, Union, Dict
from python_src.models.model_utils import TextSegmentationModels, set_seeds


class HeuristicConfiguration:

    def __init__(self, configuration: Dict):
        self.text_segmentation_model_type = configuration.get("text_segmentation_model_type",
                                                              TextSegmentationModels.heuristic.value)
        self.model_name = configuration.get("model_name", "spacy")

        # break after x sentences
        self.deterministic_sentence_breaks = configuration.get("deterministic_sentence_breaks", [1, 2, 3])
        self.use_small_dataset = configuration.get("use_small_dataset", False)

        # probability of breaking randomly
        self.random_break_probabilities = configuration.get("random_break_probabilities", [0.25, 0.5, 0.75, 1.0])

        self.output_dir = configuration.get("output_dir", './model_results/heuristics')  # output directory

        self.logging_dir = configuration.get("logging_dir", './logs')  # directory for storing logs
        self.logging_steps = configuration.get("logging_steps", 50 if self.use_small_dataset else 100)
        self.seed = configuration.get("seed", 42)
        self.train_run_name = configuration.get("train_run_name", self.model_name)
        self.test_run_name = configuration.get("test_run_name", self.model_name + "_test")
        self.train_number_samples = configuration.get("train_number_samples", 1000 if self.use_small_dataset else None)
        self.eval_number_samples = configuration.get("eval_number_samples", 100 if self.use_small_dataset else None)

        # datasets
        self.train_data_location = configuration.get("train_data_location", "./data/train.tsv")
        self.valid_data_location = configuration.get("valid_data_location", "./data/valid.tsv")
        self.test_data_location = configuration.get("test_data_location", "./data/test.tsv")


def random_sentence_break(prob_to_break: float, dataset_sentences: List[List[str]], random_seed: Union[None, int],
                          ids_list: List[int], titles_list: List[str],
                          original_examples_list: List[List[str]], examples_list: List[str],
                          output_folder: str, file_name: str):
    if random_seed is not None:
        random.seed(random_seed)
    model_output_labels = []
    for example in dataset_sentences:
        current_offset = 0
        model_output_labels.append([])  # append an empty list
        for sentence in example[:-1]:  # do not count with the last one because it is trivial to break
            current_offset += len(sentence)
            prob = random.random()
            if prob <= prob_to_break:  # break
                model_output_labels[-1].append(current_offset)
            current_offset += 1  # sum 1 because of the space between sentences

    # write to file
    # original_examples_list and examples_list do not change
    # add the labels predicted by the model
    write_split_to_file(ids_list, titles_list, original_examples_list, examples_list, model_output_labels,
                        output_folder, file_name)


def sentence_level_break(break_len: int, dataset_sentences: List[List[str]],
                         ids_list: List[int], title_list: List[str],
                         original_examples_list: List[List[str]], examples_list: List[str],
                         output_folder: str, file_name: str):
    model_output_labels = []
    for example in dataset_sentences:
        current_offset = 0
        model_output_labels.append([])  # append an empty list
        for i, sentence in enumerate(example[:-1]):  # do not count with the last one because it is trivial to break
            current_offset += len(sentence)
            if i % break_len == 0:
                if i == 0 and break_len != 1:  # only break the first time if the break_len is 1
                    pass
                else:  # break
                    model_output_labels[-1].append(current_offset)
            current_offset += 1  # sum 1 because of the space between sentences

    # write to file
    # original_examples_list and examples_list do not change
    # add the labels predicted by the model
    write_split_to_file(ids_list, title_list, original_examples_list, examples_list, model_output_labels,
                        output_folder, file_name)


def print_and_write_official_metrics(test_data_path: str, run_name: str, output_dir: str,
                                     number_samples: Union[int, None] = None):
    # function that writes results to file
    official_metrics = evaluate_over_dataset(test_data_path, os.path.join(output_dir, run_name + ".tsv"),
                                             number_samples)
    # add the model name to the results dict
    official_metrics["model_name"] = run_name

    print(f"OFFICIAL Test Results {run_name}:")
    print(official_metrics)
    print()

    # write the result to the file
    write_dict_to_json_file(data=official_metrics,
                            output_path=os.path.join(output_dir, run_name + "_results.json"))


def evaluate_model(configuration: Dict, eval_number_samples: Union[int, None],
                   test_data_location: Union[str, None], suffix: str = "eval"):

    # get the model configuration
    model_configuration = HeuristicConfiguration(configuration)

    # set the attributes passed in the model configuration
    model_configuration.eval_number_samples = eval_number_samples
    model_configuration.test_data_location = test_data_location

    print("Model Configuration:")
    print(class_attributes_to_dict(model_configuration))
    print()

    # set seeds for reproducibility
    set_seeds(model_configuration.seed)

    # load the data
    # no need to apply these methods to the train and validation set
    '''
    _, _, original_train_examples, train_examples_list, train_labels_list = load_dataset_split(
        model_configuration.train_data_location,
        model_configuration.train_number_samples
    )
    _, _, original_valid_examples, valid_examples_list, valid_labels_list = load_dataset_split(
        model_configuration.valid_data_location,
        model_configuration.eval_number_samples
    )
    '''

    test_ids_list, test_titles_list, original_test_examples, test_examples_list, test_labels_list = load_dataset_split(
        model_configuration.test_data_location,
        model_configuration.eval_number_samples
    )

    dataset_sentences = []  # type: List[List[str]]

    # turn the entire dataset into individual sentences per recipe
    print("Running spacy over examples...")
    spacy_nlp = get_spacy_nlp()
    for example in tqdm(test_examples_list[:model_configuration.eval_number_samples]):
        current_sentences = spacy_nlp(example).sents  # get spacy sentences
        current_sentences = [i.text.strip() for i in current_sentences]  # convert to str
        dataset_sentences.append(current_sentences)  # add to list

    # apply the random method to the dataset
    for prob in model_configuration.random_break_probabilities:
        if model_configuration.seed != 42:
            suffix = f"_seed_{model_configuration.seed}_{suffix}"
        run_name = f"random_{prob}_{model_configuration.test_run_name}{suffix}_predictions"
        random_sentence_break(prob_to_break=prob, dataset_sentences=dataset_sentences,
                              random_seed=model_configuration.seed,
                              ids_list=test_ids_list, titles_list=test_titles_list,
                              original_examples_list=original_test_examples, examples_list=test_examples_list,
                              output_folder=model_configuration.output_dir,
                              file_name=run_name)
        # print and write results to file
        print_and_write_official_metrics(model_configuration.test_data_location, run_name,
                                         model_configuration.output_dir, model_configuration.eval_number_samples)

    # apply the deterministic method to the dataset
    for value in model_configuration.deterministic_sentence_breaks:
        run_name = f"sentence_break_{value}_{model_configuration.test_run_name}{suffix}_predictions"
        sentence_level_break(break_len=value, dataset_sentences=dataset_sentences,
                             ids_list=test_ids_list, title_list=test_titles_list,
                             original_examples_list=original_test_examples, examples_list=test_examples_list,
                             output_folder=model_configuration.output_dir,
                             file_name=run_name)

        # print and write results to file
        print_and_write_official_metrics(model_configuration.test_data_location, run_name,
                                         model_configuration.output_dir, model_configuration.eval_number_samples)

    write_configuration_to_file(config_class=model_configuration,
                                output_path=os.path.join(model_configuration.output_dir,
                                                         model_configuration.test_run_name + suffix + "_config.json"))
