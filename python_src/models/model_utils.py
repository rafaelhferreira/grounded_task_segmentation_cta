from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from nltk import pk
import re
import random
import torch
import os
from enum import Enum
from typing import Tuple, Dict, List, Union, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import IntervalStrategy, TrainingArguments
from python_src.dataset.utils import write_configuration_to_file, write_split_to_file
from python_src.metrics import evaluate_over_dataset


class TextSegmentationModels(Enum):
    token_level = "token_level"
    t5_encoder_token_level = "t5_encoder_token_level"
    t5_encoder_decoder_token_level = "t5_encoder_decoder_token_level"
    cross_encoder = "cross_encoder"
    heuristic = "heuristic"
    text_tiling = "text_tiling"


class DefaultModelConfiguration:

    def __init__(self, configuration: Dict):
        # model and tokenizer
        self.model_type = configuration.get("model_type", TextSegmentationModels.token_level.value)
        self.suffix = configuration.get("suffix", "")  # way to distinguish between identical models
        self.model_name = configuration.get("model_name", "bert-base-uncased")
        self.tokenizer_name = configuration.get("tokenizer_name", "bert-base-uncased")
        self.text_segmentation_model_type = configuration.get("text_segmentation_model_type", None)
        self.do_lower_case = configuration.get("do_lower_case", True)
        self.use_small_dataset = configuration.get("use_small_dataset", False)

        # training and eval params
        self.num_train_epochs = configuration.get("num_train_epochs", 5)  # total number of training epochs
        # batch size per device during training
        self.per_device_train_batch_size = configuration.get("per_device_train_batch_size", 16)
        # batch size for evaluation
        self.per_device_eval_batch_size = configuration.get("per_device_eval_batch_size", 16)
        self.gradient_accumulation_steps = configuration.get("gradient_accumulation_steps", 1)
        self.fp16 = configuration.get("fp16", False)  # reduce size
        self.warmup_steps = configuration.get("warmup_steps", 100 if self.use_small_dataset else 1000)
        self.learning_rate = configuration.get("learning_rate", 5e-5)
        self.weight_decay = configuration.get("weight_decay", 0.01)  # strength of weight decay
        self.logging_dir = configuration.get("logging_dir", './logs')  # directory for storing logs
        self.logging_steps = configuration.get("logging_steps", 50 if self.use_small_dataset else 100)
        self.evaluation_strategy = configuration.get("evaluation_strategy", IntervalStrategy.EPOCH.value)
        self.save_strategy = configuration.get("save_strategy", IntervalStrategy.EPOCH.value)
        # self.save_steps = configuration.get("save_steps",200 if self.use_small_dataset else 2000)
        self.save_total_limit = configuration.get("save_total_limit", 1)
        self.no_cuda = configuration.get("no_cuda", False)
        self.seed = configuration.get("seed", 42)
        self.train_run_name = configuration.get("train_run_name", self.model_name.replace("/", "-")) + self.suffix
        self.test_run_name = configuration.get("test_run_name", self.model_name.replace("/", "-") + "_test") + self.suffix
        self.metric_for_best_model = configuration.get("metric_for_best_model", "f1_batch_value")
        self.greater_is_better = configuration.get("greater_is_better", True)
        self.train_number_samples = configuration.get("train_number_samples", 200 if self.use_small_dataset else None)
        self.eval_number_samples = configuration.get("eval_number_samples", 50 if self.use_small_dataset else None)
        self.label_names = ["labels"]

        # datasets
        self.train_data_location = configuration.get("train_data_location", "./data/train.tsv")
        self.valid_data_location = configuration.get("valid_data_location", "./data/valid.tsv")
        self.test_data_location = configuration.get("test_data_location", "./data/test.tsv")

        # for the recipes dataset use True else False
        self.spacy_segmentation = configuration.get("spacy_segmentation", True)

        # output directory
        self.output_dir = configuration.get("output_dir", f'./model_results/'
                                                          f'{self.train_run_name}_{self.learning_rate}_'
                                                          f'{self.per_device_train_batch_size}')

    def get_trainer_training_arguments(self) -> TrainingArguments:
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            label_names=self.label_names,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=self.save_strategy,
            # save_steps=training_configuration.save_steps,
            save_total_limit=self.save_total_limit,
            no_cuda=self.no_cuda,
            seed=self.seed,
            run_name=self.train_run_name,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
            fp16=self.fp16,
            report_to="none"  # use "all" or "none"
        )
        return training_args

    def get_trainer_testing_arguments(self) -> TrainingArguments:
        test_arguments = TrainingArguments(
            output_dir=self.output_dir,  # output directory
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            label_names=self.label_names,
            # batch size for evaluation
            logging_dir=self.logging_dir,  # directory for storing logs
            logging_steps=self.logging_steps,
            evaluation_strategy=self.evaluation_strategy,
            save_strategy=IntervalStrategy.NO.value,
            no_cuda=False,
            seed=self.seed,
            run_name=self.test_run_name,
            report_to="none"  # use "all" or "none"
        )
        return test_arguments


# calculate metrics accuracy, precision, recall, f1 and pk at a token level
def compute_metrics_token_level(eval_pred) -> Dict[str, float]:
    label_to_ignore = -100
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    final_prediction_list, final_label_list = [], []
    accuracy_batch, precision_batch, recall_batch, f1_batch, pk_batch = [], [], [], [], []
    number_labels_list = []  # for weight purposes
    for pred_batch, label_batch in zip(predictions, labels):  # iterate over the batch
        final_prediction_list.append([])
        final_label_list.append([])
        for pred, label in zip(pred_batch, label_batch):  # iterate over the list in each batch
            # if label is not to ignore add to list
            if label != label_to_ignore:
                final_prediction_list[-1].append(pred)
                final_label_list[-1].append(label)

        number_labels_list.append(len(final_label_list))

        # perfom the calculation over a single batch
        accuracy_example = accuracy_score(final_prediction_list[-1], final_label_list[-1])
        precision_example = precision_score(final_prediction_list[-1], final_label_list[-1])

        if any(final_label_list[-1]):  # if there is at least one non-zero
            recall_example = recall_score(final_prediction_list[-1], final_label_list[-1])
        else:  # all labels are zeros
            if any(final_prediction_list[-1]):  # there is at least one element in the predictions
                recall_example = 0.0
            else:
                recall_example = recall_score(final_prediction_list[-1], final_label_list[-1], zero_division=1)
        f1_example = f1_score(final_prediction_list[-1], final_label_list[-1])

        # turn into str to calculate pk (approximate pk since it is tokenized)
        try:
            pk_example = pk(ref="".join([str(i) for i in final_label_list[-1]]),
                            hyp="".join([str(i) for i in final_prediction_list[-1]]))
        except ZeroDivisionError:  # when model does not predict a single break put max value
            pk_example = 1.0

        accuracy_batch.append(accuracy_example)
        precision_batch.append(precision_example)
        recall_batch.append(recall_example)
        f1_batch.append(f1_example)
        pk_batch.append(pk_example)

    # mean considering each batch independently
    accuracy_batch_value = np.mean(accuracy_batch)
    precision_batch_value = np.mean(precision_batch)
    recall_batch_value = np.mean(recall_batch)
    f1_batch_value = np.mean(f1_batch)
    pk_batch_value = np.mean(pk_batch)

    # mean weighted by the size of the sequence in each batch
    weighted_accuracy, weighted_precision, weighted_recall, weighted_f1, weighted_pk = 0, 0, 0, 0, 0
    for i, number_labels in enumerate(number_labels_list):
        weighted_accuracy += number_labels * accuracy_batch[i]
        weighted_precision += number_labels * precision_batch[i]
        weighted_recall += number_labels * recall_batch[i]
        weighted_f1 += number_labels * f1_batch[i]
        weighted_pk += number_labels * pk_batch[i]

    sum_of_labels = sum(number_labels_list)
    if sum_of_labels:
        weighted_accuracy = weighted_accuracy / sum_of_labels
        weighted_precision = weighted_precision / sum_of_labels
        weighted_recall = weighted_recall / sum_of_labels
        weighted_f1 = weighted_f1 / sum_of_labels
        weighted_pk = weighted_pk / sum_of_labels

    return {
        "accuracy_batch_value": accuracy_batch_value,
        "precision_batch_value": precision_batch_value,
        "recall_batch_value": recall_batch_value,
        "f1_batch_value": f1_batch_value,
        "pk_batch_value": pk_batch_value,
        "weighted_accuracy": weighted_accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "weighted_pk": weighted_pk,
    }


def compute_metrics_cross_segment(eval_pred) -> Dict[str, float]:
    # compute metrics when there is only one label per input
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(predictions, labels)
    # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
    precision_micro = precision_score(predictions, labels, average='micro')
    recall_micro = recall_score(predictions, labels, average='micro')
    f1_micro = f1_score(predictions, labels, average='micro')

    precision_macro = precision_score(predictions, labels, average='macro')
    recall_macro = recall_score(predictions, labels, average='macro')
    f1_macro = f1_score(predictions, labels, average='macro')

    return {
        "accuracy": accuracy,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }


def write_results_to_file_and_officially_evaluate_model(model_configuration: DefaultModelConfiguration,
                                                        test_ids_list: List[int], test_titles_list: List[str],
                                                        original_test_examples: List[List[str]],
                                                        test_examples_list: List[str],
                                                        model_output_labels: List[List[int]],
                                                        model_name_or_path: Union[str, None] = None,
                                                        checkpoint: Union[str, None] = None,
                                                        suffix: str = "") -> Dict:

    if checkpoint:
        checkpoint = "_" + checkpoint
    else:
        checkpoint = ""

    # write to file the model predictions
    predictions_file_path = model_configuration.test_run_name + checkpoint + suffix + "_predictions"
    write_split_to_file(ids_list=test_ids_list, titles_list=test_titles_list,
                        original_steps_list=original_test_examples, examples_list=test_examples_list,
                        labels_list=model_output_labels, out_file_path=model_configuration.output_dir,
                        out_file_name=predictions_file_path)
    print(f"Model predictions written to file: {os.path.join(model_configuration.output_dir, predictions_file_path)}")
    print()

    # write the configuration used to a file for later use
    model_configuration_file_path = os.path.join(model_configuration.output_dir,
                                                 model_configuration.test_run_name + suffix + "_config.json")
    model_config_dict = write_configuration_to_file(config_class=model_configuration,
                                                    output_path=model_configuration_file_path)
    print(f"Model configuration written to file: {model_configuration_file_path}")
    print(model_config_dict)
    print()

    # evaluate using the official script given the file written before
    official_metrics_file_path = os.path.join(model_configuration.output_dir,
                                              model_configuration.test_run_name + checkpoint + suffix + "_results.json")
    official_metrics = evaluate_over_dataset(split_file_path=model_configuration.test_data_location,
                                             model_output_file=os.path.join(model_configuration.output_dir,
                                                                            predictions_file_path + ".tsv"),
                                             number_samples=model_configuration.eval_number_samples,
                                             output_file=official_metrics_file_path,
                                             model_name=model_name_or_path,
                                             checkpoint=checkpoint)
    print(f"Model official metrics written to file: {official_metrics_file_path}")
    print("OFFICIAL Test Results:")
    print(official_metrics)
    print()

    # create the final dict that holds the configuration and the results
    model_config_dict.update(official_metrics)
    return model_config_dict


def set_seeds(seed: int):
    # set all seeds for reproducibility
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def extract_checkpoint_name(text: str) -> str:
    # extract the name of the checkpoint given a str
    matches = re.findall(r"checkpoint-\d\d*", text)
    if matches:
        return matches[-1]  # if matches return the last match
    else:
        return ""  # if no matches return empty str


@dataclass
class TokenClassifierCustomOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)


def add_suffix_to_metrics(suffix: str, current_metrics: Dict, considered_metrics: List[str] = None):
    if considered_metrics is None:
        considered_metrics = ["split_file_path", "number_samples",
                              # main metrics
                              "pk", "pk_word_level", "precision", "recall", "f1",
                              # other stats
                              "perfect_matches", "perfect_matches_perc", "same_number_breaks", "same_number_breaks_per",
                              "one_of_difference", "one_of_difference_per", "more_breaks", "more_breaks_perc",
                              "less_breaks", "less_breaks_perc", "avg_number_breaks", "avg_number_model_breaks",
                              "avg_words_model_break", "avg_words_break"]

    new_dict = current_metrics.copy()
    for k in current_metrics.keys():
        if k in considered_metrics:
            new_dict[suffix + "_" + k] = new_dict.pop(k)

    return new_dict
