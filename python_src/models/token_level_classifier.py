from python_src.models.token_level_t5 import T5EncoderTokenClassification, T5TokenClassificationConfig, \
    T5EncoderDecoderTokenClassification
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
import os
from python_src.models.model_utils import compute_metrics_token_level, DefaultModelConfiguration, \
    write_results_to_file_and_officially_evaluate_model, TextSegmentationModels, set_seeds, extract_checkpoint_name, \
    add_suffix_to_metrics
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, PreTrainedModel, PreTrainedTokenizer, \
    PreTrainedTokenizerBase, AutoConfig, Trainer
from typing import List, Tuple, Dict, Union, Optional
from python_src.dataset.create_dataset import TokenLevelDataset, convert_char_label_to_token_labels, \
    convert_char_label_to_token_labels_given_segments
from python_src.dataset.utils import load_dataset_split, class_attributes_to_dict, get_spacy_nlp, get_segments, \
    write_dict_to_json_file


class TokenLevelClassifierConfiguration(DefaultModelConfiguration):

    def __init__(self, configuration: Dict):
        super().__init__(configuration)

        # overwritten attributes
        self.text_segmentation_model_type = configuration.get("text_segmentation_model_type",
                                                              TextSegmentationModels.token_level.value)

        self.metric_for_best_model = configuration.get("metric_for_best_model", "f1_batch_value")

        # token level specific
        # token must be one of these to be considered a break (used as a post-processing technique)
        self.only_consider_segments = configuration.get("only_consider_segments", False)
        self.break_tokens_list = configuration.get("break_tokens_list",
                                                   [".", "!", "?", "...", ";", ")", ":", "\n"])  # or None
        self.label_to_ignore = configuration.get("label_to_ignore", -100)

        self.label_names = ["labels"]

        self.train_run_name = configuration.get("train_run_name", self.model_name.replace("/", "-")) + self.suffix
        self.test_run_name = configuration.get("test_run_name",
                                               self.model_name.replace("/", "-") + "_test") + self.suffix

        # output directory
        self.output_dir = configuration.get("output_dir", self.__create_output_folder_name())

    def __create_output_folder_name(self):
        output_dir = f'./model_results/' \
                     f'{self.train_run_name}_{self.model_type}' \
                     f'_{self.learning_rate}_{self.per_device_train_batch_size}_token' \
                     f'{"_only_consider_segments" if self.only_consider_segments else ""}' \
                     f'{self.seed if self.seed != 42 else ""}'

        if output_dir.endswith("_"):  # remove trainling underscore
            output_dir = output_dir[:-1]

        return output_dir


def load_model_tokenizer(model_name_path: Union[str, PreTrainedModel], tokenizer_name_path: str, do_lower_case: bool,
                         model_configuration: TokenLevelClassifierConfiguration
                         ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_path, do_lower_case=do_lower_case)

    if isinstance(model_name_path, str):  # load if it is a str

        if model_configuration.model_type == TextSegmentationModels.token_level.value:
            model = AutoModelForTokenClassification.from_pretrained(model_name_path, num_labels=2)
        elif model_configuration.model_type == TextSegmentationModels.t5_encoder_token_level.value:
            # token classification using the encoder part of the model
            model_config = AutoConfig.from_pretrained(model_name_path).to_dict()
            # remove redundandant keys from model_config
            for k in class_attributes_to_dict(model_configuration).keys():
                if k in model_config:
                    model_config.pop(k)
            config = T5TokenClassificationConfig(**model_config)
            model = T5EncoderTokenClassification.from_pretrained(model_name_path, config=config)
        elif model_configuration.model_type == TextSegmentationModels.t5_encoder_decoder_token_level.value:
            # token classification using the complete model (i.e. enc-dec)
            model_config = AutoConfig.from_pretrained(model_name_path).to_dict()
            # remove redundandant keys from model_config
            for k in class_attributes_to_dict(model_configuration).keys():
                if k in model_config:
                    model_config.pop(k)
            config = T5TokenClassificationConfig(**model_config)
            model = T5EncoderDecoderTokenClassification.from_pretrained(model_name_path, config=config)
        else:
            raise ValueError(f"Model type: {model_configuration.model_type} was not recognized.")

    else:  # if it is a model we do not need to do anything
        model = model_name_path

    return model, tokenizer


def get_model_predictions(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, examples_list: List[str],
                          labels_list: List[List[int]], spacy_segmentation: bool, only_consider_segments: bool,
                          ignore_label: Union[int, None], break_tokens_list: Union[List[str], None],
                          ) -> List[List[int]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    model_output_labels = []

    with torch.no_grad():
        spacy_nlp = get_spacy_nlp() if spacy_segmentation else None
        for example_str, labels in tqdm(zip(examples_list, labels_list), total=len(labels_list)):
            current_labels = []
            inputs = tokenizer(example_str, add_special_tokens=True,
                               truncation=True,
                               return_offsets_mapping=True,
                               # padding="max_length", max_length=tokenizer.model_max_length,
                               return_tensors="pt").to(device)

            offset_mapping = inputs.pop("offset_mapping").squeeze()

            if only_consider_segments:
                # get the sentences
                current_sentences = get_segments(text=example_str, spacy_nlp=spacy_nlp, separator="\n",
                                                 remove_empty=False, keep_separator=True)
                # convert sentence lens to possible break tokens
                current_offset = 0
                sentence_break_lens = []
                for sentence in current_sentences:
                    current_offset += len(sentence) + (-1 if not spacy_segmentation else 0)
                    if current_offset in labels:
                        sentence_break_lens.append(current_offset)
                    elif not spacy_segmentation and any(
                            current_offset - 1 <= la <= current_offset + 1 for la in labels):
                        sentence_break_lens.append(current_offset)
                    else:
                        sentence_break_lens.append(current_offset)
                    current_offset += 1

                token_level_labels = convert_char_label_to_token_labels_given_segments(offset_mapping.tolist(), labels,
                                                                                       sentence_break_lens,
                                                                                       ignore_label)
            else:
                token_level_labels = convert_char_label_to_token_labels(offset_mapping.tolist(), labels)

            token_level_labels = torch.tensor(token_level_labels).to(device)  # convert to tensor

            if len(token_level_labels.size()) == 1:
                token_level_labels = token_level_labels.unsqueeze(dim=0)

            output = model(**inputs, labels=token_level_labels, return_dict=True)
            # loss = output.loss

            predictions = torch.argmax(output.logits.squeeze(), dim=-1)

            # apply the attention mask to remove predictions in padding
            # predictions = inputs.attention_mask.squeeze() * predictions

            # remove predictions over ignore tokens by replacing with zero
            if ignore_label is not None:
                predictions = torch.where(token_level_labels.squeeze(dim=0) == ignore_label, 0, predictions)

            sentence_breaks = torch.nonzero(predictions).squeeze().tolist()

            if isinstance(sentence_breaks, int):  # avoid cases where returns a single int
                sentence_breaks = [sentence_breaks]

            for sentence_break in sentence_breaks:
                offset_mapping_break = offset_mapping[sentence_break]
                break_position = offset_mapping_break[1].item()  # [1] to get the end

                if break_position != 0 and break_position < len(example_str):  # to avoid referencing special tokens
                    if break_tokens_list:  # if we have a post_processing_list we add to the labels if it matches
                        # iterate over the list to consider convert id to str to compare
                        if any([s in tokenizer.convert_ids_to_tokens([inputs.input_ids[0][sentence_break]]) for s in
                                break_tokens_list]):
                            current_labels.append(break_position)
                    else:  # always add
                        current_labels.append(break_position)  # [1] to get the end

            model_output_labels.append(current_labels)

    assert len(examples_list) == len(model_output_labels)

    return model_output_labels


@dataclass
class DataCollatorForTokenClassificationCustom:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    # only works with tensors
    def __call__(self, features):

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        # labels padding
        if labels is not None:
            self.simple_padding(batch, label_name, labels, padding_side, sequence_length, self.label_pad_token_id)

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch

    @staticmethod
    def simple_padding(batch, label_name: str, labels: List, padding_side: str, sequence_length: int,
                       label_pad_token_id: int):
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [label_pad_token_id] * (sequence_length - len(label)) for label in labels
            ]
        else:
            batch[label_name] = [
                [label_pad_token_id] * (sequence_length - len(label)) + list(label) for label in labels
            ]


def train_eval_model(configuration: Dict):
    # train model and evaluate on validation set

    # get the model configuration
    model_configuration = TokenLevelClassifierConfiguration(configuration)
    print("Model Configuration:")
    print(class_attributes_to_dict(model_configuration))
    print()

    # set seeds for reproducibility
    set_seeds(model_configuration.seed)

    # get model and tokenizer
    model, tokenizer = load_model_tokenizer(
        model_name_path=model_configuration.model_name,
        tokenizer_name_path=model_configuration.tokenizer_name,
        do_lower_case=model_configuration.do_lower_case,
        model_configuration=model_configuration
    )

    # update configurations for logging purposes
    custom_config_dict = class_attributes_to_dict(model_configuration)
    final_dict = {}
    for k, v in custom_config_dict.items():
        if model.config and k not in model.config.to_dict():  # avoid putting on top of an existing value
            final_dict[k] = v
    model.config.update(final_dict)

    # load the data
    _, _, _, train_examples_list, train_labels_list = load_dataset_split(model_configuration.train_data_location,
                                                                         model_configuration.train_number_samples)
    _, _, _, valid_examples_list, valid_labels_list = load_dataset_split(model_configuration.valid_data_location,
                                                                         model_configuration.eval_number_samples)
    test_ids_list, test_titles_list, original_test_examples, test_examples_list, test_labels_list = load_dataset_split(
        model_configuration.test_data_location,
        model_configuration.eval_number_samples
    )

    # create the datasets using the tokenized data and labels
    print("Creating Train Dataset...")
    train_dataset = TokenLevelDataset(
        tokenizer, train_examples_list, train_labels_list,
        ignore_label=model_configuration.label_to_ignore,
        spacy_segmentation=model_configuration.spacy_segmentation,
        only_consider_segments=model_configuration.only_consider_segments,
    )
    print("Creating Validation Dataset...")
    valid_dataset = TokenLevelDataset(
        tokenizer, valid_examples_list, valid_labels_list,
        spacy_segmentation=model_configuration.spacy_segmentation,
        only_consider_segments=model_configuration.only_consider_segments,
    )

    # create the trainer object for training and validation
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=model_configuration.get_trainer_training_arguments(),  # get the training arguments
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        # metrics to compute
        compute_metrics=compute_metrics_token_level,
        data_collator=DataCollatorForTokenClassificationCustom(tokenizer),
    )

    # actual training and validation procedure
    print("Started Training...")
    train_results = trainer.train(ignore_keys_for_eval=["hidden_states", "attentions"])
    print("Training Results:")
    print(train_results)
    print()

    # testing procedure
    print("Getting model predictions...")
    model_output_labels = get_model_predictions(
        model=trainer.model,
        tokenizer=tokenizer,
        examples_list=test_examples_list,
        labels_list=test_labels_list,
        spacy_segmentation=model_configuration.spacy_segmentation,
        only_consider_segments=model_configuration.only_consider_segments,
        ignore_label=model_configuration.label_to_ignore,
        break_tokens_list=model_configuration.break_tokens_list,
    )

    # evaluate best model and write predictions to file
    checkpoint_name = extract_checkpoint_name(str(trainer.state.best_model_checkpoint))
    test_config_and_metrics = write_results_to_file_and_officially_evaluate_model(
        model_configuration=model_configuration,
        test_ids_list=test_ids_list,
        test_titles_list=test_titles_list,
        original_test_examples=original_test_examples,
        test_examples_list=test_examples_list,
        model_output_labels=model_output_labels,
        model_name_or_path=str(trainer.model.name_or_path),
        checkpoint=checkpoint_name
    )
    test_config_and_metrics = add_suffix_to_metrics(suffix="test_set", current_metrics=test_config_and_metrics)

    all_config_metrics_path = model_configuration.test_run_name + checkpoint_name + model_configuration.suffix + "_all.json"
    all_config_metrics_path = os.path.join(model_configuration.output_dir, all_config_metrics_path)
    write_dict_to_json_file(data=test_config_and_metrics, output_path=all_config_metrics_path)


def evaluate_model(configuration: Dict, model_or_location: Union[str, PreTrainedModel],
                   eval_number_samples: Union[int, None],
                   test_data_location: Union[str, None],
                   suffix: str = "eval"):
    # test the model on the training set
    # model_location is a path to a folder with a model saved e.g. "./results/bert-base-uncased/checkpoint-xxx"

    # get the model configuration
    model_configuration = TokenLevelClassifierConfiguration(configuration)

    # set the attributes passed in the model configuration
    model_configuration.eval_number_samples = eval_number_samples
    model_configuration.test_data_location = test_data_location

    print("Model Configuration:")
    print(class_attributes_to_dict(model_configuration))
    print()

    # get model and tokenizer
    print("Loading Model and Tokenizer...")
    model, tokenizer = load_model_tokenizer(
        model_name_path=model_or_location,
        tokenizer_name_path=model_configuration.tokenizer_name,
        do_lower_case=model_configuration.do_lower_case,
        model_configuration=model_configuration
    )

    # put model in the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model.to(device)
    model.eval()

    # read dataset file
    ids_list, titles_list, original_test_examples, test_examples_list, test_labels_list = load_dataset_split(
        file_path=model_configuration.test_data_location,
        number_samples=model_configuration.eval_number_samples
    )

    # evaluate the model on the test set
    print("Getting model predictions...")
    model_output_labels = get_model_predictions(
        model=model, tokenizer=tokenizer,
        examples_list=test_examples_list,
        labels_list=test_labels_list,
        spacy_segmentation=model_configuration.spacy_segmentation,
        only_consider_segments=model_configuration.only_consider_segments,
        ignore_label=model_configuration.label_to_ignore,
        break_tokens_list=model_configuration.break_tokens_list,
    )

    # evaluate best model and write predictions to file
    if isinstance(model_or_location, str):
        checkpoint_name = extract_checkpoint_name(model_or_location)
    else:
        checkpoint_name = extract_checkpoint_name(str(model_or_location.name_or_path))

    config_and_metrics = write_results_to_file_and_officially_evaluate_model(
        model_configuration=model_configuration,
        test_ids_list=ids_list,
        test_titles_list=titles_list,
        original_test_examples=original_test_examples,
        test_examples_list=test_examples_list,
        model_output_labels=model_output_labels,
        model_name_or_path=str(model.name_or_path),
        checkpoint=checkpoint_name,
        suffix=suffix
    )

    return config_and_metrics
