import os

from tqdm import tqdm
from python_src.models.model_utils import compute_metrics_cross_segment, DefaultModelConfiguration, \
    write_results_to_file_and_officially_evaluate_model, TextSegmentationModels, set_seeds, extract_checkpoint_name, \
    add_suffix_to_metrics
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    Trainer, PreTrainedModel, PreTrainedTokenizer, DataCollatorWithPadding
from typing import List, Dict, Tuple, Union
from python_src.dataset.create_dataset import CrossEncoderDataset, CrossEncoderWithoutResetContextDataset
from python_src.dataset.utils import load_dataset_split, get_spacy_nlp, class_attributes_to_dict, get_segments, \
    CONVENTIONAL_END_TOKENS, write_dict_to_json_file


class CrossEncoderModelConfiguration(DefaultModelConfiguration):

    def __init__(self, configuration: Dict):
        super().__init__(configuration)

        # overwritten attributes
        self.text_segmentation_model_type = configuration.get("text_segmentation_model_type",
                                                              TextSegmentationModels.cross_encoder.value)
        self.metric_for_best_model = configuration.get("metric_for_best_model", "f1_micro")

        # cross encoder specific
        self.with_reset_context = configuration.get("with_reset_context", True)
        self.size_ignore_small_non_break = configuration.get("size_ignore_small_non_break", None)  # or 50
        self.prob_small_non_break_to_ignore = configuration.get("prob_small_non_break_to_ignore", None)  # or 0.5

        self.train_run_name = configuration.get("train_run_name", self.model_name.replace("/", "-")) + self.suffix
        self.test_run_name = configuration.get("test_run_name",
                                               self.model_name.replace("/", "-") + "_test") + self.suffix

        # output directory
        self.output_dir = configuration.get("output_dir", f'./model_results/'
                                                          f'{self.train_run_name}_{self.learning_rate}_'
                                                          f'{self.per_device_train_batch_size}_cross_encoder_reset_'
                                                          f'{self.with_reset_context}')  # output directory


def load_model_tokenizer(model_name_path: str, tokenizer_name_path: str, do_lower_case: bool) -> \
        Tuple[PreTrainedModel, PreTrainedTokenizer]:
    # model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_path, do_lower_case=do_lower_case)

    model = AutoModelForSequenceClassification.from_pretrained(model_name_path, num_labels=2)

    return model, tokenizer


def get_model_predictions(model: PreTrainedModel, tokenizer: PreTrainedTokenizer,
                          examples_list: List[str], labels_list: List[List[int]],
                          reset_context: bool, spacy_segmentation: bool) -> List[List[int]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    model_output_labels = []
    spacy_nlp = get_spacy_nlp() if spacy_segmentation else None

    with torch.no_grad():
        for example_str, labels in tqdm(zip(examples_list, labels_list), total=len(examples_list)):
            current_labels = []
            # because the cross encoder does not detect boudaries we first get sentences
            # using spacy and feed pairs to the model
            current_sentences = get_segments(text=example_str, spacy_nlp=spacy_nlp, separator="\n",
                                             remove_empty=False, keep_separator=True)
            if reset_context:
                left_context, right_context = "", ""
                reset_context = False
                current_offset = 0
                for i in range(
                        len(current_sentences) - 1):  # do not count with the last one because it is trivial to break
                    if reset_context:
                        left_context = ""
                    left_context += ("" if not left_context or not spacy_segmentation else " ") + current_sentences[i]
                    right_context = current_sentences[i + 1]  # right context is always the next sentence
                    current_offset += len(current_sentences[i])

                    if not left_context.strip() or not right_context.strip():  # if any of the contexts is an empty str
                        continue  # skip

                    inputs = tokenizer(left_context, right_context, add_special_tokens=True,
                                       truncation="longest_first",
                                       padding=True, max_length=tokenizer.model_max_length,
                                       return_tensors="pt").to(device)

                    if current_offset in current_labels:  # if we need to break in the next iteration we will reset the left context
                        current_label = 1  # break label
                    # account for the imprecisions in the dataset
                    elif not spacy_segmentation and any(
                            current_offset - 2 <= la <= current_offset + 2 for la in current_labels):
                        current_label = 1  # break label
                    else:
                        current_label = 0  # do not break label

                    output = model(**inputs, labels=torch.tensor(current_label).to(device))
                    prediction = torch.argmax(output.logits.squeeze(), dim=-1)
                    if prediction:  # if we need to break in the next iteration we will reset the left context
                        reset_context = True
                        current_labels.append(current_offset)
                    else:
                        reset_context = False  # no need to reset

                    # we only add the extra space in case spacy was used to add the space between sentences
                    if spacy_segmentation:
                        current_offset += 1

            else:  # never resets context
                current_offset = 0
                candidate_breaks = []
                # get the candidate breaks
                for sentence in current_sentences[:-1]:  # do not count with the last one because it is trivial to break
                    current_offset += len(sentence) + (-1 if not spacy_segmentation else 0)
                    candidate_breaks.append(current_offset)
                    current_offset += 1

                # tokens without special tokens for the full sentence
                inputs = tokenizer(example_str, add_special_tokens=False,
                                   # truncation=True,
                                   return_offsets_mapping=True,
                                   # padding="max_length",
                                   # max_length=self.block_size
                                   )

                # get the max that the model can take on each side of a break
                # -3 because of the special tokens we still need to add
                each_side_context_size = int((tokenizer.model_max_length - 3) / 2)
                for candidate_break in candidate_breaks:  # iterate through candidate breaks
                    current_offset = -1
                    end_char = None
                    for i, span_tuple in enumerate(inputs.offset_mapping):  # iterate through offset_mapping
                        _, end_char = span_tuple
                        if end_char == candidate_break:  # it is exactly the same as the candidate break
                            current_offset = i
                            break
                        elif end_char > candidate_break:  # we passed the candidate break so we try to get the closest match
                            _, end_char_before = inputs.offset_mapping[i - 1]  # get the position before
                            _, end_char_after = inputs.offset_mapping[i + 1]  # get the position after

                            if example_str[end_char - 1] in CONVENTIONAL_END_TOKENS:  # start with current
                                current_offset = i
                            elif example_str[end_char_before - 1] in CONVENTIONAL_END_TOKENS:  # check before
                                end_char = end_char_before
                                current_offset = i - 1
                            elif example_str[end_char_after - 1] in CONVENTIONAL_END_TOKENS:  # check after
                                end_char = end_char_after
                                current_offset = i + 1
                            break
                    if current_offset != -1 and end_char is not None:  # we found one of the breaks
                        segment_start, _ = inputs.offset_mapping[max(0, current_offset - each_side_context_size)]
                        _, segment_end = inputs.offset_mapping[min(current_offset + each_side_context_size,
                                                                   len(inputs.offset_mapping) - 1)]
                        if end_char in labels:  # exact match
                            current_label = 1
                        # take into account imprecisions of the dataset
                        elif not spacy_segmentation and any(end_char - 3 <= la <= end_char + 3 for la in labels):
                            current_label = 1
                        else:  # not a match
                            current_label = 0

                        # get the left and right context in string format
                        left_context = example_str[segment_start:end_char + 1]
                        right_context = example_str[end_char + 1: segment_end]

                        # encode again to add the special tokens
                        final_inputs = tokenizer(left_context, right_context, add_special_tokens=True,
                                                 truncation="longest_first", padding="max_length",
                                                 max_length=tokenizer.model_max_length, return_tensors="pt").to(device)

                        output = model(**final_inputs, labels=torch.tensor(current_label).to(device))
                        prediction = torch.argmax(output.logits.squeeze(), dim=-1)
                        if prediction:  # if we break we add the break position
                            current_labels.append(end_char)

            model_output_labels.append(current_labels)

    assert len(examples_list) == len(model_output_labels)

    return model_output_labels


def train_eval_model(configuration: Dict):
    # train model and evaluate on validation set

    # get the model configuration
    model_configuration = CrossEncoderModelConfiguration(configuration)
    print("Model Configuration:")
    print(class_attributes_to_dict(model_configuration))
    print()

    # set seeds for reproducibility
    set_seeds(model_configuration.seed)

    # get model and tokenizer
    model, tokenizer = load_model_tokenizer(
        model_name_path=model_configuration.model_name,
        tokenizer_name_path=model_configuration.tokenizer_name,
        do_lower_case=model_configuration.do_lower_case
    )

    # load the data
    _, _, _, train_examples_list, train_labels_list = load_dataset_split(model_configuration.train_data_location,
                                                                         model_configuration.train_number_samples)
    _, _, _, valid_examples_list, valid_labels_list = load_dataset_split(model_configuration.valid_data_location,
                                                                         model_configuration.eval_number_samples
                                                                         )
    test_ids_list, test_titles_list, original_test_examples, test_examples_list, test_labels_list = load_dataset_split(
        model_configuration.test_data_location, model_configuration.eval_number_samples
    )

    # create the datasets using the tokenized data and labels
    if model_configuration.with_reset_context:
        print("Creating Train Dataset")
        # we only apply the special size_ignore_small_non_break and prob_ignore_small_non_break
        # on the training set because validation and testing should not be changed

        train_dataset = CrossEncoderDataset(
            tokenizer, train_examples_list, train_labels_list,
            size_ignore_small_non_break=model_configuration.size_ignore_small_non_break,
            prob_ignore_small_non_break=model_configuration.prob_small_non_break_to_ignore,
            random_seed=model_configuration.seed,
            spacy_segmentation=model_configuration.spacy_segmentation
        )
        print("Creating Validation Dataset")
        valid_dataset = CrossEncoderDataset(
            tokenizer, valid_examples_list, valid_labels_list,
            spacy_segmentation=model_configuration.spacy_segmentation
        )
    else:
        print("Creating Train Dataset")
        train_dataset = CrossEncoderWithoutResetContextDataset(
            tokenizer, train_examples_list, train_labels_list,
            spacy_segmentation=model_configuration.spacy_segmentation
        )
        print("Creating Validation Dataset")
        valid_dataset = CrossEncoderWithoutResetContextDataset(
            tokenizer, valid_examples_list, valid_labels_list,
            spacy_segmentation=model_configuration.spacy_segmentation
        )

    # create the trainer object for training and validation
    trainer = Trainer(
        model=model,  # the instantiated Transformers model to be trained
        args=model_configuration.get_trainer_training_arguments(),  # get the training arguments
        train_dataset=train_dataset,  # training dataset
        eval_dataset=valid_dataset,  # evaluation dataset
        # metrics to compute
        compute_metrics=compute_metrics_cross_segment,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer, padding=True)
    )

    # actual training and validation procedure
    print("Started Training...")
    train_results = trainer.train()
    print("Training Results:")
    print(train_results)
    print()

    print("Getting model predictions...")
    model_output_labels = get_model_predictions(
        model=trainer.model,
        tokenizer=tokenizer,
        examples_list=test_examples_list,
        labels_list=test_labels_list,
        reset_context=model_configuration.with_reset_context,
        spacy_segmentation=model_configuration.spacy_segmentation
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
    model_configuration = CrossEncoderModelConfiguration(configuration)
    print("Model Configuration:")
    print(class_attributes_to_dict(model_configuration))
    print()

    # get model and tokenizer
    print("Loading Model and Tokenizer...")
    model, tokenizer = load_model_tokenizer(
        model_name_path=model_or_location,
        tokenizer_name_path=model_configuration.tokenizer_name,
        do_lower_case=model_configuration.do_lower_case
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
        examples_list=test_examples_list, labels_list=test_labels_list,
        reset_context=model_configuration.with_reset_context,
        spacy_segmentation=model_configuration.spacy_segmentation
    )

    # evaluate best model and write predictions to file
    if isinstance(model_or_location, str):
        checkpoint_name = extract_checkpoint_name(model_or_location)
    else:
        checkpoint_name = extract_checkpoint_name(str(model_or_location.name_or_path))

    # evaluate best model and write predictions to file
    config_and_metrics = write_results_to_file_and_officially_evaluate_model(
        model_configuration=model_configuration,
        test_ids_list=ids_list,
        test_titles_list=titles_list,
        original_test_examples=original_test_examples,
        test_examples_list=test_examples_list,
        model_output_labels=model_output_labels,
        model_name_or_path=str(model.name_or_path),
        checkpoint=checkpoint_name
    )

    return config_and_metrics
