import csv
import json
import os
import spacy
from typing import List, Tuple, Dict, Union
import numpy as np
from tqdm import tqdm


CONVENTIONAL_END_TOKENS = [".", "!", "?", "...", ";", "\n"]


def get_spacy_nlp(fast: bool = True):
    if fast:
        # for_faster_performance disable some things that are currently not in use
        return spacy.load("en_core_web_sm", disable=["tagger", "ner", "lemmatizer", "textcat"])
    else:
        return spacy.load("en_core_web_sm")


def get_segments(text: str, spacy_nlp, separator: Union[str, None],
                 remove_empty: bool, keep_separator: bool) -> List[str]:
    # spacy nlp is a spacy object obtained for example by calling the get_spacy_nlp function
    # depending on the params a segment can be considered a sentence or a set of sentences separated by new lines

    if spacy_nlp is not None:  # uses spacy to get the sentences
        current_sentences = spacy_nlp(text).sents  # get spacy sentences
        current_sentences = [i.text.strip() for i in current_sentences]  # convert to str
    elif separator:  # separate by delimeter
        if keep_separator:  # keep the delimeter after splitting
            split_result = text.split(separator)
            current_sentences = [sentence + separator for sentence in split_result[:-1]] + [split_result[-1]]
        else:
            current_sentences = text.split(separator)
    else:
        raise ValueError("At least one of spacy_nlp or separator must be provided")

    if remove_empty:  # remove empty sentences
        current_sentences = [sentence for sentence in current_sentences if sentence.strip()]

    return current_sentences


def calculate_spacy_statistics(original_steps_list: List[List[str]], examples_list: List[str], spacy_nlp=None) -> Dict[str, np.ndarray]:
    if spacy_nlp is None:
        spacy_nlp = get_spacy_nlp(fast=False)

    avg_number_sentences = []
    avg_number_tokens = []
    avg_number_breaks = []
    avg_number_instructions = []
    avg_number_token_per_instruction = []
    avg_number_sentences_per_instruction = []
    avg_verb_count, avg_verb_count_per_instruction = [], []
    avg_noun_count, avg_noun_count_per_instruction = [], []
    for i, example in tqdm(enumerate(spacy_nlp.pipe(examples_list, batch_size=100)), total=len(examples_list)):
        avg_number_sentences.append((len(list(example.sents))))
        avg_number_tokens.append(len(example))
        avg_number_breaks.append(len(original_steps_list[i]) - 1)
        avg_number_instructions.append(len(original_steps_list[i]))

        verb_count, noun_count = 0, 0
        for token in example:
            if token.pos_ == "VERB":
                verb_count += 1
            elif token.pos_ == "NOUN":
                noun_count += 1

        avg_verb_count.append(verb_count)
        avg_noun_count.append(noun_count)

        for original_instruction in spacy_nlp.pipe(original_steps_list[i], batch_size=100):
            verb_count, noun_count = 0, 0
            for token in original_instruction:
                if token.pos_ == "VERB":
                    verb_count += 1
                elif token.pos_ == "NOUN":
                    noun_count += 1

            avg_verb_count_per_instruction.append(verb_count)
            avg_noun_count_per_instruction.append(noun_count)

            avg_number_token_per_instruction.append(len(original_instruction))
            avg_number_sentences_per_instruction.append((len(list(original_instruction.sents))))

    return {
        "total_number_examples": len(examples_list),
        "total_number_intructions": sum(avg_number_instructions),
        "avg_number_sentences": np.mean(avg_number_sentences),
        "std_number_sentences": np.std(avg_number_sentences),
        "avg_number_tokens": np.mean(avg_number_tokens),
        "std_number_tokens": np.std(avg_number_tokens),
        "avg_number_breaks": np.mean(avg_number_breaks),
        "std_number_breaks": np.std(avg_number_breaks),
        "avg_number_instructions": np.mean(avg_number_instructions),
        "std_number_instructions": np.std(avg_number_instructions),
        "avg_number_token_per_instruction": np.mean(avg_number_token_per_instruction),
        "std_number_token_per_instruction": np.std(avg_number_token_per_instruction),
        "avg_number_sentences_per_instruction": np.mean(avg_number_sentences_per_instruction),
        "std_number_sentences_per_instruction": np.std(avg_number_sentences_per_instruction),
        "avg_verb_count": np.mean(avg_verb_count),
        "std_verb_count": np.std(avg_verb_count),
        "avg_verb_count_per_instruction": np.mean(avg_verb_count_per_instruction),
        "std_verb_count_per_instruction": np.std(avg_verb_count_per_instruction),
        "avg_noun_count": np.mean(avg_noun_count),
        "std_noun_count": np.std(avg_noun_count),
        "avg_noun_count_per_instruction": np.mean(avg_noun_count_per_instruction),
        "std_noun_count_per_instruction": np.std(avg_noun_count_per_instruction),
    }


def load_dataset_split(file_path: str, number_samples: Union[int, None] = None) \
        -> Tuple[List[int], List[str], List[List[str]], List[str], List[List[int]]]:
    ids_list, titles_list, original_steps_list, steps, labels_list = [], [], [], [], []
    with open(file_path) as f:
        read_tsv = csv.reader(f, delimiter="\t")
        for i, row in enumerate(read_tsv):
            if i == 0:  # ignore header
                continue
            if number_samples and i > number_samples:
                break
            current_id = str(row[0])
            try:  # for cases where the id is numeric
                ids_list.append(int(current_id))
            except ValueError:
                ids_list.append(current_id)

            titles_list.append(json.loads(row[1]))
            original_steps_list.append(json.loads(row[2]))
            steps.append(json.loads(row[3]))
            labels_list.append(json.loads(row[4]))

    assert len(original_steps_list) == len(steps) == len(labels_list)

    return ids_list, titles_list, original_steps_list, steps, labels_list


def get_segments_from_labels(example: str, label_list: List[int]) -> List[str]:
    segments = []
    last_label = 0
    for label in label_list:
        segments.append(example[last_label:label+1].strip())
        last_label = label
    # account for the last label
    last_segment = example[last_label+1:].strip()
    if last_segment:
        segments.append(last_segment)
    # if no segments is just add the entire segment
    if not segments:
        segments.append(example)

    return segments


def get_segments_from_labels_list(label_list: List[int], split_value: int = 1, ignore_value=-100) -> List[List[int]]:
    # segments a list of ints by a specific value
    segments = []
    offset = 0
    for i, value in enumerate(label_list):
        if value == split_value:
            segments.append(label_list[offset:i + 1])
            offset = i + 1
    # account for the last segment
    last_segment = label_list[offset:]
    if last_segment:
        segments.append(last_segment)
    # there are segments but no splits add the entire list as a segment
    if not segments:
        segments.append(label_list)

    # remove values to ignore
    final_segments = []
    if ignore_value is not None:
        for segment in segments:
            final_segments.append([])
            for value in segment:
                if value != ignore_value:
                    final_segments[-1].append(value)
    else:
        final_segments = segments

    return final_segments


def write_dict_to_json_file(data: Dict, output_path: str, indent: int = 2):
    with open(output_path, "w") as f_open:
        json.dump(data, f_open, indent=indent)


def class_attributes_to_dict(config_class: object) -> Dict:
    class_attributes = config_class.__dict__.items()
    json_attributes = {}
    for attr_name, attr_value in class_attributes:
        if attr_name not in ["__module__", "__dict__", "__weakref__", "__doc__"]:
            json_attributes[attr_name] = attr_value
    return json_attributes


def write_configuration_to_file(config_class: object, output_path: str, indent: int = 2) -> Dict:
    json_attributes = class_attributes_to_dict(config_class)
    write_dict_to_json_file(json_attributes, f"{output_path}", indent)
    return json_attributes


def write_split_to_file(ids_list: List[int], titles_list: List[str],
                        original_steps_list: List[List[str]], examples_list: List[str], labels_list: List[List[int]],
                        out_file_path: str, out_file_name: str, method_labels_list: List[List[int]] = None):
    with open(os.path.join(out_file_path, out_file_name + ".tsv"), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')

        # write header
        header_titles = ["id", "title", "original_steps_list", "steps", "labels_list"]
        if method_labels_list:  # if using this add to the headers
            header_titles.append("method_level_labels_list")
        tsv_writer.writerow(header_titles)

        # write each line to file
        for i in range(len(ids_list)):
            if method_labels_list:  # just adds another column
                tsv_writer.writerow([ids_list[i], json.dumps(titles_list[i]), json.dumps(original_steps_list[i]),
                                     json.dumps(examples_list[i]), json.dumps(labels_list[i]),
                                     json.dumps(method_labels_list[i])])
            else:
                tsv_writer.writerow([ids_list[i], json.dumps(titles_list[i]), json.dumps(original_steps_list[i]),
                                     json.dumps(examples_list[i]), json.dumps(labels_list[i])])
