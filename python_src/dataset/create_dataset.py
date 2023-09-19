import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Tuple, Union
from transformers import PreTrainedTokenizer, TensorType
from python_src.dataset.utils import get_spacy_nlp, get_segments, get_segments_from_labels_list, CONVENTIONAL_END_TOKENS
import numpy as np


def convert_char_label_to_token_labels(offset_mapping: List[Tuple[int, int]], char_labels: List[int]) -> List[int]:
    token_labels = []
    for i in offset_mapping:
        _, end_char = i
        if end_char in char_labels:
            token_labels.append(1)
        else:
            token_labels.append(0)
    return token_labels


def convert_char_label_to_token_labels_given_segments(offset_mapping: List[Tuple[int, int]], char_labels: List[int],
                                                      sentence_break_lens: List[int], ignore_label: int) -> List[int]:
    # only puts 0 and 1 in segment locations while the others use an ignore label
    token_labels = []
    for i in offset_mapping:
        _, end_char = i
        if end_char in sentence_break_lens:
            if end_char in char_labels:  # it is a sentence break and needs segmentation
                token_labels.append(1)
            else:
                token_labels.append(0)  # it is a sentence break but does not need to segment
        else:
            token_labels.append(ignore_label)  # ignore tokens that are not segment breaks
    return token_labels


class TokenLevelDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, examples_list: List[str], labels_list: List[List[int]],
                 spacy_segmentation: bool, only_consider_segments: bool,
                 block_size: Union[None, int] = None, ignore_label: int = -100):

        self.inputs, self.labels, self.texts = [], [], []
        avg_seg_sizes = []
        self.tokenizer = tokenizer
        if block_size is None:
            self.block_size = tokenizer.model_max_length
        else:
            self.block_size = block_size

        # iterate through examples and labels gives positive examples
        spacy_nlp = get_spacy_nlp() if spacy_segmentation else None
        for example_str, labels in tqdm(zip(examples_list, labels_list), total=len(examples_list)):
            # must use fast tokenizer because of return_offsets_mapping
            inputs = tokenizer(example_str, add_special_tokens=True,
                               truncation=True,
                               return_offsets_mapping=True,
                               # padding="max_length",
                               max_length=self.block_size)

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

                token_level_labels = convert_char_label_to_token_labels_given_segments(inputs.offset_mapping, labels,
                                                                                       sentence_break_lens,
                                                                                       ignore_label)
            else:
                token_level_labels = convert_char_label_to_token_labels(inputs.offset_mapping, labels)

            token_level_labels = torch.tensor(token_level_labels)

            if ignore_label is not None:
                # replace by ignore_label in the positions where the attention mask is zero
                # so that they are ignored by the model when calculating metrics
                token_level_labels = torch.where(
                    torch.tensor(inputs.attention_mask) == 1,
                    token_level_labels,
                    torch.tensor(ignore_label).type_as(token_level_labels))

            self.inputs.append(inputs)
            self.labels.append(token_level_labels.tolist())
            self.texts.append(example_str)

            # get avg_size of breaks using tokenized words
            avg_seg_sizes += [len(segment) for segment in get_segments_from_labels_list(label_list=token_level_labels.tolist(), split_value=1,
                                                                                        ignore_value=None)]  # changed ignore_label to None

        self.avg_seg_size = np.mean(avg_seg_sizes)
        self.stdev_seg_size = np.std(avg_seg_sizes)

        print("avg_seg_size", self.avg_seg_size, "stdev_seg_size", self.stdev_seg_size)

        self.verb_tokens, self.noun_tokens = [], []
        self.avg_verbs_size, self.avg_nouns_size = 0, 0
        self.stdev_verbs_size, self.stdev_nouns_size = 0, 0

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i: int):
        item = {key: val for key, val in self.inputs[i].items() if key != 'offset_mapping'}
        item['labels'] = self.labels[i]
        return item


# cross encoder dataset that which can have the context reset after each break
# a single label per example 0 or 1 if it should not or should break
class CrossEncoderDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, examples_list: List[str], labels_list: List[List[int]],
                 spacy_segmentation: bool, block_size: Union[None, int] = None,
                 size_ignore_small_non_break: Union[None, int] = None,
                 prob_ignore_small_non_break: Union[None, float] = None, random_seed: int = 42):

        if random_seed is not None:
            random.seed(random_seed)

        self.inputs, self.labels = [], []
        self.tokenizer = tokenizer
        if block_size is None:
            self.block_size = tokenizer.model_max_length
        else:
            self.block_size = block_size
        spacy_nlp = get_spacy_nlp() if spacy_segmentation else None

        for example_str, labels in tqdm(zip(examples_list, labels_list), total=len(examples_list)):
            current_sentences = get_segments(text=example_str, spacy_nlp=spacy_nlp, separator="\n",
                                             remove_empty=False, keep_separator=True)
            left_context, right_context = "", ""
            labels_set = set(labels)  # create a set for faster access
            reset_context = False
            current_offset = 0
            for i in range(len(current_sentences)-1):  # do not count with the last one because it is trivial to break
                if reset_context:
                    left_context = ""

                # concat the current sentence
                left_context += ("" if not left_context or not spacy_segmentation else " ") + current_sentences[i]
                right_context = current_sentences[i+1]  # right context is always the next sentence
                current_offset += len(current_sentences[i])

                if not left_context.strip() or not right_context.strip():  # if any of the contexts is an empty str
                    continue  # skip

                if current_offset in labels_set:  # if we need to break in the next iteration we will reset the left context
                    reset_context = True
                    current_label = 1  # break label
                # account for the imprecisions in the dataset
                elif not spacy_segmentation and any(current_offset-2 <= la <= current_offset+2 for la in labels_set):
                    reset_context = True
                    current_label = 1  # break label
                else:
                    reset_context = False  # no need to reset
                    current_label = 0  # do not break label

                # we only add the extra space in case spacy was used to add the space between sentences
                if spacy_segmentation:
                    current_offset += 1

                if self.should_skip_easier_example(current_label, left_context, right_context,
                                                   size_ignore_small_non_break, prob_ignore_small_non_break):
                    continue  # skip

                inputs = self.tokenizer(left_context, right_context, add_special_tokens=True,
                                        truncation="longest_first",
                                        max_length=self.block_size)

                self.inputs.append(inputs.convert_to_tensors(tensor_type=TensorType.PYTORCH.value,
                                                             prepend_batch_axis=False))
                self.labels.append(torch.tensor(current_label))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i: int):
        item = {key: val for key, val in self.inputs[i].items()}
        item['labels'] = self.labels[i]
        return item

    @staticmethod
    def should_skip_easier_example(current_label: int, left_context: str, right_context: str,
                                   size_ignore_small_non_break: Union[None, int] = None,
                                   prob_small_non_break_to_ignore: Union[None, float] = None) -> bool:
        # if is not a break example i.e. current_label = 0
        # the size of left and right context is lower than size_ignore_small_non_break
        # we check the prob_small_non_break_to_ignore to remove this example
        # resulting in less easier examples and less unbalance between positive and negative classes
        if current_label == 0 and size_ignore_small_non_break is not None \
                and prob_small_non_break_to_ignore is not None:
            prob = random.random()
            if len(f"{left_context} {right_context}".split()) <= size_ignore_small_non_break \
                    and prob <= prob_small_non_break_to_ignore:
                return True
        return False


# cross encoder dataset that to each candidate break fetches tokens to the left and right until filling the 512 tokens
# similar to the one in the paper Text Segmentation by Cross Segment Attention
# a single label per example 0 or 1 if it should not or should break
class CrossEncoderWithoutResetContextDataset(Dataset):

    def __init__(self, tokenizer: PreTrainedTokenizer, examples_list: List[str], labels_list: List[List[int]],
                 spacy_segmentation: bool, block_size: Union[None, int] = None):

        self.inputs, self.labels = [], []
        self.tokenizer = tokenizer
        if block_size is None:
            self.block_size = tokenizer.model_max_length
        else:
            self.block_size = block_size

        spacy_nlp = get_spacy_nlp() if spacy_segmentation else None
        for example_str, labels in tqdm(zip(examples_list, labels_list), total=len(examples_list)):
            current_sentences = get_segments(text=example_str, spacy_nlp=spacy_nlp, separator="\n",
                                             remove_empty=False, keep_separator=True)
            current_offset = 0
            candidate_breaks = []

            # get the candidate breaks
            for sentence in current_sentences[:-1]:  # do not count with the last one because it is trivial to break
                current_offset += len(sentence) + (-1 if not spacy_segmentation else 0)
                candidate_breaks.append(current_offset)
                current_offset += 1

            # tokens without special tokens for the full sentence
            inputs = tokenizer(example_str, add_special_tokens=False,
                               return_offsets_mapping=True)

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

                        if example_str[end_char-1] in CONVENTIONAL_END_TOKENS:  # start with current
                            current_offset = i
                        elif example_str[end_char_before-1] in CONVENTIONAL_END_TOKENS:  # check before
                            end_char = end_char_before
                            current_offset = i - 1
                        elif example_str[end_char_after-1] in CONVENTIONAL_END_TOKENS:  # check after
                            end_char = end_char_after
                            current_offset = i + 1
                        break
                if current_offset != -1 and end_char is not None:  # we found one of the breaks
                    segment_start, _ = inputs.offset_mapping[max(0, current_offset-each_side_context_size)]
                    _, segment_end = inputs.offset_mapping[min(current_offset+each_side_context_size,
                                                               len(inputs.offset_mapping)-1)]
                    if end_char in labels:  # exact match
                        current_label = 1
                    # take into account imprecisions of the dataset
                    elif not spacy_segmentation and any(end_char - 3 <= la <= end_char + 3 for la in labels):
                        current_label = 1
                    else:  # not a match
                        current_label = 0

                    # get the left and right context in string format
                    left_context = example_str[segment_start:end_char+1]
                    right_context = example_str[end_char+1: segment_end]

                    # encode again to add the special tokens
                    final_inputs = self.tokenizer(left_context, right_context, add_special_tokens=True,
                                                  truncation="longest_first",
                                                  max_length=self.block_size
                                                  )

                    self.inputs.append(final_inputs.convert_to_tensors(tensor_type=TensorType.PYTORCH.value,
                                                                       prepend_batch_axis=False))
                    self.labels.append(torch.tensor(current_label))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i: int):
        item = {key: val for key, val in self.inputs[i].items()}
        item['labels'] = self.labels[i]
        return item
