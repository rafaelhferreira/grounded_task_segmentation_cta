from nltk.metrics.segmentation import pk
from python_src.dataset.utils import load_dataset_split, write_dict_to_json_file, get_segments_from_labels
from typing import Dict, List, Union
from statistics import mean

SPECIAL_BOUNDARY_TOKEN = "|"
SPECIAL_REPLACEMENT_TOKEN = "#"  # used to replace SPECIAL_BOUNDARY_TOKEN when it appears in the text


def evaluate_over_dataset(split_file_path: str, model_output_file: str,
                          number_samples: Union[int, None] = None, output_file: Union[str, None] = None,
                          model_name: Union[str, None] = None, checkpoint: Union[str, None] = None) -> Dict:

    test_ids, _, steps_list, steps, labels_list = load_dataset_split(split_file_path)
    model_ids, _, model_steps_list, model_steps, model_labels_list = load_dataset_split(model_output_file)

    if number_samples:  # consider everything
        steps, labels_list = steps[:number_samples], labels_list[:number_samples]
        model_steps, model_labels_list = model_steps[:number_samples], model_labels_list[:number_samples]

    assert len(steps) == len(model_steps), f"Len of evaluation ({len(steps)}) and model output ({len(model_steps)}) is not the same"

    pk_final, pk_word_level_final = [], []
    precision_final, recall_final, f1_final = [], [], []
    avg_number_model_breaks, avg_number_breaks = [], []
    same_number_breaks, one_of_difference, more_breaks, less_breaks = 0, 0, 0, 0
    avg_words_model_break, avg_words_break = [], []  # in number of words
    for test_id, model_id, steps_l, model_steps_l, step, model_step, label_list, model_labels in zip(test_ids, model_ids, steps_list, model_steps_list,
                                                                                                     steps, model_steps, labels_list, model_labels_list):

        assert test_id == model_id, f"test_id {test_id} and model_id {model_id} do not match."
        # token level evaluation
        original = convert_sentence(step, label_list)
        model_output = convert_sentence(model_step, model_labels)
        pk_result = pk(original, model_output, boundary=SPECIAL_BOUNDARY_TOKEN)
        pk_final.append(pk_result)

        # word level evaluation
        original_word_level = word_level_sentence(step, label_list)
        model_output_word_level = word_level_sentence(model_step, model_labels)
        pk_word_level_result = pk(original_word_level, model_output_word_level, boundary=SPECIAL_BOUNDARY_TOKEN)
        pk_word_level_final.append(pk_word_level_result)

        avg_words_break.append(sum(len(s.split()) for s in steps_l) / len(steps_l))
        model_labels_len = len(model_labels) if len(model_labels) else 1
        avg_words_model_break.append(sum(len(s.split()) for s in get_segments_from_labels(model_step, model_labels)) / model_labels_len)

        # precision, recall and f1 following https://www.ijcai.org/proceedings/2018/0579.pdf
        precision_final.append(calc_precision(label_list, model_labels))
        recall_final.append(calc_recall(label_list, model_labels))
        f1_final.append(calc_f1(label_list, model_labels))

        # other useful stats
        if len(model_labels) == len(label_list):
            same_number_breaks += 1
        if abs(len(model_labels) - len(label_list)) <= 1:
            one_of_difference += 1
        if len(model_labels) > len(label_list):
            more_breaks += 1
        elif len(model_labels) < len(label_list):
            less_breaks += 1

        avg_number_model_breaks.append(len(model_labels))
        avg_number_breaks.append(len(label_list))

    pk_final = mean(pk_final)
    pk_word_level_final = mean(pk_word_level_final)
    perfect_matches = sum(f1_score >= 0.99 for f1_score in f1_final)
    precision_final = mean(precision_final)
    recall_final = mean(recall_final)
    f1_final = mean(f1_final)
    avg_number_model_breaks = mean(avg_number_model_breaks)
    avg_number_breaks = mean(avg_number_breaks)
    avg_words_model_break = mean(avg_words_model_break)
    avg_words_break = mean(avg_words_break)

    total_examples = len(steps)
    official_metrics = {
        # general info
        "split_file_path": split_file_path,
        "model_output_file": model_output_file,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "number_samples": total_examples,
        # main metrics
        "pk": pk_final,
        "pk_word_level": pk_word_level_final,
        "precision": precision_final,
        "recall": recall_final,
        "f1": f1_final,
        # other stats
        "perfect_matches": perfect_matches,
        "perfect_matches_perc": round(perfect_matches * 100 / total_examples, 2),
        "same_number_breaks": same_number_breaks,
        "same_number_breaks_per": round(same_number_breaks * 100 / total_examples, 2),
        "one_of_difference": one_of_difference,
        "one_of_difference_per": round(one_of_difference * 100 / total_examples, 2),
        "more_breaks": more_breaks,
        "more_breaks_perc": round(more_breaks * 100 / total_examples, 2),
        "less_breaks": less_breaks,
        "less_breaks_perc": round(less_breaks * 100 / total_examples, 2),
        "avg_number_breaks": avg_number_breaks,
        "avg_number_model_breaks": avg_number_model_breaks,
        "avg_words_model_break": avg_words_model_break,
        "avg_words_break": avg_words_break
    }

    # write the result to the file
    if output_file:
        write_dict_to_json_file(data=official_metrics, output_path=output_file)

    return official_metrics


def convert_sentence(sentence: str, labels: List[int], boundary: str = SPECIAL_BOUNDARY_TOKEN,
                     replacement_token: str = SPECIAL_REPLACEMENT_TOKEN) -> str:
    # start by replacing the special token by another random token to not influence the calculations
    new_sentence = sentence.replace(boundary, replacement_token)
    for label in labels:
        # replaces the token at the specific place by a special token to calculate metrics like pk
        if label != len(sentence):  # ignore last label since it is trivial
            new_sentence = new_sentence[:label] + boundary + new_sentence[label + 1:]
    if new_sentence.endswith(SPECIAL_BOUNDARY_TOKEN):  # remove break if it is the last token
        new_sentence = new_sentence[:-1]

    return new_sentence


def word_level_sentence(sentence: str, labels: List[int], boundary: str = SPECIAL_BOUNDARY_TOKEN,
                        replacement_token: str = SPECIAL_REPLACEMENT_TOKEN) -> str:
    # creates a zero and one list where each value is a word
    new_sentence = convert_sentence(sentence, labels, boundary, replacement_token)

    # print("new_sentence", len(new_sentence), new_sentence)

    output = ""
    segments = new_sentence.split(boundary)
    # print(segments)
    for i, segment in enumerate(segments):
        n_words = len(segment.split())
        # print(n_words)
        if n_words > 0:  # ignore erroneous labels
            output += "0" * (n_words-1)  # just a placeholder value
            if i < len(segments)-1:  # ignore last segment
                output += boundary

    if output.endswith(SPECIAL_BOUNDARY_TOKEN):  # remove break if it is the last token
        output = output[:-1]

    return output


def get_number_correct_boundaries(labels_list: List[int], model_labels: List[int]) -> int:
    return len(set(labels_list) & set(model_labels))  # and of sets gives the number of correct labels


def calc_precision(labels_list: List[int], model_labels: List[int]) -> float:
    # using same method as paper https://www.ijcai.org/proceedings/2018/0579.pdf
    correct = get_number_correct_boundaries(labels_list, model_labels)
    if len(model_labels) > 0:
        return correct / len(model_labels)
    else:
        if len(labels_list) == 0:
            return 1
        else:
            return 0


def calc_recall(labels_list: List[int], model_labels: List[int]) -> float:
    # using same method as paper https://www.ijcai.org/proceedings/2018/0579.pdf
    correct = get_number_correct_boundaries(labels_list, model_labels)
    if len(labels_list) > 0:
        return correct / len(labels_list)
    else:
        if len(model_labels) == 0:
            return 1
        else:
            return 0


def calc_f1(labels_list: List[int], model_labels: List[int]) -> float:
    # using same method as paper https://www.ijcai.org/proceedings/2018/0579.pdf
    correct = get_number_correct_boundaries(labels_list, model_labels)
    denominador = len(labels_list) + len(model_labels)
    if denominador > 0:
        return (2*correct) / denominador
    else:
        return 1
