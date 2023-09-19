import os
from typing import Dict, List, Union
from nltk.tokenize.texttiling import BLOCK_COMPARISON, DEFAULT_SMOOTHING, HC
import re
from nltk import TextTilingTokenizer
from tqdm import tqdm
from python_src.dataset.utils import class_attributes_to_dict, load_dataset_split, get_spacy_nlp, \
    write_configuration_to_file, write_split_to_file
from python_src.models.heuristic_baselines import print_and_write_official_metrics
from python_src.models.model_utils import TextSegmentationModels, set_seeds


class CustomTextTilingTokenizer(TextTilingTokenizer):
    # Custom TextTilingTokenizer to reduce the size of MIN_PARAGRAPH

    def __init__(self, w=20, k=10, similarity_method=BLOCK_COMPARISON, stopwords=None,
                 smoothing_method=DEFAULT_SMOOTHING, smoothing_width=2, smoothing_rounds=1, cutoff_policy=HC,
                 demo_mode=False):
        super().__init__(w, k, similarity_method, stopwords, smoothing_method, smoothing_width, smoothing_rounds,
                         cutoff_policy, demo_mode)

    def _mark_paragraph_breaks(self, text):
        """Identifies indented text or line breaks as the beginning of
        paragraphs"""
        MIN_PARAGRAPH = 5
        pattern = re.compile("[ \t\r\f\v]*\n[ \t\r\f\v]*\n[ \t\r\f\v]*")
        matches = pattern.finditer(text)

        last_break = 0
        pbreaks = [0]
        for pb in matches:
            if pb.start() - last_break < MIN_PARAGRAPH:
                continue
            else:
                pbreaks.append(pb.start())
                last_break = pb.start()

        return pbreaks


class TextTilingConfiguration:
    def __init__(self, configuration: Dict):
        self.text_segmentation_model_type = configuration.get("text_segmentation_model_type",
                                                              TextSegmentationModels.text_tiling.value)
        self.model_name = configuration.get("model_name", "text_tiling")

        self.w = configuration.get("w", 3)
        self.k = configuration.get("k", 2)

        self.use_small_dataset = configuration.get("use_small_dataset", False)

        self.output_dir = configuration.get("output_dir", './model_results/text_tiling')  # output directory

        self.seed = configuration.get("seed", 42)
        self.train_run_name = configuration.get("train_run_name", f"{self.model_name}_w_{self.w}_k_{self.k}")
        self.test_run_name = configuration.get("test_run_name", f"{self.model_name}_w_{self.w}_k_{self.k}" + "_test")
        self.train_number_samples = configuration.get("train_number_samples", 1000 if self.use_small_dataset else None)
        self.eval_number_samples = configuration.get("eval_number_samples", 100 if self.use_small_dataset else None)

        # datasets
        self.train_data_location = configuration.get("train_data_location", "./data/train.tsv")
        self.valid_data_location = configuration.get("valid_data_location", "./data/valid.tsv")
        self.test_data_location = configuration.get("test_data_location", "./data/test.tsv")


def evaluate_model(configuration: Dict, eval_number_samples: Union[int, None],
                   test_data_location: Union[str, None], suffix: str = "eval"):

    # get the model configuration
    model_configuration = TextTilingConfiguration(configuration)

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

    # turn the entire dataset into individual sentences per recipe
    print("Running spacy over examples...")
    spacy_nlp = get_spacy_nlp()
    text_tiling = CustomTextTilingTokenizer(w=model_configuration.w, k=model_configuration.k)
    model_output_labels = []
    for example in tqdm(test_examples_list[:model_configuration.eval_number_samples]):
        current_labels = []
        current_sentences = spacy_nlp(example).sents  # get spacy sentences
        current_sentences = [i.text.strip() for i in current_sentences]  # convert to str

        # when using text tiling algorithm it needs a \n\n at the end of each sentence not counting the last
        for i in range(len(current_sentences)-1):
            current_sentences[i] = current_sentences[i] + "\n\n"

        # print("current_sentences", len(current_sentences), current_sentences)

        try:
            tiling_sentences = text_tiling.tokenize(" ".join(current_sentences))  # type: List[str]
        except ValueError:  # when task is smaller than window
            tiling_sentences = [" ".join(current_sentences)]

        # print("tiling_sentences", len(tiling_sentences), tiling_sentences)

        # remove extra spaces and new lines and get the offset
        current_offset = 0
        for tiling_sentence in tiling_sentences[:-1]:
            current_offset += len(tiling_sentence.strip().replace("\n", ""))
            current_labels.append(current_offset)
            current_offset += 1

        model_output_labels.append(current_labels)

    if model_configuration.seed != 42:
        suffix = f"_seed_{model_configuration.seed}_{suffix}"

    write_split_to_file(test_ids_list, test_titles_list, original_test_examples,
                        test_examples_list, model_output_labels,
                        model_configuration.output_dir,
                        model_configuration.train_run_name + suffix + "_predictions")

    # print and write results to file
    print_and_write_official_metrics(model_configuration.test_data_location,
                                     model_configuration.train_run_name + suffix + "_predictions",
                                     model_configuration.output_dir, model_configuration.eval_number_samples)

    write_configuration_to_file(config_class=model_configuration,
                                output_path=os.path.join(model_configuration.output_dir,
                                                         model_configuration.test_run_name + suffix + "_config.json"))
