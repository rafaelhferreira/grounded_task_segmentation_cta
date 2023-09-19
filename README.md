# Grounded Complex Task Segmentation for Conversational Assistants

This is the repository for the paper Grounded Complex Task Segmentation for Conversational Assistants published at SIGDIAL 2023 [here](https://sigdialinlg2023.github.io/paper_sigdial80.html).

## Getting Started

### Creating the Environment
If you use conda you can create the env using the **environment.yml** file
(depending on the hardware some versions might need to be different):

`conda env create -f environment.yml`

Activate environment with:

`conda activate task_segmentation`

### The ConvRecipes Dataset
The data used in this paper is available in the [data](data) folder.
There you can find the following files:
* **all_recipes_train.tsv** - training set consisting of all of the recipes (regardless of their conversational suitability).
* **test_set_annotations.tsv** - a file containing both the original test recipes as well as the annotation of the new segments.
* **train.tsv** - the training set used in the paper.
* **valid.tsv** - the validation set used in the paper.
* **test.tsv** - the test set used in the paper, that originates from test_set_annotations.tsv.

The dataset splits for training, validation and testing are available in the data folder.\
Each split is a .tsv file (tab separated file) with  5 columns: 
* **id** - int, representing the recipe's unique identifier
* **title** - str, title of the recipe
* **original_steps_list** - list of str, representing the original steps of the recipe
* **steps** - concatenation of the original_steps_list with some cleaning of extra spaces
* **labels_list** - list of int, representing the place where the text should be broken 
  with respect to the beginning of the recipe. The last break (last character of the recipe) is not represented since it is a trivial case.
  
You can train with your own data by providing a file with the same format.


## Task Segmentation Models
Models available:
* **cross_encoder** - methods based on a BERT cross-encoder model.
* **token_level** - methods based on models that consider token level features such as BERT and T5 (enc and enc-dec versions).
* **heuristic** - methods based on heuristic rules such as probability to break, and break every _n_ sentences.
* **text_tiling** - methods based on the TextTiling algorithm.


### Training and Evaluation
To train and evaluate the model use the [train_eval.py](train_eval.py) script.

For example to run the default BERT model:\
`python3 train_eval.py --model_type token_level`

You can also give a path to a json file with a model configuration: \
`python3 train_eval.py --model_type t5_encoder_token_level --config ./run_configurations/t5_encoder_token_level.json`

In the [run_configurations](./run_configurations) folder there are some example configurations.


In the end this creates in the provided **output_folder** a folder for the model checkpoints,
where each one has: config.json, pytorch .bin model, traininer_state.json and optimizer parameters, as explained in Huggingface's Trainer documentation.

Adding to the checkpoints the script also creates:
* **_config.json** - model configuration used to train the model which can then be used to load the model for other purposes.
* **_predictions.tsv** - the model predictions for the test set in the official format. 
* **_results.json** - metrics given by the official evaluation script.

### Creating your own Models

#### Using a Custom Configuration
You can create your own model by providing a different configuration json file. \
The parameters available for each model are a combination of the variables at: 
* [./python_src/models/model_utils.py](./python_src/models/model_utils.py) in the DefaultModelConfiguration class
* With a corresponding model architecture configuration class [./python_src/models](./python_src/models) (a class that inherits from DefaultModelConfiguration).


#### Creating a New Model from Scratch
To create a new architecture from scratch, i.e., one not present in the paper, you can follow a similar approach 
to the ones in [./python_src/models](./python_src/models):
1. Add the new model name configuration to the Enum class *TextSegmentationModels* in [./python_src/models/model_utils.py](./python_src/models/model_utils.py).
2. Create a new python file and create a class that inherits from *DefaultModelConfiguration* and add your new default values and/or new attributes.
3. Update the function *load_model_tokenizer*, *get_model_predictions*, *train_eval_model*, and *evaluate_model* if needed.

Or you can implement your model outside this codebase and use just the official evaluation script (Evaluating Models section of the README).

#### Running the Custom Configuration/Model
After having a custom configuration file or model run:\
`python3 train_eval.py --model_type <model_type> --config <path_to_config>`


### Testing Models
The training procedure in [train_eval.py](train_eval.py) in the end already performs the evaluation of the best model in the validation set, 
but if you just want to test the performance of a particular model in the test set you can use [test_model.py](test_model.py):\
`python3 test_model.py --model_type <model_type> --config <config> --model_location <model_location>`

E.g.
* Heuristic - `python3 test_model.py --model_type heuristic `
* TextTiling - `python3 test_model.py --model_type text_tiling `
* Token-Level - `python3 test_model.py --model_type token_level --config <model_config> --model_location <model_location>`

Like in [train_eval.py](train_eval.py) this script also creates:
* **_config.json** file, which can then be used to load the model for other purposes.
* **_predictions.tsv** file with the model predictions for the test set in the official format. 
* **_results.json** file with the metrics given by the official evaluation script.

#### Evaluating Models
To evaluate your model use the function evaluate_over_dataset in [./python_src/metrics.py](./python_src/metrics.py).
Alternatively you can use the [calculate_metrics.py](calculate_metrics.py) script:\
`python3 calculate_metrics.py --predictions <path to .tsv predictions file> --output_file <path to json file to store the results>`

The .tsv predictions file is a .tsv (tab separated) file with 5 columns as in the dataset: 
* id - int, representing the recipe's unique identifier
* title - str, title of the recipe
* original_steps_list - list of str, representing the original steps of the recipe
* steps - concatenation of the original_steps_list with some cleaning of extra spaces
* labels_list - list of int, representing the place where the model predicted the break 
  with respect to the beginning of the recipe. The last break (last character of the recipe) is not represented since it is a trivial case.
  
To evaluate, the only columns that need to be filled in the .tsv file are the id, steps, and labels_list, 
the rest must be there but can be left empty. 


## Citation
If you find it useful please cite our work:
```
@inproceedings{ferreira_task_segmentation,
  author       = {Rafael Ferreira and
                  David Semedo and
                  Joao Magalhaes},
  title        = {Grounded Complex Task Segmentation for Conversational Assistants},
  booktitle    = {Proceedings of the 24rd Annual Meeting of the Special Interest Group
                  on Discourse and Dialogue, {SIGDIAL} 2023, Prague, CZ, 13-15 September
                  2023},
  publisher    = {Association for Computational Linguistics},
  year         = {2023},
}
```
