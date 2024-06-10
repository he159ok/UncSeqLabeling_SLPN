import itertools
import logging
import typing
import warnings
from abc import abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn
from torch.nn.modules.loss import _Loss
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

import flair
from flair import file_utils
from flair.data import DT, DT2, Dictionary, Sentence
from flair.datasets import DataLoader, FlairDatapointDataset
from flair.embeddings import Embeddings
from flair.file_utils import Tqdm
from flair.training_utils import Result, store_embeddings

import numpy as np
from flair.unc_eval.general_class_eval import *
from flair.unc_eval.ECE import ece_score


from scipy.special import softmax
from scipy.stats import entropy

# for drawing t-SNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import manifold
from datetime import datetime

log = logging.getLogger("flair")


class Model(torch.nn.Module, typing.Generic[DT]):
    """Abstract base class for all downstream task models in Flair,
    such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    model_card: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def label_type(self):
        """Each model predicts labels of a certain type.
        TODO: can we find a better name for this?"""
        raise NotImplementedError

    @abstractmethod
    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        """Performs a forward pass and returns a loss tensor for backpropagation.
        Implement this to enable training."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU  # noqa: E501
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        raise NotImplementedError

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        state_dict = {"state_dict": self.state_dict()}

        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        """Initialize the model from a state dictionary."""
        model = cls(**kwargs)

        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path], checkpoint: bool = False):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        # in Flair <0.9.1, optimizer and scheduler used to train model are not saved
        optimizer = scheduler = None

        # write out a "model card" if one is set
        if self.model_card is not None:

            # special handling for optimizer:
            # remember optimizer class and state dictionary
            if "training_parameters" in self.model_card:
                training_parameters = self.model_card["training_parameters"]

                if "optimizer" in training_parameters:
                    optimizer = training_parameters["optimizer"]
                    if checkpoint:
                        training_parameters["optimizer_state_dict"] = optimizer.state_dict()
                    training_parameters["optimizer"] = optimizer.__class__

                if "scheduler" in training_parameters:
                    scheduler = training_parameters["scheduler"]
                    if checkpoint:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            training_parameters["scheduler_state_dict"] = scheduler.state_dict()
                    training_parameters["scheduler"] = scheduler.__class__

            model_state["model_card"] = self.model_card

        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

        # restore optimizer and scheduler to model card if set
        if self.model_card is not None:
            if optimizer:
                self.model_card["training_parameters"]["optimizer"] = optimizer
            if scheduler:
                self.model_card["training_parameters"]["scheduler"] = scheduler

    @classmethod
    def load(cls, model_path: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model_path: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model_path))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround byhttps://github.com/highway11git
            # to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location="cpu")

        model = cls._init_model_with_state_dict(state)

        if "model_card" in state:
            model.model_card = state["model_card"]

        model.eval()
        model.to(flair.device)

        return model

    def print_model_card(self):
        if hasattr(self, "model_card"):
            param_out = "\n------------------------------------\n"
            param_out += "--------- Flair Model Card ---------\n"
            param_out += "------------------------------------\n"
            param_out += "- this Flair model was trained with:\n"
            param_out += f"-- Flair version {self.model_card['flair_version']}\n"
            param_out += f"-- PyTorch version {self.model_card['pytorch_version']}\n"
            if "transformers_version" in self.model_card:
                param_out += "-- Transformers version " f"{self.model_card['transformers_version']}\n"
            param_out += "------------------------------------\n"

            param_out += "------- Training Parameters: -------\n"
            param_out += "------------------------------------\n"
            training_params = "\n".join(
                f'-- {param} = {self.model_card["training_parameters"][param]}'
                for param in self.model_card["training_parameters"]
            )
            param_out += training_params + "\n"
            param_out += "------------------------------------\n"

            log.info(param_out)
        else:
            log.info(
                "This model has no model card (likely because it is not yet "
                "trained or was trained with Flair version < 0.9.1)"
            )


class Classifier(Model[DT], typing.Generic[DT]):
    """Abstract base class for all Flair models that do classification,
    both single- and multi-label. It inherits from flair.nn.Model and adds an
    unified evaluate() function so that all classification models use the same
    evaluation routines and compute the same numbers."""

    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}

            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    label_name="predicted",
                    return_loss=return_loss,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    # get the gold labels
                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value]
                        else:
                            all_predicted_values[representation].append(predicted_span.value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            for span in all_spans:
                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else ["O"]
                )

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "w", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O")
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == "O":
                continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            "\nResults:"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}"
            f"\n- Accuracy {accuracy_score}"
            "\n\nBy class:\n" + classification_report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        return result

    @abstractmethod
    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.  # noqa: E501
        """
        raise NotImplementedError

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            # check if there is a label mismatch
            g = [label.labeled_identifier for label in datapoint.get_labels(gold_label_type)]
            p = [label.labeled_identifier for label in datapoint.get_labels("predicted")]
            g.sort()
            p.sort()
            correct_string = " -> MISMATCH!\n" if g != p else ""
            # print info
            eval_line = (
                f"{datapoint.to_original_text()}\n"
                f" - Gold: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels(gold_label_type))}\n"
                f" - Pred: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels('predicted'))}\n{correct_string}\n"
            )
            lines.append(eval_line)
        return lines


class DefaultClassifier(Classifier[DT], typing.Generic[DT, DT2]):
    """Default base class for all Flair models that do classification, both
    single- and multi-label. It inherits from flair.nn.Classifier and thus from
    flair.nn.Model. All features shared by all classifiers are implemented here,
    including the loss calculation and the predict() method. Currently, the
    TextClassifier, RelationExtractor, TextPairClassifier and
    SimpleSequenceTagger implement this class.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        label_dictionary: Dictionary,
        final_embedding_size: int,
        dropout: float = 0.0,
        locked_dropout: float = 0.0,
        word_dropout: float = 0.0,
        multi_label: bool = False,
        multi_label_threshold: float = 0.5,
        loss_weights: Dict[str, float] = None,
        decoder: Optional[torch.nn.Module] = None,
        inverse_model: bool = False,
        train_on_gold_pairs_only: bool = False,
    ):

        super().__init__()

        # set the embeddings
        self.embeddings = embeddings

        # initialize the label dictionary
        self.label_dictionary: Dictionary = label_dictionary

        # initialize the decoder
        if decoder is not None:
            self.decoder = decoder
            self._custom_decoder = True
        else:
            self.decoder = torch.nn.Linear(final_embedding_size, len(self.label_dictionary))
            torch.nn.init.xavier_uniform_(self.decoder.weight)
            self._custom_decoder = False

        # set up multi-label logic
        self.multi_label = multi_label
        self.multi_label_threshold = multi_label_threshold
        self.final_embedding_size = final_embedding_size
        self.inverse_model = inverse_model

        # init dropouts
        self.dropout: torch.nn.Dropout = torch.nn.Dropout(dropout)
        self.locked_dropout = flair.nn.LockedDropout(locked_dropout)
        self.word_dropout = flair.nn.WordDropout(word_dropout)

        # loss weights and loss function
        self.weight_dict = loss_weights
        # Initialize the weight tensor
        if loss_weights is not None:
            n_classes = len(self.label_dictionary)
            weight_list = [1.0 for i in range(n_classes)]
            for i, tag in enumerate(self.label_dictionary.get_items()):
                if tag in loss_weights.keys():
                    weight_list[i] = loss_weights[tag]
            self.loss_weights: Optional[torch.Tensor] = torch.FloatTensor(weight_list).to(flair.device)
        else:
            self.loss_weights = None

        # set up gradient reversal if so specified
        if inverse_model:
            from pytorch_revgrad import RevGrad

            self.gradient_reversal = RevGrad()

        if self.multi_label:
            self.loss_function: _Loss = torch.nn.BCEWithLogitsLoss(weight=self.loss_weights, reduction="sum")
        else:
            self.loss_function = torch.nn.CrossEntropyLoss(weight=self.loss_weights, reduction="sum")
        self.train_on_gold_pairs_only = train_on_gold_pairs_only

    def _filter_data_point(self, data_point: DT) -> bool:
        """Specify if a data point should be kept. That way you can remove for example empty texts.
        Return true if the data point should be kept and false if it should be removed.
        """
        return True if len(data_point) > 0 else False

    @abstractmethod
    def _get_embedding_for_data_point(self, prediction_data_point: DT2) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def _get_data_points_from_sentence(self, sentence: DT) -> List[DT2]:
        """Returns the data_points to which labels are added (Sentence, Span, Token, ... objects)"""
        raise NotImplementedError

    def _get_data_points_for_batch(self, sentences: List[DT]) -> List[DT2]:
        """Returns the data_points to which labels are added (Sentence, Span, Token, ... objects)"""
        return [data_point for sentence in sentences for data_point in self._get_data_points_from_sentence(sentence)]

    def _get_label_of_datapoint(self, data_point: DT2) -> List[str]:
        """Extracts the labels from the data points.
        Each data point might return a list of strings, representing multiple labels.
        """
        if self.multi_label:
            return [label.value for label in data_point.get_labels(self.label_type)]
        else:
            return [data_point.get_label(self.label_type).value]

    @property
    def multi_label_threshold(self):
        return self._multi_label_threshold

    @multi_label_threshold.setter
    def multi_label_threshold(self, x):  # setter method
        if type(x) is dict:
            if "default" in x:
                self._multi_label_threshold = x
            else:
                raise Exception('multi_label_threshold dict should have a "default" key')
        else:
            self._multi_label_threshold = {"default": x}

    def _prepare_label_tensor(self, prediction_data_points: List[DT2]) -> torch.Tensor:
        labels = [self._get_label_of_datapoint(dp) for dp in prediction_data_points]
        if self.multi_label:
            return torch.tensor(
                [
                    [1 if label in all_labels_for_point else 0 for label in self.label_dictionary.get_items()]
                    for all_labels_for_point in labels
                ],
                dtype=torch.float,
                device=flair.device,
            )
        else:
            return torch.tensor(
                [
                    self.label_dictionary.get_idx_for_item(label[0])
                    if len(label) > 0
                    else self.label_dictionary.get_idx_for_item("O")
                    for label in labels
                ],
                dtype=torch.long,
                device=flair.device,
            )

    def _encode_data_points(self, sentences: List[DT], data_points: List[DT2]):

        # embed sentences
        self.embeddings.embed(sentences)

        # get a tensor of data points
        data_point_tensor = torch.stack([self._get_embedding_for_data_point(data_point) for data_point in data_points])

        # do dropout
        data_point_tensor = data_point_tensor.unsqueeze(1)
        data_point_tensor = self.dropout(data_point_tensor)
        data_point_tensor = self.locked_dropout(data_point_tensor)
        data_point_tensor = self.word_dropout(data_point_tensor)
        data_point_tensor = data_point_tensor.squeeze(1)

        return data_point_tensor

    def forward_loss(self, sentences: List[DT]) -> Tuple[torch.Tensor, int]:

        # make a forward pass to produce embedded data points and labels
        sentences = [sentence for sentence in sentences if self._filter_data_point(sentence)]

        # get the data points for which to predict labels
        data_points = self._get_data_points_for_batch(sentences)
        if len(data_points) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # get their gold labels as a tensor
        label_tensor = self._prepare_label_tensor(data_points)
        if label_tensor.size(0) == 0:
            return torch.tensor(0.0, requires_grad=True, device=flair.device), 1

        # pass data points through network to get encoded data point tensor
        data_point_tensor = self._encode_data_points(sentences, data_points)

        # decode
        scores = self.decoder(data_point_tensor)

        # calculate the loss
        return self._calculate_loss(scores, label_tensor)

    def _calculate_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, int]:
        return self.loss_function(scores, labels), labels.size(0)

    def _sort_data(self, data_points: List[DT]) -> List[DT]:

        if len(data_points) == 0:
            return []

        if not isinstance(data_points[0], Sentence):
            return data_points

        # filter empty sentences
        sentences = [sentence for sentence in typing.cast(List[Sentence], data_points) if len(sentence) > 0]

        # reverse sort all sequences by their length
        reordered_sentences = sorted(sentences, key=len, reverse=True)

        return typing.cast(List[DT], reordered_sentences)

    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively.  # noqa: E501
        'gpu' to store embeddings in GPU memory.
        """
        if label_name is None:
            label_name = self.label_type if self.label_type is not None else "label"

        with torch.no_grad():
            if not sentences:
                return sentences

            if not isinstance(sentences, list):
                sentences = [sentences]

            reordered_sentences = self._sort_data(sentences)

            if len(reordered_sentences) == 0:
                return sentences

            if len(reordered_sentences) > mini_batch_size:
                batches: Union[DataLoader, List[List[DT]]] = DataLoader(
                    dataset=FlairDatapointDataset(reordered_sentences),
                    batch_size=mini_batch_size,
                )
                # progress bar for verbosity
                if verbose:
                    progress_bar = tqdm(batches)
                    progress_bar.set_description("Batch inference")
                    batches = progress_bar
            else:
                batches = [reordered_sentences]

            overall_loss = torch.zeros(1, device=flair.device)
            label_count = 0
            for batch in batches:

                # filter data points in batch
                batch = [dp for dp in batch if self._filter_data_point(dp)]

                # stop if all sentences are empty
                if not batch:
                    continue

                data_points = self._get_data_points_for_batch(batch)

                if not data_points:
                    continue

                # pass data points through network and decode
                data_point_tensor = self._encode_data_points(batch, data_points)
                scores = self.decoder(data_point_tensor)

                # if anything could possibly be predicted
                if len(data_points) > 0:
                    # remove previously predicted labels of this type
                    for sentence in data_points:
                        sentence.remove_labels(label_name)

                    if return_loss:
                        gold_labels = self._prepare_label_tensor(data_points)
                        overall_loss += self._calculate_loss(scores, gold_labels)[0]
                        label_count += len(data_points)

                    if self.multi_label:
                        sigmoided = torch.sigmoid(scores)  # size: (n_sentences, n_classes)
                        n_labels = sigmoided.size(1)
                        for s_idx, data_point in enumerate(data_points):
                            for l_idx in range(n_labels):
                                label_value = self.label_dictionary.get_item_for_index(l_idx)
                                if label_value == "O":
                                    continue
                                label_threshold = self._get_label_threshold(label_value)
                                label_score = sigmoided[s_idx, l_idx].item()
                                if label_score > label_threshold or return_probabilities_for_all_classes:
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)
                    else:
                        softmax = torch.nn.functional.softmax(scores, dim=-1)

                        if return_probabilities_for_all_classes:
                            n_labels = softmax.size(1)
                            for s_idx, data_point in enumerate(data_points):
                                for l_idx in range(n_labels):
                                    label_value = self.label_dictionary.get_item_for_index(l_idx)
                                    if label_value == "O":
                                        continue
                                    label_score = softmax[s_idx, l_idx].item()
                                    data_point.add_label(typename=label_name, value=label_value, score=label_score)
                        else:
                            conf, idx = torch.max(softmax, dim=-1)
                            for data_point, c, i in zip(data_points, conf, idx):
                                label_value = self.label_dictionary.get_item_for_index(i.item())
                                if label_value == "O":
                                    continue
                                data_point.add_label(typename=label_name, value=label_value, score=c.item())

                store_embeddings(batch, storage_mode=embedding_storage_mode)

            if return_loss:
                return overall_loss, label_count

    def _get_label_threshold(self, label_value):
        label_threshold = self.multi_label_threshold["default"]
        if label_value in self.multi_label_threshold:
            label_threshold = self.multi_label_threshold[label_value]

        return label_threshold

    def __str__(self):
        return (
            super(flair.nn.Model, self).__str__().rstrip(")")
            + f"  (weights): {self.weight_dict}\n"
            + f"  (weight_tensor) {self.loss_weights}\n)"
        )

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        # add DefaultClassifier arguments
        for arg in [
            "decoder",
            "dropout",
            "word_dropout",
            "locked_dropout",
            "multi_label",
            "multi_label_threshold",
            "loss_weights",
            "train_on_gold_pairs_only",
            "inverse_model",
        ]:
            if arg not in kwargs and arg in state:
                kwargs[arg] = state[arg]

        return super(Classifier, cls)._init_model_with_state_dict(state, **kwargs)

    def _get_state_dict(self):
        state = super()._get_state_dict()

        # add variables of DefaultClassifier
        state["dropout"] = self.dropout.p
        state["word_dropout"] = self.word_dropout.dropout_rate
        state["locked_dropout"] = self.locked_dropout.dropout_rate
        state["multi_label"] = self.multi_label
        state["multi_label_threshold"] = self.multi_label_threshold
        state["loss_weights"] = self.loss_weights
        state["train_on_gold_pairs_only"] = self.train_on_gold_pairs_only
        state["inverse_model"] = self.inverse_model
        if self._custom_decoder:
            state["decoder"] = self.decoder

        return state


#### below is self-modified
class UncModel(torch.nn.Module, typing.Generic[DT]):
    """Abstract base class for all downstream task models in Flair,
    such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""

    model_card: Optional[Dict[str, Any]] = None

    @property
    @abstractmethod
    def label_type(self):
        """Each model predicts labels of a certain type.
        TODO: can we find a better name for this?"""
        raise NotImplementedError

    @abstractmethod
    def forward_loss(self, data_points: List[DT]) -> Tuple[torch.Tensor, int]:
        """Performs a forward pass and returns a loss tensor for backpropagation.
        Implement this to enable training."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        **kwargs,
    ) -> Result:
        """Evaluates the model. Returns a Result object containing evaluation
        results and a loss value. Implement this to enable evaluation.
        :param data_loader: DataLoader that iterates over dataset to be evaluated
        :param out_path: Optional output path to store predictions
        :param embedding_storage_mode: One of 'none', 'cpu' or 'gpu'. 'none' means all embeddings are deleted and freshly recomputed, 'cpu' means all embeddings are stored on CPU, or 'gpu' means all embeddings are stored on GPU  # noqa: E501
        :return: Returns a Tuple consisting of a Result object and a loss float value
        """
        raise NotImplementedError

    def _get_state_dict(self):
        """Returns the state dictionary for this model."""
        state_dict = {"state_dict": self.state_dict()}

        return state_dict

    @classmethod
    def _init_model_with_state_dict(cls, state, **kwargs):
        """Initialize the model from a state dictionary."""
        model = cls(**kwargs)

        model.load_state_dict(state["state_dict"])
        return model

    @staticmethod
    def _fetch_model(model_name) -> str:
        return model_name

    def save(self, model_file: Union[str, Path], checkpoint: bool = False):
        """
        Saves the current model to the provided file.
        :param model_file: the model file
        """
        model_state = self._get_state_dict()

        # in Flair <0.9.1, optimizer and scheduler used to train model are not saved
        optimizer = scheduler = None

        # write out a "model card" if one is set
        if self.model_card is not None:

            # special handling for optimizer:
            # remember optimizer class and state dictionary
            if "training_parameters" in self.model_card:
                training_parameters = self.model_card["training_parameters"]

                if "optimizer" in training_parameters:
                    optimizer = training_parameters["optimizer"]
                    if checkpoint:
                        training_parameters["optimizer_state_dict"] = optimizer.state_dict()
                    training_parameters["optimizer"] = optimizer.__class__

                if "scheduler" in training_parameters:
                    scheduler = training_parameters["scheduler"]
                    if checkpoint:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            training_parameters["scheduler_state_dict"] = scheduler.state_dict()
                    training_parameters["scheduler"] = scheduler.__class__

            model_state["model_card"] = self.model_card

        # save model
        torch.save(model_state, str(model_file), pickle_protocol=4)

        # restore optimizer and scheduler to model card if set
        if self.model_card is not None:
            if optimizer:
                self.model_card["training_parameters"]["optimizer"] = optimizer
            if scheduler:
                self.model_card["training_parameters"]["scheduler"] = scheduler

    @classmethod
    def load(cls, model_path: Union[str, Path]):
        """
        Loads the model from the given file.
        :param model_path: the model file
        :return: the loaded text classifier model
        """
        model_file = cls._fetch_model(str(model_path))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            # load_big_file is a workaround byhttps://github.com/highway11git
            # to load models on some Mac/Windows setups
            # see https://github.com/zalandoresearch/flair/issues/351
            f = file_utils.load_big_file(str(model_file))
            state = torch.load(f, map_location="cpu")

        model = cls._init_model_with_state_dict(state)

        if "model_card" in state:
            model.model_card = state["model_card"]

        model.eval()
        model.to(flair.device)

        return model

    def print_model_card(self):
        if hasattr(self, "model_card"):
            param_out = "\n------------------------------------\n"
            param_out += "--------- Flair Model Card ---------\n"
            param_out += "------------------------------------\n"
            param_out += "- this Flair model was trained with:\n"
            param_out += f"-- Flair version {self.model_card['flair_version']}\n"
            param_out += f"-- PyTorch version {self.model_card['pytorch_version']}\n"
            if "transformers_version" in self.model_card:
                param_out += "-- Transformers version " f"{self.model_card['transformers_version']}\n"
            param_out += "------------------------------------\n"

            param_out += "------- Training Parameters: -------\n"
            param_out += "------------------------------------\n"
            training_params = "\n".join(
                f'-- {param} = {self.model_card["training_parameters"][param]}'
                for param in self.model_card["training_parameters"]
            )
            param_out += training_params + "\n"
            param_out += "------------------------------------\n"

            log.info(param_out)
        else:
            log.info(
                "This model has no model card (likely because it is not yet "
                "trained or was trained with Flair version < 0.9.1)"
            )


class UncClassifier(UncModel[DT], typing.Generic[DT]):
    """Abstract base class for all Flair models that do classification,
    both single- and multi-label. It inherits from flair.nn.Model and adds an
    unified evaluate() function so that all classification models use the same
    evaluation routines and compute the same numbers."""

    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        use_alpha: bool = False,
        sfmx_mode: str = None,
        uncertainty_metrics: List[str]=[],
        cal_unique_predict_scores: bool = False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)
        if use_alpha == False:
            default_score_dict = self.get_default_score_dict(interv_dict)
        else:
            default_score_dict = self.get_default_alpha_dict(interv_dict)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            all_predicted_score_dict = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    return_probabilities_for_all_classes=True, # self-added
                    label_name="predicted",
                    return_loss=return_loss,
                    use_alpha=use_alpha,
                    sfmx_mode=sfmx_mode,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    # get the gold labels
                    if cal_unique_predict_scores: unique_gold_representation_set = set()

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass
                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        if cal_unique_predict_scores:
                            if representation in unique_gold_representation_set:
                                unique_gold_representation_set.remove(representation)

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value]
                            all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]

                        else:
                            all_predicted_values[representation].append(predicted_span.value)
                            all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True))

                        if representation not in all_spans:
                            all_spans.add(representation)

                    ### calculate the unique predicted entities score vec accroding to ground-truth index
                    if cal_unique_predict_scores:
                        for rep in unique_gold_representation_set:
                            if rep not in unique_gt_in_pd_score_dict.keys():
                                unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)]
                            else:
                                unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)])


                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            predicted_scoredict_span_aligned = [] # self-added
            for span in all_spans:
                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"] # self-comment: need change for OOD detection metric
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else ["O"] # self-comment: need change for OOD detection metric
                )

                if cal_unique_predict_scores:
                    if span in all_predicted_values:
                        to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
                    else:
                        to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
                    predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
                else:
                    predicted_scoredict_span_aligned.append(
                        all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
                    )
                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                  # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O") # self-comment: need change for OOD detection metric
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        if use_alpha == False:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]
        else:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]



        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == "O":
                continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        # self-added: calculate unc metrics for the span-level uncertainty
        # cal unc_scores


        # add dissonance, vaculity, entropy, epistemic at here
        # use [dissonance, vaculity, entropy, epistemic]

        # max_unc = [
        #    1.0/(max(ele)+1e-8) for ele in y_pred_vec_alpha_or_sfmx
        # ]
        # enpy_unc = [
        #    entropy(ele) for ele in y_pred_vec_alpha_or_sfmx
        # ]
        #
        # auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score = show_roc_auc(y_true, y_pred, enpy_unc)
        # print(f"entropy_unc auroc & aupr are {auroc_score}, {aupr_score}")
        # writent_list.append("\n")
        # writent_list.append(f"entropy_unc_auroc is {auroc_score}\n")
        # writent_list.append(f"entropy_unc_aupr is  {aupr_score}\n")
        #
        # auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score = show_roc_auc(y_true, y_pred, max_unc)
        # print(f"max_unc auroc & aupr are {auroc_score}, {aupr_score}")
        # writent_list.append(f"max_unc_auroc is {auroc_score}\n")
        # writent_list.append(f"max_unc_aupr is  {aupr_score}\n")
        #
        # ece_res = ece_score(np.array(y_pred_vec_alpha_or_sfmx), y_true, n_bins=10)
        # print("ece is ", ece_res)
        # writent_list.append(f"ece is , {ece_res}\n")
        # writent_list.append("\n")

        if use_alpha == True:
            if 'diss' in uncertainty_metrics:
                diss_unc = dissonance_uncertainty(y_pred_vec_alpha_or_sfmx)
                diss_auroc, diss_aupr, diss_wrlist = get_auroc_aupr(y_true, y_pred, diss_unc, keyword='diss')
                writent_list.extend(diss_wrlist)
            if 'vacu' in uncertainty_metrics:
                vacu_unc = vacuity_uncertainty(y_pred_vec_alpha_or_sfmx)
                vacu_auroc, vacu_aupr, vacu_wrlist = get_auroc_aupr(y_true, y_pred, vacu_unc, keyword='vacu')
                writent_list.extend(vacu_wrlist)
            if 'entr_mean' in uncertainty_metrics: # only applied to alpha form
                entr_unc = get_un_entropy(y_pred_vec_alpha_or_sfmx, mode='mean')
                entr_auroc, entr_aupr, entr_wrlist = get_auroc_aupr(y_true, y_pred, entr_unc, keyword='entr_mean')
                writent_list.extend(entr_wrlist)
            if 'epis' in uncertainty_metrics:
                # for misclassification
                epis_unc = one_over_max(y_pred_vec_alpha_or_sfmx)
                epis_auroc, epis_aupr, epis_wrlist = get_auroc_aupr(y_true, y_pred, epis_unc, keyword='epis')
                writent_list.extend(epis_wrlist)
                # for OOD detection -> use sum of alpha

            if 'alea' in uncertainty_metrics:
                raise ValueError('alea in use_apha=True is not supported')
            if 'entr_sfmx' in uncertainty_metrics:
                raise ValueError('entr_sfmx in use_apha=True is not supported')


        else:
            # below is for use_alpha == False -> use sfmx of probability
            if 'epis' in uncertainty_metrics:
                raise ValueError('epis in use_apha=True is not supported')
            if 'entr_mean' in uncertainty_metrics:
                raise ValueError('entr_mean in use_apha=True is not supported')
            if 'alea' in uncertainty_metrics:
                # for misclassification
                alea_unc = one_over_max(y_pred_vec_alpha_or_sfmx)
                alea_auroc, epis_aupr, epis_wrlist = get_auroc_aupr(y_true, y_pred, alea_unc, keyword='alea_'+sfmx_mode)
                writent_list.extend(epis_wrlist)
            if 'entr_sfmx' in uncertainty_metrics: # only applied to non-alpha form
                entr_unc = get_un_entropy(y_pred_vec_alpha_or_sfmx, mode='sfmx')
                entr_auroc, entr_aupr, entr_wrlist = get_auroc_aupr(y_true, y_pred, entr_unc, keyword='entr_sfmx')


        # write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'
        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        # print(str(out_path.absolute())[:-8] + 'unc.txt')
        print(write_file_name)




        return result


    # this is used for the ood testing process only;
    # for the ood validation process, please keep using the traditional
    def ood_evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        use_alpha: bool = False,
        sfmx_mode: str = None,
        uncertainty_metrics: List[str]=[],
        leave_out_labels: List[str]=[],
        cal_unique_predict_scores: bool = True,
        use_var_metric=False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        # required for OOD detection
        assert cal_unique_predict_scores==True
        self.use_var_metric=use_var_metric
        self.shared_entity_num = 0
        self.unique_gt_entity_num=0

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)
        if use_alpha == False:
            default_score_dict = self.get_default_score_dict(interv_dict) # this is span-level data
        else:
            default_score_dict = self.get_default_alpha_dict(interv_dict)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            cls_all_true_values = {}
            cls_all_predicted_values = {}
            all_predicted_score_dict = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
            ID_label, OOD_label = 'ID_label', 'OOD_label' # OOD and ID labels definition
            wrong_span_label = 'wrong_span_label'
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    return_probabilities_for_all_classes=True, # self-added
                    label_name="predicted",
                    return_loss=return_loss,
                    use_alpha=use_alpha,
                    sfmx_mode=sfmx_mode,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    #### （to do）generate binary ood labels for the ood evaluation
                    # get the gold labels
                    if cal_unique_predict_scores: unique_gold_representation_set = set()

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        cls_value = gold_label.value ## different from misclassification
                        if gold_label.value != 'OOD':
                            value = ID_label
                        else:
                            value = OOD_label

                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                            cls_all_true_values[representation] = [cls_value]
                        else:
                            all_true_values[representation].append(value)
                            cls_all_true_values[representation].append(cls_value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass
                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        if cal_unique_predict_scores:
                            if representation in unique_gold_representation_set:
                                unique_gold_representation_set.remove(representation)

                        # if predicted_span.value != 'OOD':
                        #     predicted_span_value = ID_label
                        # else:
                        #     predicted_span_value = OOD_label
                        cls_predicted_span_value = predicted_span.value
                        predicted_span_value = ID_label  # predicted_span.value

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span_value]
                            all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                            cls_all_predicted_values[representation] = [cls_predicted_span_value]
                        else:
                            all_predicted_values[representation].append(predicted_span_value)
                            all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True))
                            cls_all_predicted_values[representation].append(cls_predicted_span_value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    ### calculate the unique predicted entities score vec accroding to ground-truth index
                    if cal_unique_predict_scores:
                        for rep in unique_gold_representation_set:
                            if rep not in unique_gt_in_pd_score_dict.keys():
                                unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)]
                            else:
                                unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)])

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            predicted_scoredict_span_aligned = [] # self-added
            cls_true_values_span_aligned = []
            cls_predicted_values_span_aligned = []
            # (to do) generate label list: cls_true_values_span_aligned, cls_predicted_values_span_aligned
            for span in all_spans:
                # self-added: count shared_num
                if span in all_true_values and span in all_predicted_values:
                    self.shared_entity_num += 1
                if span in all_true_values and span not in all_predicted_values:
                    self.unique_gt_entity_num += 1

                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else [wrong_span_label] # self-comment: need change for OOD detection metric
                cls_list_of_gold_values_for_span = cls_all_true_values[span] if span in cls_all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                        cls_list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                cls_true_values_span_aligned.append(cls_list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else [wrong_span_label] # self-comment: need change for OOD detection metric
                )
                cls_predicted_values_span_aligned.append(
                    cls_all_predicted_values[span] if span in cls_all_predicted_values else ["O"]
                )

                if cal_unique_predict_scores:   #### related to case study
                    if span in all_predicted_values:
                        to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
                    else:
                        to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
                    predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
                else:
                    predicted_scoredict_span_aligned.append(
                        all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
                    )
                # O： 1.0 Loc： 1.0 Per: 1.0 Event 1.0
                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item(wrong_span_label) # self-comment: need change for OOD detection metric
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords

            default_evaluation_label_dictionary = Dictionary(add_unk=False)
            for label in default_score_dict.keys():
                default_evaluation_label_dictionary.add_item(label) # (check) expect default_evaluation_label_dictionary in 1 + c keywords

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation

            # # below is for the OOD ture pred,
            # # (to do) pack to function
            # ood_y_true = []
            # ood_y_pred = []
            # ood_y_pred_vec_alpha_or_sfmx = []
            # for ind in range(len(true_values_span_aligned)):
            #     ele = true_values_span_aligned[ind]
            #     predicted_scoredict_ins = predicted_scoredict_span_aligned[ind]
            #     if ele[0] != wrong_span_label:
            #         ood_y_true.append(evaluation_label_dictionary.get_idx_for_item(true_values_span_aligned[ind][0]))
            #         ood_y_pred.append(evaluation_label_dictionary.get_idx_for_item(predicted_values_span_aligned[ind][0]))
            #         if use_alpha == False:
            #             ood_y_pred_vec_alpha_or_sfmx.append(self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none'))
            #         else:
            #             ood_y_pred_vec_alpha_or_sfmx.append(self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none'))

            ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
                true_values_span_aligned=true_values_span_aligned,
                predicted_values_span_aligned=predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=wrong_span_label, # in ood setting, we need to remove the wrong_span_label
            )


            # below is for the wrong_span ture pred,
            ws_y_true, ws_y_pred, ws_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
                true_values_span_aligned=true_values_span_aligned,
                predicted_values_span_aligned=predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=None, # in ood setting, we need to remove the wrong_span_label
            )

            # below is for the in-domain c-level semantic class ture pred,
            cls_y_true, cls_y_pred, cls_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
                true_values_span_aligned=cls_true_values_span_aligned,
                predicted_values_span_aligned=cls_predicted_values_span_aligned,
                evaluation_label_dictionary=default_evaluation_label_dictionary, # changed into c-class dict
                predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label='OOD', # in ood setting, we need to remove the wrong_span_label
            )


        if use_alpha == False:
            # y_pred_vec_alpha_or_sfmx = [
            #     self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
            #     for predicted_scoredict_ins in predicted_scoredict_span_aligned
            # ]
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]

        else:
        #     y_pred_vec_alpha_or_sfmx = [
        #         self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
        #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     ]
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]



        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(cls_all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(cls_all_predicted_values.values())))

        for label_name, count in counter.most_common(): # (might need to change for the ID classification task)
            # if label_name == "O":
            # if label_name == wrong_span_label:
            if label_name == OOD_label or label_name == 'OOD':
                continue
            target_names.append(label_name)
            # labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))
            labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))


        ### (to do) skip the classification process
        # there is at least one gold label or one prediction (default)
        if len(cls_all_true_values) + len(cls_all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                cls_y_true, # [0, 1, 2] 2: OOD_label -> [0, 1] -> transfer -> c 个语义类
                cls_y_pred, # [0, 1]
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(cls_y_true, cls_y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        ood_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx=ood_y_pred_vec_alpha_or_sfmx,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='ood_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ood_unc_list)

        ws_unc_list = self.get_unc_score_list(
            y_true=ws_y_true,
            y_pred=ws_y_pred,
            y_pred_vec_alpha_or_sfmx=ws_y_pred_vec_alpha_or_sfmx,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='wrong_span_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(wrong_span_label)
        )
        writent_list.extend(ws_unc_list)




        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        # print(str(out_path.absolute())[:-8] + 'unc.txt')
        print(write_file_name)




        return result

    def ood_evaluate_falsepos(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        use_alpha: bool = False,
        sfmx_mode: str = None,
        uncertainty_metrics: List[str]=[],
        leave_out_labels: List[str]=[],
        cal_unique_predict_scores: bool = True,
        use_var_metric=False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        # required for OOD detection
        assert cal_unique_predict_scores==True
        self.use_var_metric=use_var_metric
        self.shared_entity_num = 0
        self.unique_gt_entity_num=0

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)
        if use_alpha == False:
            default_score_dict = self.get_default_score_dict(interv_dict) # this is span-level data
        else:
            default_score_dict = self.get_default_alpha_dict(interv_dict)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            cls_all_true_values = {}
            cls_all_predicted_values = {}
            all_predicted_score_dict = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
            ID_label, OOD_label = 'ID_label', 'OOD_label' # OOD and ID labels definition
            wrong_span_label = 'wrong_span_label'
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    return_probabilities_for_all_classes=True, # self-added
                    label_name="predicted",
                    return_loss=return_loss,
                    use_alpha=use_alpha,
                    sfmx_mode=sfmx_mode,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    #### （to do）generate binary ood labels for the ood evaluation
                    # get the gold labels
                    if cal_unique_predict_scores: unique_gold_representation_set = set()

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        cls_value = gold_label.value ## different from misclassification
                        if gold_label.value != 'OOD':
                            value = ID_label
                        else:
                            value = OOD_label

                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                            cls_all_true_values[representation] = [cls_value]
                        else:
                            all_true_values[representation].append(value)
                            cls_all_true_values[representation].append(cls_value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass
                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        if cal_unique_predict_scores:
                            if representation in unique_gold_representation_set:
                                unique_gold_representation_set.remove(representation)

                        # if predicted_span.value != 'OOD':
                        #     predicted_span_value = ID_label
                        # else:
                        #     predicted_span_value = OOD_label
                        cls_predicted_span_value = predicted_span.value
                        predicted_span_value = ID_label  # predicted_span.value

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span_value]
                            all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                            cls_all_predicted_values[representation] = [cls_predicted_span_value]
                        else:
                            all_predicted_values[representation].append(predicted_span_value)
                            all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True))
                            cls_all_predicted_values[representation].append(cls_predicted_span_value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    ### calculate the unique predicted entities score vec accroding to ground-truth index
                    if cal_unique_predict_scores:
                        for rep in unique_gold_representation_set:
                            if rep not in unique_gt_in_pd_score_dict.keys():
                                unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)]
                            else:
                                unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)])

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            predicted_scoredict_span_aligned = [] # self-added
            cls_true_values_span_aligned = []
            cls_predicted_values_span_aligned = []
            # (to do) generate label list: cls_true_values_span_aligned, cls_predicted_values_span_aligned
            for span in all_spans:
                # self-added: count shared_num
                if span in all_true_values and span in all_predicted_values:
                    self.shared_entity_num += 1
                if span in all_true_values and span not in all_predicted_values:
                    self.unique_gt_entity_num += 1

                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else [wrong_span_label] # self-comment: actual wrong span
                cls_list_of_gold_values_for_span = cls_all_true_values[span] if span in cls_all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                        cls_list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                cls_true_values_span_aligned.append(cls_list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else [wrong_span_label] # self-comment: not actual ones, but the part unique in the gt
                )
                cls_predicted_values_span_aligned.append(
                    cls_all_predicted_values[span] if span in cls_all_predicted_values else ["O"]
                )

                if cal_unique_predict_scores:
                    if span in all_predicted_values:
                        to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
                    else:
                        to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
                    predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
                else:
                    predicted_scoredict_span_aligned.append(
                        all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
                    )
                # O： 1.0 Loc： 1.0 Per: 1.0 Event 1.0
                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item(wrong_span_label) # self-comment: need change for OOD detection metric
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords

            default_evaluation_label_dictionary = Dictionary(add_unk=False)
            for label in default_score_dict.keys():
                default_evaluation_label_dictionary.add_item(label) # (check) expect default_evaluation_label_dictionary in 1 + c keywords

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation

            # # below is for the OOD ture pred,
            # # (to do) pack to function
            # ood_y_true = []
            # ood_y_pred = []
            # ood_y_pred_vec_alpha_or_sfmx = []
            # for ind in range(len(true_values_span_aligned)):
            #     ele = true_values_span_aligned[ind]
            #     predicted_scoredict_ins = predicted_scoredict_span_aligned[ind]
            #     if ele[0] != wrong_span_label:
            #         ood_y_true.append(evaluation_label_dictionary.get_idx_for_item(true_values_span_aligned[ind][0]))
            #         ood_y_pred.append(evaluation_label_dictionary.get_idx_for_item(predicted_values_span_aligned[ind][0]))
            #         if use_alpha == False:
            #             ood_y_pred_vec_alpha_or_sfmx.append(self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none'))
            #         else:
            #             ood_y_pred_vec_alpha_or_sfmx.append(self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none'))

            # process to transfer ood+ws -> ood
            # true: ws -> ood
            # predict: ws -> id
            fp_true_values_span_aligned = []
            fp_predicted_values_span_aligned = []
            for ele in true_values_span_aligned:
                mid_res = []
                for sub_ele in ele:
                    if sub_ele == wrong_span_label:
                        mid_res.append(OOD_label)
                    else:
                        mid_res.append(sub_ele)
                fp_true_values_span_aligned.append(mid_res)
            for ele in predicted_values_span_aligned:
                mid_res = []
                for sub_ele in ele:
                    if sub_ele == wrong_span_label:
                        mid_res.append(ID_label)
                    else:
                        mid_res.append(sub_ele)
                fp_predicted_values_span_aligned.append(mid_res)


            ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
                true_values_span_aligned=fp_true_values_span_aligned, # true_values_span_aligned
                predicted_values_span_aligned=fp_predicted_values_span_aligned, # predicted_values_span_aligned
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=None, # different from ood_evaluate, here set the move_out_label as None, instead of wrong_span_label
            )


            # below is for the wrong_span ture pred,
            ws_y_true, ws_y_pred, ws_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
                true_values_span_aligned=true_values_span_aligned,
                predicted_values_span_aligned=predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=None, # in ood setting, we need to remove the wrong_span_label
            )

            # below is for the in-domain c-level semantic class ture pred,
            cls_y_true, cls_y_pred, cls_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
                true_values_span_aligned=cls_true_values_span_aligned,
                predicted_values_span_aligned=cls_predicted_values_span_aligned,
                evaluation_label_dictionary=default_evaluation_label_dictionary, # changed into c-class dict
                predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label='OOD', # in ood setting, we need to remove the wrong_span_label
            )


        if use_alpha == False:
            # y_pred_vec_alpha_or_sfmx = [
            #     self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
            #     for predicted_scoredict_ins in predicted_scoredict_span_aligned
            # ]
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]

        else:
        #     y_pred_vec_alpha_or_sfmx = [
        #         self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
        #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     ]
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]



        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(cls_all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(cls_all_predicted_values.values())))

        for label_name, count in counter.most_common(): # (might need to change for the ID classification task)
            # if label_name == "O":
            # if label_name == wrong_span_label:
            if label_name == OOD_label or label_name == 'OOD':
                continue
            target_names.append(label_name)
            # labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))
            labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))


        ### (to do) skip the classification process
        # there is at least one gold label or one prediction (default)
        if len(cls_all_true_values) + len(cls_all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                cls_y_true, # [0, 1, 2] 2: OOD_label -> [0, 1] -> transfer -> c 个语义类
                cls_y_pred, # [0, 1]
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(cls_y_true, cls_y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        ood_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx=ood_y_pred_vec_alpha_or_sfmx,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='ood_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ood_unc_list)

        ws_unc_list = self.get_unc_score_list(
            y_true=ws_y_true,
            y_pred=ws_y_pred,
            y_pred_vec_alpha_or_sfmx=ws_y_pred_vec_alpha_or_sfmx,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='wrong_span_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(wrong_span_label)
        )
        writent_list.extend(ws_unc_list)




        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        # print(str(out_path.absolute())[:-8] + 'unc.txt')
        print(write_file_name)




        return result

    def ood_evaluate_token(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        use_alpha: bool = False,
        sfmx_mode: str = None,
        uncertainty_metrics: List[str]=[],
        leave_out_labels: List[str]=[],
        cal_unique_predict_scores: bool = True,
        use_var_metric=False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        # required for OOD detection
        assert cal_unique_predict_scores==True
        self.use_var_metric=use_var_metric
        self.shared_entity_num = 0
        self.unique_gt_entity_num=0

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)
        if use_alpha == False:
            default_score_dict = self.get_default_score_dict(interv_dict) # this is span-level data
        else:
            default_score_dict = self.get_default_alpha_dict(interv_dict)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            cls_all_true_values = {}
            cls_all_predicted_values = {}
            all_predicted_score_dict = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
            ID_label, OOD_label = 'ID_label', 'OOD_label' # OOD and ID labels definition
            wrong_span_label = 'wrong_span_label'
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            # token:
            ood_y_true, ood_y_pred = [], []
            cls_y_true, cls_y_pred = [], []
            ood_y_pred_vec_alpha_or_sfmx = []


            sentence_id = 0
            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                # predict for batch
                loss_and_count = self.predict(
                    batch,
                    embedding_storage_mode=embedding_storage_mode,
                    mini_batch_size=mini_batch_size,
                    return_probabilities_for_all_classes=True, # self-added
                    label_name="predicted",
                    return_loss=return_loss,
                    use_alpha=use_alpha,
                    sfmx_mode=sfmx_mode,
                )

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    #### （to do）generate binary ood labels for the ood evaluation
                    # get the gold labels
                    # if cal_unique_predict_scores: unique_gold_representation_set = set()

                    # token: initializi the vectors
                    token_num = len(datapoint)
                    mid_ood_y_true = [ID_label] * token_num
                    mid_ood_y_pred = [ID_label] * token_num
                    mid_cls_y_true = ['O'] * token_num
                    mid_cls_y_pred = ['O'] * token_num

                    for gold_label in datapoint.get_labels(gold_label_type):
                        # token:
                        total_representation, total_index = self.get_token_info(datapoint, sentence_id, gold_label.unlabeled_identifier)
                        for index in range(len(total_index)):
                            representation = str(sentence_id) + ": " + total_representation[index]
                            y_index = total_index[index]

                            # if cal_unique_predict_scores: unique_gold_representation_set.add(representation) # might be usefulness in token-level

                            mid_cls_y_true[y_index] = gold_label.value ## different from misclassification
                            if gold_label.value != 'OOD':
                                mid_ood_y_true[y_index] = ID_label
                            else:
                                mid_ood_y_true[y_index] = OOD_label

                            # if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            #     mid_ood_y_true[index] = "<unk>"
                            #
                            # if representation not in all_true_values:
                            #     all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                            #     cls_all_true_values[representation] = [cls_value]
                            # else:
                            #     all_true_values[representation].append(value)
                            #     cls_all_true_values[representation].append(cls_value)
                            #
                            # if representation not in all_spans:
                            #     all_spans.add(representation)

                    ood_y_true.extend(mid_ood_y_true)
                    cls_y_true.extend(mid_cls_y_true)

                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass
                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        # token:
                        total_representation, total_index = self.get_token_info(datapoint, sentence_id,
                                                                                           predicted_span.unlabeled_identifier)

                        for index in range(len(total_index)):
                            representation = str(sentence_id) + ": " + total_representation[index]
                            y_index = total_index[index]

                            # remove the existing representation, if the representation is not unique in the ground-truth
                            # if cal_unique_predict_scores:
                            #     if representation in unique_gold_representation_set:
                            #         unique_gold_representation_set.remove(representation)

                            # if predicted_span.value != 'OOD':
                            #     predicted_span_value = ID_label
                            # else:
                            #     predicted_span_value = OOD_label
                            mid_cls_y_pred[y_index] = predicted_span.value
                            mid_ood_y_pred[y_index] = ID_label


                            # # add to all_predicted_values
                            # if representation not in all_predicted_values:
                            #     all_predicted_values[representation] = [predicted_span_value]
                            #     all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                            #     cls_all_predicted_values[representation] = [cls_predicted_span_value]
                            # else:
                            #     all_predicted_values[representation].append(predicted_span_value)
                            #     all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True))
                            #     cls_all_predicted_values[representation].append(cls_predicted_span_value)
                            #
                            # if representation not in all_spans:
                            #     all_spans.add(representation)
                    mid_ood_y_pred_vec_alpha_or_sfmx = self.extract_token_merged_score(datapoint, interv_dict)
                    ood_y_pred.extend(mid_ood_y_pred)
                    cls_y_pred.extend(mid_cls_y_pred)
                    ood_y_pred_vec_alpha_or_sfmx.extend(mid_ood_y_pred_vec_alpha_or_sfmx)


                    ### calculate the unique predicted entities score vec accroding to ground-truth index
                    # if cal_unique_predict_scores:
                    #     for rep in unique_gold_representation_set:
                    #         if rep not in unique_gt_in_pd_score_dict.keys():
                    #             unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)]
                    #         else:
                    #             unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)])

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            new_cls_y_true, new_cls_y_pred = [], []
            for idx in range(len(cls_y_true)):
                if cls_y_true[idx] == OOD_label or cls_y_true[idx] == 'OOD':
                    continue
                new_cls_y_true.append(cls_y_true[idx])
                new_cls_y_pred.append(cls_y_pred[idx])
            cls_y_true, cls_y_pred = new_cls_y_true, new_cls_y_pred

            # # convert true and predicted values to two span-aligned lists
            # true_values_span_aligned = []
            # predicted_values_span_aligned = []
            # predicted_scoredict_span_aligned = [] # self-added
            # cls_true_values_span_aligned = []
            # cls_predicted_values_span_aligned = []
            # # (to do) generate label list: cls_true_values_span_aligned, cls_predicted_values_span_aligned
            # for span in all_spans:
            #     # self-added: count shared_num
            #     if span in all_true_values and span in all_predicted_values:
            #         self.shared_entity_num += 1
            #     if span in all_true_values and span not in all_predicted_values:
            #         self.unique_gt_entity_num += 1
            #
            #     list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else [wrong_span_label] # self-comment: actual wrong span
            #     cls_list_of_gold_values_for_span = cls_all_true_values[span] if span in cls_all_true_values else ["O"]
            #     # delete exluded labels if exclude_labels is given
            #     for excluded_label in exclude_labels:
            #         if excluded_label in list_of_gold_values_for_span:
            #             list_of_gold_values_for_span.remove(excluded_label)
            #             cls_list_of_gold_values_for_span.remove(excluded_label)
            #     # if after excluding labels, no label is left, ignore the datapoint
            #     if not list_of_gold_values_for_span:
            #         continue
            #     true_values_span_aligned.append(list_of_gold_values_for_span)
            #     cls_true_values_span_aligned.append(cls_list_of_gold_values_for_span)
            #     predicted_values_span_aligned.append(
            #         all_predicted_values[span] if span in all_predicted_values else [wrong_span_label] # self-comment: not actual ones, but the part unique in the gt
            #     )
            #     cls_predicted_values_span_aligned.append(
            #         cls_all_predicted_values[span] if span in cls_all_predicted_values else ["O"]
            #     )
            #
            #     if cal_unique_predict_scores:
            #         if span in all_predicted_values:
            #             to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
            #         else:
            #             to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
            #         predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
            #     else:
            #         predicted_scoredict_span_aligned.append(
            #             all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
            #         )
            #     # O： 1.0 Loc： 1.0 Per: 1.0 Event 1.0
            #     # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
            #     # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item(wrong_span_label) # self-comment: need change for OOD detection metric
            for true_values in ood_y_true:  # token
                evaluation_label_dictionary.add_item(true_values)
            for predicted_values in all_predicted_values.values():
                evaluation_label_dictionary.add_item(predicted_values)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords

            default_evaluation_label_dictionary = Dictionary(add_unk=False)
            for label in default_score_dict.keys():
                default_evaluation_label_dictionary.add_item(label) # (check) expect default_evaluation_label_dictionary in 1 + c keywords

        # check if this is a multi-label problem
        multi_label = False
        # for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
        #     if len(true_instance) > 1 or len(predicted_instance) > 1:
        #         multi_label = True
        #         break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        # y_true = []
        # y_pred = []
        # if multi_label:
        #     # multi-label problems require a multi-hot vector for each true and predicted label
        #     for true_instance in true_values_span_aligned:
        #         y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
        #         for true_value in true_instance:
        #             y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
        #         y_true.append(y_true_instance.tolist())
        #
        #     for predicted_values in predicted_values_span_aligned:
        #         y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
        #         for predicted_value in predicted_values:
        #             y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
        #         y_pred.append(y_pred_instance.tolist())
        # else:
        #     # single-label problems can do with a single index for each true and predicted label
        #     y_true = [
        #         evaluation_label_dictionary.get_idx_for_item(true_instance[0])
        #         for true_instance in true_values_span_aligned
        #     ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
        #     y_pred = [
        #         evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
        #         for predicted_instance in predicted_values_span_aligned
        #     ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        # # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        #
        #     # process to transfer ood+ws -> ood
        #     # true: ws -> ood
        #     # predict: ws -> id
        #     fp_true_values_span_aligned = []
        #     fp_predicted_values_span_aligned = []
        #     for ele in true_values_span_aligned:
        #         mid_res = []
        #         for sub_ele in ele:
        #             if sub_ele == wrong_span_label:
        #                 mid_res.append(OOD_label)
        #             else:
        #                 mid_res.append(sub_ele)
        #         fp_true_values_span_aligned.append(mid_res)
        #     for ele in predicted_values_span_aligned:
        #         mid_res = []
        #         for sub_ele in ele:
        #             if sub_ele == wrong_span_label:
        #                 mid_res.append(ID_label)
        #             else:
        #                 mid_res.append(sub_ele)
        #         fp_predicted_values_span_aligned.append(mid_res)

        y_pred_vec_alpha_or_sfmx = []

        for index in range(len(ood_y_pred_vec_alpha_or_sfmx)):
            if use_alpha == False:
                y_pred_vec_alpha_or_sfmx.append(
                    self.map_dict2vec(default_evaluation_label_dictionary, ood_y_pred_vec_alpha_or_sfmx[index], 'none'))
            else:
                y_pred_vec_alpha_or_sfmx.append(
                    self.map_dict2vec(default_evaluation_label_dictionary, ood_y_pred_vec_alpha_or_sfmx[index], 'none'))


            # ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
            #     true_values_span_aligned=fp_true_values_span_aligned, # true_values_span_aligned
            #     predicted_values_span_aligned=fp_predicted_values_span_aligned, # predicted_values_span_aligned
            #     evaluation_label_dictionary=evaluation_label_dictionary,
            #     predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
            #     default_evaluation_label_dictionary=default_evaluation_label_dictionary,
            #     use_alpha=use_alpha,
            #     move_out_label=None, # different from ood_evaluate, here set the move_out_label as None, instead of wrong_span_label
            # )


            # below is for the wrong_span ture pred,
            # ws_y_true, ws_y_pred, ws_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
            #     true_values_span_aligned=true_values_span_aligned,
            #     predicted_values_span_aligned=predicted_values_span_aligned,
            #     evaluation_label_dictionary=evaluation_label_dictionary,
            #     predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
            #     default_evaluation_label_dictionary=default_evaluation_label_dictionary,
            #     use_alpha=use_alpha,
            #     move_out_label=None, # in ood setting, we need to remove the wrong_span_label
            # )
            #
            # # below is for the in-domain c-level semantic class ture pred,
            # cls_y_true, cls_y_pred, cls_y_pred_vec_alpha_or_sfmx = self.get_true_pred_vectors(
            #     true_values_span_aligned=cls_true_values_span_aligned,
            #     predicted_values_span_aligned=cls_predicted_values_span_aligned,
            #     evaluation_label_dictionary=default_evaluation_label_dictionary, # changed into c-class dict
            #     predicted_scoredict_span_aligned=predicted_scoredict_span_aligned,
            #     default_evaluation_label_dictionary=default_evaluation_label_dictionary,
            #     use_alpha=use_alpha,
            #     move_out_label='OOD', # in ood setting, we need to remove the wrong_span_label
            # )


        # if use_alpha == False:
        #     # y_pred_vec_alpha_or_sfmx = [
        #     #     self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
        #     #     for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     # ]
        #     y_pred_vec_alpha_or_sfmx = [
        #         self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
        #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     ]
        #
        # else:
        # #     y_pred_vec_alpha_or_sfmx = [
        # #         self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
        # #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        # #     ]
        #     y_pred_vec_alpha_or_sfmx = [
        #         self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
        #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     ]



        # now, calculate evaluation numbers
        # target_names = []
        # labels = []
        #
        # counter = Counter(itertools.chain.from_iterable(cls_y_true))
        # counter.update(list(itertools.chain.from_iterable(cls_y_pred)))
        #
        # for label_name, count in counter.most_common(): # (might need to change for the ID classification task)
        #     # if label_name == "O":
        #     # if label_name == wrong_span_label:
        #     if label_name == OOD_label or label_name == 'OOD':
        #         continue
        #     target_names.append(label_name)
        #     # labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))
        #     labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))

        target_names = set(cls_y_true)
        target_names = list(target_names)
        labels=[]
        for label_name in target_names:
            labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))

        # token: transfer from str to id
        cls_y_true, cls_y_pred = self.transfer_str_to_id(cls_y_true, cls_y_pred, default_evaluation_label_dictionary)
        ood_y_true, ood_y_true = self.transfer_str_to_id(ood_y_true, ood_y_true, evaluation_label_dictionary)



        ### (to do) skip the classification process
        # there is at least one gold label or one prediction (default)
        if len(cls_all_true_values) + len(cls_all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                cls_y_true, # [0, 1, 2] 2: OOD_label -> [0, 1] -> transfer -> c 个语义类
                cls_y_pred, # [0, 1]
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(cls_y_true, cls_y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        ood_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx=y_pred_vec_alpha_or_sfmx,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='ood_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ood_unc_list)

        ws_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx=y_pred_vec_alpha_or_sfmx,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='wrong_span_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label), # evaluation_label_dictionary.get_idx_for_item(wrong_span_label)
        )
        writent_list.extend(ws_unc_list)




        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        # print(str(out_path.absolute())[:-8] + 'unc.txt')
        print(write_file_name)




        return result


    # def cal_unique_gt_in_pd_scores(self, datapoint, unique_gt_rp_set, interv_dict):
    #     res = {}
    #     return res

    def get_token_info(self, datapoint, sentence_id, unlabeled_identifier):
        reprensentation_list = []
        index_list = []
        try:
            un_identifier = unlabeled_identifier  # e.g: 'Span[1:2]: "asian"'
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            comma_idx = un_identifier.index(':')
        except:
            un_identifier = str(unlabeled_identifier)
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            new_str = un_identifier[lf_braket_idx:rg_braket_idx]
            new_comma_idx = new_str.index(':')
            comma_idx = new_comma_idx + lf_braket_idx
        low_idx = int(un_identifier[lf_braket_idx + 1:comma_idx])
        high_idx = int(un_identifier[comma_idx + 1:rg_braket_idx])

        for i in range(low_idx, high_idx):
            cur_rep = str(sentence_id) + ": [" + str(i) + "]" + datapoint[i].text
            reprensentation_list.append(cur_rep)
            index_list.append(i)
        return reprensentation_list, index_list

    def extract_token_merged_score(self, datapoint, interv_dict):
        res = []
        for i in range(len(datapoint)):
            token_merged_score_dict = self.cal_max_score_in_interv(datapoint.tokens[i], interv_dict)
            res.append(token_merged_score_dict)
        return res

    def transfer_str_to_id(self, true_values_span_aligned, predicted_scoredict_span_aligned, evaluation_label_dictionary):
        ood_y_true = []
        ood_y_pred = []
        for ind in range(len(true_values_span_aligned)):
            # ele = true_values_span_aligned[ind]
            # predicted_scoredict_ins = predicted_scoredict_span_aligned[ind]
            ood_y_true.append(evaluation_label_dictionary.get_idx_for_item(true_values_span_aligned[ind]))
            ood_y_pred.append(evaluation_label_dictionary.get_idx_for_item(predicted_scoredict_span_aligned[ind]))
        return ood_y_pred, ood_y_pred



    def get_true_pred_vectors(self,
                              true_values_span_aligned,  # list of ground_truth labels in string
                              predicted_values_span_aligned, # list of predicted labels in string
                              evaluation_label_dictionary, # a dict of interested of dictionary (could be different from c-semantic-class dict)
                              predicted_scoredict_span_aligned, # a list of c-semantic class vector
                              default_evaluation_label_dictionary, # a dict of c-semantic class dict
                              use_alpha, # whether use alpha or not
                              move_out_label=None # the case of moving out label (e.g. OOD-> remove wrong_span)
                              ):
        if move_out_label is not None:
            assert move_out_label=="OOD" or move_out_label in evaluation_label_dictionary.get_items()
        ood_y_true = []
        ood_y_pred = []
        ood_y_pred_vec_alpha_or_sfmx = []
        for ind in range(len(true_values_span_aligned)):
            ele = true_values_span_aligned[ind]
            predicted_scoredict_ins = predicted_scoredict_span_aligned[ind]
            if move_out_label is None or (move_out_label is not None and ele[0] != move_out_label):
                ood_y_true.append(evaluation_label_dictionary.get_idx_for_item(true_values_span_aligned[ind][0]))
                ood_y_pred.append(evaluation_label_dictionary.get_idx_for_item(predicted_values_span_aligned[ind][0]))
                if use_alpha == False:
                    ood_y_pred_vec_alpha_or_sfmx.append(
                        self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none'))
                else:
                    ood_y_pred_vec_alpha_or_sfmx.append(
                        self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none'))

        return ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx

    def get_unc_score_list(self,
                           y_true, # the ground-truth label list
                           y_pred, # the predicted label list
                           y_pred_vec_alpha_or_sfmx, # the c-class probability vector
                           use_alpha, # whether use alpha (e.g PN-based) or not (e.g. ensemble).
                           uncertainty_metrics, # a list of considered uncertainty metrics
                           sfmx_mode, # the softmax (normalization) mode used in aleatoric unc
                           task_keyword=None, # the name of task ['ood_', 'wrong_span_', '']
                           auroc_aupr_pos_id=None, # if postive id exist, use the postive id for evalue; else use the (y_true==y_pred) for the binary id
                           ):
        writent_list = []
        if task_keyword not in ['', 'ood_', 'wrong_span_']:
            raise ValueError(f"the task_keywod={task_keyword} is not in ['', 'ood_', 'wrong_span_']")

        if task_keyword == 'wrong_span_':
            print(f'auroc_aupr_pos_id={auroc_aupr_pos_id}')
            # assert auroc_aupr_pos_id not in y_true
            writent_list.append(f'total(three parts) entities are {len(y_true)}\n')
            print(f'total(three parts) entities are {len(y_true)}')
            ws_num_jsq = 0
            for ele1 in y_true:
                if ele1 == auroc_aupr_pos_id:
                    ws_num_jsq += 1
            writent_list.append(f'total(wrong_span part) entities are {ws_num_jsq}\n')
            print(f'total(wrong_span part) entities are {ws_num_jsq}')
            writent_list.append(f'total(shared part) entities are {self.shared_entity_num}\n')
            print(f'total(shared part) entities are {self.shared_entity_num}')
            writent_list.append(f'total(unique_gt part) entities are {self.unique_gt_entity_num}\n')
            print(f'total(unique_gt part) entities are {self.unique_gt_entity_num}')


        if use_alpha == True:
            if 'diss' in uncertainty_metrics:
                diss_unc = dissonance_uncertainty(y_pred_vec_alpha_or_sfmx)
                diss_auroc, diss_aupr, diss_wrlist = get_auroc_aupr(y_true, y_pred, diss_unc, keyword=task_keyword+'diss', auroc_aupr_pos_id=auroc_aupr_pos_id)
                writent_list.extend(diss_wrlist)
            if 'vacu' in uncertainty_metrics:
                vacu_unc = vacuity_uncertainty(y_pred_vec_alpha_or_sfmx)
                vacu_auroc, vacu_aupr, vacu_wrlist = get_auroc_aupr(y_true, y_pred, vacu_unc, keyword=task_keyword+'vacu', auroc_aupr_pos_id=auroc_aupr_pos_id)
                writent_list.extend(vacu_wrlist)
            if 'entr_mean' in uncertainty_metrics: # only applied to alpha form
                entr_unc = get_un_entropy(y_pred_vec_alpha_or_sfmx, mode='mean')
                entr_auroc, entr_aupr, entr_wrlist = get_auroc_aupr(y_true, y_pred, entr_unc, keyword=task_keyword+'entr_mean', auroc_aupr_pos_id=auroc_aupr_pos_id)
                writent_list.extend(entr_wrlist)
            if 'epis' in uncertainty_metrics:
                # for misclassification (to do consider the difference between misclassifciation and ood~)
                if task_keyword in ['ood_', 'wrong_span_']:
                    epis_unc = one_over_sum(y_pred_vec_alpha_or_sfmx)
                elif task_keyword in ['']:
                    epis_unc = one_over_max(y_pred_vec_alpha_or_sfmx)
                epis_auroc, epis_aupr, epis_wrlist = get_auroc_aupr(y_true, y_pred, epis_unc, keyword=task_keyword+'epis', auroc_aupr_pos_id=auroc_aupr_pos_id)
                writent_list.extend(epis_wrlist)
                # for OOD detection -> use sum of alpha

            if 'alea' in uncertainty_metrics:
                raise ValueError('alea in use_apha=True is not supported')
            if 'entr_sfmx' in uncertainty_metrics:
                raise ValueError('entr_sfmx in use_apha=True is not supported')


        else:
            # below is for use_alpha == False -> use sfmx of probability
            if 'epis' in uncertainty_metrics:
                raise ValueError('epis in use_apha=True is not supported')
            if 'entr_mean' in uncertainty_metrics:
                raise ValueError('entr_mean in use_apha=True is not supported')
            if 'alea' in uncertainty_metrics:
                # for misclassification
                alea_unc = one_over_max(y_pred_vec_alpha_or_sfmx)
                alea_auroc, epis_aupr, epis_wrlist = get_auroc_aupr(y_true, y_pred, alea_unc, keyword=task_keyword+'alea_'+sfmx_mode, auroc_aupr_pos_id=auroc_aupr_pos_id)
                writent_list.extend(epis_wrlist)
            if 'entr_sfmx' in uncertainty_metrics: # only applied to non-alpha form
                entr_unc = get_un_entropy(y_pred_vec_alpha_or_sfmx, mode='sfmx')
                entr_auroc, entr_aupr, entr_wrlist = get_auroc_aupr(y_true, y_pred, entr_unc, keyword=task_keyword+'entr_sfmx', auroc_aupr_pos_id=auroc_aupr_pos_id)
                writent_list.extend(entr_wrlist)
        return writent_list


    @abstractmethod
    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.  # noqa: E501
        """
        raise NotImplementedError

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            # check if there is a label mismatch
            g = [label.labeled_identifier for label in datapoint.get_labels(gold_label_type)]
            p = [label.labeled_identifier for label in datapoint.get_labels("predicted")]
            g.sort()
            p.sort()
            correct_string = " -> MISMATCH!\n" if g != p else ""
            # print info
            eval_line = (
                f"{datapoint.to_original_text()}\n"
                f" - Gold: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels(gold_label_type))}\n"
                f" - Pred: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels('predicted'))}\n{correct_string}\n"
            )
            lines.append(eval_line)
        return lines

    def cal_max_score_in_interv(self, token, interv_dict):
        token_cat_num = len(token.tags_proba_dist['predicted'])
        token_info = token.tags_proba_dist['predicted']
        token_val_scr_dict = {}
        token_merged_score_dict = {}

        for i in range(token_cat_num):
            token_val_scr_dict[token_info[i].value] = token_info[i].score

        # interv_dict = {
        #     'Location': ['B-Location', 'I-Location', 'E-Location', 'S-Location'],
        # }

        interv_dict_key_list = list(interv_dict.keys())
        for j in range(len(interv_dict_key_list)):
            cur_interv_score_list = []
            cur_key = interv_dict_key_list[j]
            for k in range(len(interv_dict[cur_key])):
                cur_interv_score_list.append(token_val_scr_dict[interv_dict[cur_key][k]])
            cur_interv_max_score = max(cur_interv_score_list)   # process to get the score for an interv
            token_merged_score_dict[cur_key] = cur_interv_max_score

        return token_merged_score_dict





    def merge_beios_to_one_score(self, datapoint, predicted_span, interv_dict):
        try:
            un_identifier = predicted_span.unlabeled_identifier  # e.g: 'Span[1:2]: "asian"'
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            comma_idx = un_identifier.index(':')
        except:
            un_identifier = str(predicted_span)
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            new_str = un_identifier[lf_braket_idx:rg_braket_idx]
            new_comma_idx = new_str.index(':')
            comma_idx = new_comma_idx + lf_braket_idx
        low_idx = int(un_identifier[lf_braket_idx + 1:comma_idx])
        high_idx = int(un_identifier[comma_idx + 1:rg_braket_idx])
        token_merged_score_dict = None
        for i in range(low_idx, high_idx):
            if token_merged_score_dict is None:
                token_merged_score_dict = self.cal_max_score_in_interv(datapoint.tokens[i], interv_dict)
                for dict_key in interv_dict.keys():
                    token_merged_score_dict[dict_key] = [token_merged_score_dict[dict_key]]
            else:
                next_token_score_dict = self.cal_max_score_in_interv(datapoint.tokens[i], interv_dict)
                for dict_key in interv_dict.keys():
                    token_merged_score_dict[dict_key].append(next_token_score_dict[dict_key])

        for key in token_merged_score_dict.keys():
            try:
                if self.use_var_metric==True:
                    # token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key])) / (1 + np.std(np.array(token_merged_score_dict[key])))
                    token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key])) / ((1 + np.tanh(len(token_merged_score_dict[key]))) * 2)
                else:
                    token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key]))  # process to get the merged scores for multi tokens by mean
            except:
                token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key]))
        return token_merged_score_dict


    def get_interv_dict(self, label_dictionary):
        micro_label_list = label_dictionary.get_items()
        macro_label_list = set()
        for ele in micro_label_list:
            if ele != 'O':
                macro_label_list.add(ele[2:])
            else:
                macro_label_list.add(ele)
        macro_label_list = list(macro_label_list)

        interv_dict = {}

        for mac_ele in macro_label_list:
            interv_dict[mac_ele] = []
            for mic_ele in micro_label_list:
                if mac_ele in mic_ele:
                    interv_dict[mac_ele].append(mic_ele)

        return interv_dict

    def get_default_score_dict(self, interv_dict):
        default_score_dict = {}
        for key in interv_dict.keys():
            if key == 'O':
                val = 1.0
            else:
                val = 1.0
            default_score_dict[key] = val
        return default_score_dict

    def get_default_alpha_dict(self, interv_dict):
        default_score_dict = {}
        for key in interv_dict.keys():
            val = 1.0
            default_score_dict[key] = val
        return default_score_dict

    def map_dict2vec(self, evaluation_label_dictionary, predicted_scoredict_ins, method='none'):
        assert method == 'none'
        y_pred_instance_vec = np.zeros(len(evaluation_label_dictionary), dtype=float)
        # print(predicted_scoredict_ins)
        try:
            if len(predicted_scoredict_ins) == 1:
                predicted_scoredict_ins = predicted_scoredict_ins[0]
        except:
            pass
        for key in list(predicted_scoredict_ins.keys()): # check alpha use None or not
            # print(evaluation_label_dictionary)
            # print(key) # few samples are too little to be seen.
            y_pred_instance_vec[evaluation_label_dictionary.get_idx_for_item(key)] = predicted_scoredict_ins[key]
        if method == 'mean':
            y_pred_instance_vec = y_pred_instance_vec/y_pred_instance_vec.sum()
        elif method == 'none':
            pass
        else:
            raise ValueError('the method choice is wrong!')
        return y_pred_instance_vec.tolist()

    def build_label_dict(self, str_label_list):
        str_label_set = set(str_label_list)
        ref_str_label_list = list(str_label_set)
        label2id = {}
        id2label = {}
        for i in range(len(ref_str_label_list)):
            label2id[ref_str_label_list[i]] = i
            id2label[i] = ref_str_label_list[i]
        return label2id, id2label

    def tSNE_fig(self,
                 data_points,
                 out_path,
                 mini_batch_size=1,
                 gold_label_type='ner',
                 mode='latent', #[latent, pretrain]
                 save_name_pre=''
                 ):
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        with torch.no_grad():

            span_feature_list = []
            span_label_list = []

            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            print("start prepare feature for drawing t-SNE figure")

            for batch in Tqdm.tqdm(loader):
                for datapoint in batch:
                    # cal embedding
                    # if not isinstance(data_points, list):
                    #     sentences = [data_points]
                    # else:
                    #     sentences = data_points
                    if mode == 'pretrain':
                        self.embeddings.embed(datapoint)
                    # get the gold labels
                    for gold_label in datapoint.get_labels(gold_label_type):
                        # this part is same to part of func: merge_beios_to_one_score
                        un_identifier = gold_label.unlabeled_identifier  # 'Span[1:2]: "asian"'
                        lf_braket_idx = un_identifier.index('[')
                        rg_braket_idx = un_identifier.index(']')
                        comma_idx = un_identifier.index(':')
                        low_idx = int(un_identifier[lf_braket_idx + 1:comma_idx])
                        high_idx = int(un_identifier[comma_idx + 1:rg_braket_idx])
                        span_level_emb_list = []
                        for i in range(low_idx, high_idx):
                            if mode == 'pretrain':
                                cur_token_emb = datapoint.tokens[i].embedding
                                span_level_emb_list.append(cur_token_emb.unsqueeze(0))
                            elif mode == 'latent':
                                sentence_pretrain_emb, sentence_latent_emb = self.get_latent_feature(datapoint)
                                cur_token_emb = sentence_latent_emb[:, i, :]
                                span_level_emb_list.append(cur_token_emb)

                            else:
                                raise ValueError(f'the mode={mode} is wrong! [latent, pretrain]')

                        span_level_emb_tensor_cat = torch.cat(span_level_emb_list, dim=0)
                        span_level_emb_tensor_mean = span_level_emb_tensor_cat.mean(dim=0)
                        span_feature_list.append(span_level_emb_tensor_mean.cpu().tolist())
                        span_label_list.append(gold_label.value)

            # start draw t-SNE fig
            span_label2id_dict, span_id2label_dict = self.build_label_dict(span_label_list)
            span_label_list = [
                span_label2id_dict[ele] for ele in span_label_list
            ]
            span_str_label_list = [
                span_id2label_dict[ele] for ele in range(len(list(span_label2id_dict.keys())))
            ]

            def visual(feat):
                # t-SNE的最终结果的降维与可视化
                ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

                x_ts = ts.fit_transform(feat)

                print(x_ts.shape)  # [num, 2]

                x_min, x_max = x_ts.min(0), x_ts.max(0)

                x_final = (x_ts - x_min) / (x_max - x_min)

                return x_final

            # 设置散点形状
            maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H', 'o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
            # 设置散点颜色
            colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
                      'hotpink', 'fuchisia', 'hotpink', 'crimson', 'slategray', 'lightgreen', 'palegreen', 'cornsilk', 'peru', 'maroon', 'lime']
            # 图例名称
            Label_Com = ['a', 'b', 'c', 'd']
            # 设置字体格式
            font1 = {'family': 'Times New Roman',
                     'weight': 'bold',
                     'size': 32,
                     }

            def plotlabels(S_lowDWeights, Trure_labels, used_labels, name):
                True_labels = Trure_labels.reshape((-1, 1))
                S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
                S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
                print(S_data)
                print(S_data.shape)  # [num, 3]

                for index in range(len(used_labels)):  # 假设总共有三个类别，类别的表示为0,1,2
                    X = S_data.loc[S_data['label'] == index]['x']
                    Y = S_data.loc[S_data['label'] == index]['y']
                    plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index],
                                alpha=0.65, label=used_labels[index])

                    plt.xticks([])  # 去掉横坐标值
                    plt.yticks([])  # 去掉纵坐标值

                plt.title(name, fontsize=32, fontweight='normal', pad=20)
                plt.legend(loc=4, fontsize=8)
                plt.legend(loc='upper left')

            span_feat_array = np.array(span_feature_list)  # 128个特征，每个特征的维度为1024

            span_label_array = np.array(span_label_list)
            print(span_feat_array.shape)
            print(span_label_array.shape)

            span_fig = plt.figure(figsize=(10, 10))

            plotlabels(visual(span_feat_array), span_label_array, span_str_label_list, 'span_level_tr_data_distribution')

            plt.show()
            save_name = save_name_pre + mode + '_span_level_distribution.pdf'
            span_fig.savefig(str(out_path.absolute()) + '/ ' + save_name, bbox_inches='tight')
            print(f'{str(out_path.absolute())}/{save_name} is save finished!')


class EnsembleUncClassifier(UncModel[DT], typing.Generic[DT]):
    """Abstract base class for all Flair models that do classification,
    both single- and multi-label. It inherits from flair.nn.Model and adds an
    unified evaluate() function so that all classification models use the same
    evaluation routines and compute the same numbers."""

    def evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        uncertainty_metrics: List[str]=[],
        test_dropout_num=1, #true_setting
        # test_dropout_num=3, #debug_uage
        use_alpha=False,
        sfmx_mode=None,
        cal_unique_predict_scores: bool = False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)

        default_score_dict = self.get_default_alpha_dict(interv_dict)
        default_score_dict_dp = self.get_default_alpha_dict_dp(interv_dict, test_dropout_num)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            all_predicted_score_dict = {}
            all_predicted_score_dict_dp = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
                unique_gt_in_pd_score_dict_dp = {}
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0

            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                for test_epoch in range(test_dropout_num):
                    if test_epoch != (test_dropout_num - 1):
                        self.train()
                    else:
                        self.eval()
                    # print(f'this is the test epoch {test_epoch}')
                    # predict for batch
                    loss_and_count = self.predict(
                        batch,
                        embedding_storage_mode=embedding_storage_mode,
                        mini_batch_size=mini_batch_size,
                        return_probabilities_for_all_classes=True, # self-added
                        label_name="predicted",
                        return_loss=return_loss,

                    )

                    # if test_epoch == (test_dropout_num - 1):
                    #     self.train()  # restart training, after eval

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    # get the gold labels
                    if cal_unique_predict_scores: unique_gold_representation_set = set()

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        value = gold_label.value
                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                        else:
                            all_true_values[representation].append(value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass

                    # transfer datapoint with multiple dropout results into a mean result and extract the multiple dropout results
                    ensemble_pre_key, ori_pre_key, processed_ensb_pre_key = self.extract_multi_dp_prob(datapoint, test_dropout_num)

                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        if cal_unique_predict_scores:
                            if representation in unique_gold_representation_set:
                                unique_gold_representation_set.remove(representation)

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span.value] # self-comment: need recalculate after [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                            # all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True)]
                            all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(
                                datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key,
                                processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                                cal_dp=True)

                            all_predicted_score_dict[representation], all_predicted_score_dict_dp[representation] = [all_predicted_score_dict_ele], [all_predicted_score_dict_dp_ele]

                            # 将最后一个model epoch 进行 training 关闭操作 finished!
                            # 根据最后一个 model epoch 预测的结果，或者 统一 的 对 各个 dp res 进行 merge_beios_to_one_score 的操作（改内部的关键字） # almost finished
                            # 将 merge 后的 phrase-level prob mat 用于 testing process 计算

                        else:
                            all_predicted_values[representation].append(predicted_span.value)
                            # all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint,predicted_span, interv_dict, cal_dp=True))
                            all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key, processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num, cal_dp=True)
                            all_predicted_score_dict[representation].append(all_predicted_score_dict_ele)
                            all_predicted_score_dict_dp[representation].append(all_predicted_score_dict_dp_ele)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    ### calculate the unique predicted entities score vec accroding to ground-truth index
                    if cal_unique_predict_scores:
                        for rep in unique_gold_representation_set:
                            mid_unique_gt_in_pd_score_dict_ele, mid_unique_gt_in_pd_score_dict_dp_ele = self.merge_beios_to_one_score(
                                    datapoint, rep, interv_dict, ori_pre_key=ori_pre_key,
                                    processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                                    cal_dp=True)
                            if rep not in unique_gt_in_pd_score_dict.keys():
                                # unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)] # for non-dp
                                unique_gt_in_pd_score_dict[rep],unique_gt_in_pd_score_dict_dp[rep] = [mid_unique_gt_in_pd_score_dict_ele], [mid_unique_gt_in_pd_score_dict_dp_ele]
                            else:
                                # unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)]) # for non-dp
                                unique_gt_in_pd_score_dict[rep].append(mid_unique_gt_in_pd_score_dict_ele)
                                unique_gt_in_pd_score_dict_dp[rep].append(mid_unique_gt_in_pd_score_dict_dp_ele)



                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            predicted_scoredict_span_aligned = [] # self-added
            predicted_scoredict_span_aligned_dp = [] # self-added
            for span in all_spans:
                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else ["O"] # self-comment: need change for OOD detection metric
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else ["O"] # self-comment: need change for OOD detection metric
                )


                if cal_unique_predict_scores:
                    if span in all_predicted_values:
                        to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
                        to_add_predicted_scoredict_span_aligned_dp = all_predicted_score_dict_dp[span]
                    else:
                        to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
                        to_add_predicted_scoredict_span_aligned_dp = unique_gt_in_pd_score_dict_dp[span]
                    predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
                    predicted_scoredict_span_aligned_dp.append(to_add_predicted_scoredict_span_aligned_dp)
                else:
                    predicted_scoredict_span_aligned.append(
                        all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
                    )
                    predicted_scoredict_span_aligned_dp.append(
                        # a variable with '_dp' means it is/includes a list of repeated results by dropout
                        all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp
                    )


                # predicted_scoredict_span_aligned.append(
                #     all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict
                # )
                # # add dp result at here: all_predicted_score_dict_dp[representation]
                # predicted_scoredict_span_aligned_dp.append(   # a variable with '_dp' means it is/includes a list of repeated results by dropout
                #     all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp
                # )

                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                  # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item("O") # self-comment: need change for OOD detection metric
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        if use_alpha == False:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]

            y_pred_vec_alpha_or_sfmx_dp = [
                self.map_dict2vec_dp(evaluation_label_dictionary, predicted_scoredict_ins_dp, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins_dp in predicted_scoredict_span_aligned_dp
            ]  # use y_pred_vec_alpha_or_sfmx_dp for uncertainty estimation

        else:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]



        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == "O":
                continue
            target_names.append(label_name)
            labels.append(evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(y_true, y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        # self-added: calculate unc metrics for the span-level uncertainty
        # cal unc_scores


        # add dissonance, vaculity, entropy, epistemic at here
        # use [dissonance, vaculity, entropy, epistemic]

        # max_unc = [
        #    1.0/(max(ele)+1e-8) for ele in y_pred_vec_alpha_or_sfmx
        # ]
        # enpy_unc = [
        #    entropy(ele) for ele in y_pred_vec_alpha_or_sfmx
        # ]
        #
        # auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score = show_roc_auc(y_true, y_pred, enpy_unc)
        # print(f"entropy_unc auroc & aupr are {auroc_score}, {aupr_score}")
        # writent_list.append("\n")
        # writent_list.append(f"entropy_unc_auroc is {auroc_score}\n")
        # writent_list.append(f"entropy_unc_aupr is  {aupr_score}\n")
        #
        # auroc_score, fpr, tpr, thresholds_auroc, precision, recall, thresholds_aupr, aupr_score = show_roc_auc(y_true, y_pred, max_unc)
        # print(f"max_unc auroc & aupr are {auroc_score}, {aupr_score}")
        # writent_list.append(f"max_unc_auroc is {auroc_score}\n")
        # writent_list.append(f"max_unc_aupr is  {aupr_score}\n")
        #
        # ece_res = ece_score(np.array(y_pred_vec_alpha_or_sfmx), y_true, n_bins=10)
        # print("ece is ", ece_res)
        # writent_list.append(f"ece is , {ece_res}\n")
        # writent_list.append("\n")

        if use_alpha == False:
            # print(np.array(y_pred_vec_alpha_or_sfmx_dp).shape)
            y_pred_vec_alpha_or_sfmx_dp = np.array(y_pred_vec_alpha_or_sfmx_dp).transpose(2, 0, 1)
            if 'entr_sfmx' in uncertainty_metrics:
                entr_unc, class_entr_unc = entropy_dropout(y_pred_vec_alpha_or_sfmx_dp)
                entr_unc = entr_unc.squeeze(axis=1)
                entr_unc = entr_unc.tolist()
                # entr_unc, class_entr_unc = entropy_dropout(y_pred_vec_alpha_or_sfmx_dp)
                entr_auroc, entr_aupr, entr_wrlist = get_auroc_aupr(y_true, y_pred, entr_unc, keyword='entr_sfmx')
                writent_list.extend(entr_wrlist)
            if 'alea' in uncertainty_metrics:
                # for misclassification
                alea_unc, class_alea_unc = aleatoric_dropout(y_pred_vec_alpha_or_sfmx_dp)

                alea_unc = alea_unc.squeeze(axis=1)
                alea_unc = alea_unc.tolist()

                alea_auroc, alea_aupr, alea_wrlist = get_auroc_aupr(y_true, y_pred, alea_unc, keyword='alea')
                writent_list.extend(alea_wrlist)
            if 'epis' in uncertainty_metrics:
                dr_eps_class = class_entr_unc - class_alea_unc
                epis_unc = np.sum(dr_eps_class, axis=1, keepdims=True)

                epis_unc = epis_unc.squeeze(axis=1)
                epis_unc = epis_unc.tolist()

                epis_auroc, epis_aupr, epis_wrlist = get_auroc_aupr(y_true, y_pred, epis_unc, keyword='epis')
                writent_list.extend(epis_wrlist)

        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        print(write_file_name)

        self.train()  # restart training, after eval

        return result

    def ood_evaluate(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        uncertainty_metrics: List[str]=[],
        test_dropout_num=1, #true_setting
        # test_dropout_num=3, #debug_uage
        use_alpha=False,
        sfmx_mode=None,
        leave_out_labels: List[str] = [],
        cal_unique_predict_scores: bool = True,
        use_var_metric=False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        ## required in OOD
        assert cal_unique_predict_scores == True
        self.use_var_metric=use_var_metric
        self.shared_entity_num=0
        self.unique_gt_entity_num=0

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)

        default_score_dict = self.get_default_alpha_dict(interv_dict)
        default_score_dict_dp = self.get_default_alpha_dict_dp(interv_dict, test_dropout_num)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            cls_all_true_values = {}
            cls_all_predicted_values = {}
            all_predicted_score_dict = {}
            all_predicted_score_dict_dp = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
                unique_gt_in_pd_score_dict_dp = {}
            ID_label, OOD_label = 'ID_label', 'OOD_label' # OOD and ID labels definition
            wrong_span_label = 'wrong_span_label'
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0

            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                for test_epoch in range(test_dropout_num):
                    if test_epoch != (test_dropout_num - 1):
                        self.train()
                    else:
                        self.eval()
                    # print(f'this is the test epoch {test_epoch}')
                    # predict for batch
                    loss_and_count = self.predict(
                        batch,
                        embedding_storage_mode=embedding_storage_mode,
                        mini_batch_size=mini_batch_size,
                        return_probabilities_for_all_classes=True, # self-added
                        label_name="predicted",
                        return_loss=return_loss,

                    )

                    # if test_epoch == (test_dropout_num - 1):
                    #     self.train()  # restart training, after eval

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    # get the gold labels
                    if cal_unique_predict_scores: unique_gold_representation_set = set()

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        # value = gold_label.value
                        cls_value = gold_label.value ## different from misclassification
                        if gold_label.value != 'OOD':
                            value = ID_label
                        else:
                            value = OOD_label

                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                            cls_all_true_values[representation] = [cls_value]
                        else:
                            all_true_values[representation].append(value)
                            cls_all_true_values[representation].append(cls_value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass

                    # transfer datapoint with multiple dropout results into a mean result and extract the multiple dropout results
                    ensemble_pre_key, ori_pre_key, processed_ensb_pre_key = self.extract_multi_dp_prob(datapoint, test_dropout_num)

                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        if cal_unique_predict_scores:
                            if representation in unique_gold_representation_set:
                                unique_gold_representation_set.remove(representation)

                        cls_predicted_span_value = predicted_span.value
                        predicted_span_value = ID_label  # predicted_span.value

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span_value] # self-comment: need recalculate after [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                            # all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True)]
                            all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(
                                datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key,
                                processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                                cal_dp=True)

                            all_predicted_score_dict[representation], all_predicted_score_dict_dp[representation] = [all_predicted_score_dict_ele], [all_predicted_score_dict_dp_ele]
                            cls_all_predicted_values[representation] = [cls_predicted_span_value]
                            # 将最后一个model epoch 进行 training 关闭操作 finished!
                            # 根据最后一个 model epoch 预测的结果，或者 统一 的 对 各个 dp res 进行 merge_beios_to_one_score 的操作（改内部的关键字） # almost finished
                            # 将 merge 后的 phrase-level prob mat 用于 testing process 计算

                        else:
                            all_predicted_values[representation].append(predicted_span_value)
                            # all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint,predicted_span, interv_dict, cal_dp=True))
                            all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key, processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num, cal_dp=True)
                            all_predicted_score_dict[representation].append(all_predicted_score_dict_ele)
                            all_predicted_score_dict_dp[representation].append(all_predicted_score_dict_dp_ele)
                            cls_all_predicted_values[representation].append(cls_predicted_span_value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    ### calculate the unique predicted entities score vec according to ground-truth index
                    if cal_unique_predict_scores:
                        for rep in unique_gold_representation_set:
                            mid_unique_gt_in_pd_score_dict_ele, mid_unique_gt_in_pd_score_dict_dp_ele = self.merge_beios_to_one_score(
                                    datapoint, rep, interv_dict, ori_pre_key=ori_pre_key,
                                    processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                                    cal_dp=True)
                            if rep not in unique_gt_in_pd_score_dict.keys():
                                # unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)] # for non-dp
                                unique_gt_in_pd_score_dict[rep],unique_gt_in_pd_score_dict_dp[rep] = [mid_unique_gt_in_pd_score_dict_ele], [mid_unique_gt_in_pd_score_dict_dp_ele]
                            else:
                                # unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)]) # for non-dp
                                unique_gt_in_pd_score_dict[rep].append(mid_unique_gt_in_pd_score_dict_ele)
                                unique_gt_in_pd_score_dict_dp[rep].append(mid_unique_gt_in_pd_score_dict_dp_ele)


                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            predicted_scoredict_span_aligned = [] # self-added
            predicted_scoredict_span_aligned_dp = [] # self-added
            cls_true_values_span_aligned = []
            cls_predicted_values_span_aligned = []
            for span in all_spans:
                # self-added: count shared_num
                if span in all_true_values and span in all_predicted_values:
                    self.shared_entity_num += 1
                if span in all_true_values and span not in all_predicted_values:
                    self.unique_gt_entity_num += 1

                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else [wrong_span_label] # self-comment: need change for OOD detection metric
                cls_list_of_gold_values_for_span = cls_all_true_values[span] if span in cls_all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                        cls_list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                cls_true_values_span_aligned.append(cls_list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else [wrong_span_label] # self-comment: need change for OOD detection metric
                )
                cls_predicted_values_span_aligned.append(
                    cls_all_predicted_values[span] if span in cls_all_predicted_values else ["O"]
                )


                if cal_unique_predict_scores:
                    if span in all_predicted_values:
                        to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
                        to_add_predicted_scoredict_span_aligned_dp = all_predicted_score_dict_dp[span]
                    else:
                        to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
                        to_add_predicted_scoredict_span_aligned_dp = unique_gt_in_pd_score_dict_dp[span]
                    predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
                    predicted_scoredict_span_aligned_dp.append(to_add_predicted_scoredict_span_aligned_dp)
                else:
                    predicted_scoredict_span_aligned.append(
                        all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
                    )
                    predicted_scoredict_span_aligned_dp.append(
                        # a variable with '_dp' means it is/includes a list of repeated results by dropout
                        all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp
                    )


                # predicted_scoredict_span_aligned.append(
                #     all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # (to do) need to reviese by calcualted_socore_dict
                # )
                # # add dp result at here: all_predicted_score_dict_dp[representation]
                # predicted_scoredict_span_aligned_dp.append(   # a variable with '_dp' means it is/includes a list of repeated results by dropout
                #     all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp # (to do) need to reviese by calcualted_socore_dict
                # )

                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                  # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # # make the evaluation dictionary
            # evaluation_label_dictionary = Dictionary(add_unk=False)
            # evaluation_label_dictionary.add_item("O") # self-comment: need change for OOD detection metric
            # for true_values in all_true_values.values():
            #     for label in true_values:
            #         evaluation_label_dictionary.add_item(label)
            # for predicted_values in all_predicted_values.values():
            #     for label in predicted_values:
            #         evaluation_label_dictionary.add_item(label)

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item(
                wrong_span_label)  # self-comment: need change for OOD detection metric
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords

            default_evaluation_label_dictionary = Dictionary(add_unk=False)
            for label in default_score_dict.keys():
                default_evaluation_label_dictionary.add_item(label)  # (check) expect default_evaluation_label_dictionary in 1 + c keywords

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)

            ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
                true_values_span_aligned=true_values_span_aligned,
                predicted_values_span_aligned=predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp, # this is different from PN version
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=wrong_span_label, # in ood setting, we need to remove the wrong_span_label
            )

            # below is for the wrong_span ture pred,
            ws_y_true, ws_y_pred, ws_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
                true_values_span_aligned=true_values_span_aligned,
                predicted_values_span_aligned=predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp, # this is different from PN version
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=None, # in ood setting, we need to remove the wrong_span_label
            )

            # below is for the in-domain c-level semantic class ture pred,
            cls_y_true, cls_y_pred, cls_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
                true_values_span_aligned=cls_true_values_span_aligned,
                predicted_values_span_aligned=cls_predicted_values_span_aligned,
                evaluation_label_dictionary=default_evaluation_label_dictionary, # changed into c-class dict
                predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label='OOD', # in ood setting, we need to remove the wrong_span_label
            )


        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        if use_alpha == False:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]

            y_pred_vec_alpha_or_sfmx_dp = [
                self.map_dict2vec_dp(default_evaluation_label_dictionary, predicted_scoredict_ins_dp, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins_dp in predicted_scoredict_span_aligned_dp
            ]  # use y_pred_vec_alpha_or_sfmx_dp for uncertainty estimation

        else:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]




        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(cls_all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(cls_all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == OOD_label or label_name == 'OOD':
                continue
            target_names.append(label_name)
            labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(cls_y_true, cls_y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        ood_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx_dp=ood_y_pred_vec_alpha_or_sfmx_dp,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='ood_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ood_unc_list)

        ws_unc_list = self.get_unc_score_list(
            y_true=ws_y_true,
            y_pred=ws_y_pred,
            y_pred_vec_alpha_or_sfmx_dp=ws_y_pred_vec_alpha_or_sfmx_dp,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='wrong_span_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(wrong_span_label)
        )
        writent_list.extend(ws_unc_list)

        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        print(write_file_name)

        self.train()  # restart training, after eval

        return result


    def extract_token_merged_score(self, datapoint, interv_dict, ori_pre_key, processed_ensb_pre_key=None, test_dropout_num=1, cal_dp=False):
        res = []
        res_dp = []
        assert cal_dp == True
        for i in range(len(datapoint)):
            token_merged_score_dict = self.cal_max_score_in_interv(datapoint.tokens[i], interv_dict, ori_pre_key)
            res.append(token_merged_score_dict)
            if cal_dp == True:
                token_merged_score_dict_dp = self.cal_max_score_in_interv_dp(datapoint.tokens[i], interv_dict, ori_pre_key, processed_ensb_pre_key, test_dropout_num)
                res_dp.append(token_merged_score_dict_dp)
        return res, res_dp

    def ood_evaluate_falsepos(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        uncertainty_metrics: List[str]=[],
        test_dropout_num=1, #true_setting
        # test_dropout_num=3, #debug_uage
        use_alpha=False,
        sfmx_mode=None,
        leave_out_labels: List[str] = [],
        cal_unique_predict_scores: bool = True,
        use_var_metric=False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        ## required in OOD
        assert cal_unique_predict_scores == True
        self.use_var_metric=use_var_metric
        self.shared_entity_num=0
        self.unique_gt_entity_num=0

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)

        default_score_dict = self.get_default_alpha_dict(interv_dict)
        default_score_dict_dp = self.get_default_alpha_dict_dp(interv_dict, test_dropout_num)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            cls_all_true_values = {}
            cls_all_predicted_values = {}
            all_predicted_score_dict = {}
            all_predicted_score_dict_dp = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
                unique_gt_in_pd_score_dict_dp = {}
            ID_label, OOD_label = 'ID_label', 'OOD_label' # OOD and ID labels definition
            wrong_span_label = 'wrong_span_label'
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            sentence_id = 0

            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                for test_epoch in range(test_dropout_num):
                    if test_epoch != (test_dropout_num - 1):
                        self.train()
                    else:
                        self.eval()
                    # print(f'this is the test epoch {test_epoch}')
                    # predict for batch
                    loss_and_count = self.predict(
                        batch,
                        embedding_storage_mode=embedding_storage_mode,
                        mini_batch_size=mini_batch_size,
                        return_probabilities_for_all_classes=True, # self-added
                        label_name="predicted",
                        return_loss=return_loss,

                    )

                    # if test_epoch == (test_dropout_num - 1):
                    #     self.train()  # restart training, after eval

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    # get the gold labels
                    if cal_unique_predict_scores: unique_gold_representation_set = set()

                    for gold_label in datapoint.get_labels(gold_label_type):
                        representation = str(sentence_id) + ": " + gold_label.unlabeled_identifier

                        if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        # value = gold_label.value
                        cls_value = gold_label.value ## different from misclassification
                        if gold_label.value != 'OOD':
                            value = ID_label
                        else:
                            value = OOD_label

                        if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                            value = "<unk>"

                        if representation not in all_true_values:
                            all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                            cls_all_true_values[representation] = [cls_value]
                        else:
                            all_true_values[representation].append(value)
                            cls_all_true_values[representation].append(cls_value)

                        if representation not in all_spans:
                            all_spans.add(representation)
                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass

                    # transfer datapoint with multiple dropout results into a mean result and extract the multiple dropout results
                    ensemble_pre_key, ori_pre_key, processed_ensb_pre_key = self.extract_multi_dp_prob(datapoint, test_dropout_num)

                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        representation = str(sentence_id) + ": " + predicted_span.unlabeled_identifier

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        if cal_unique_predict_scores:
                            if representation in unique_gold_representation_set:
                                unique_gold_representation_set.remove(representation)

                        cls_predicted_span_value = predicted_span.value
                        predicted_span_value = ID_label  # predicted_span.value

                        # add to all_predicted_values
                        if representation not in all_predicted_values:
                            all_predicted_values[representation] = [predicted_span_value] # self-comment: need recalculate after [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                            # all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True)]
                            all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(
                                datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key,
                                processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                                cal_dp=True)

                            all_predicted_score_dict[representation], all_predicted_score_dict_dp[representation] = [all_predicted_score_dict_ele], [all_predicted_score_dict_dp_ele]
                            cls_all_predicted_values[representation] = [cls_predicted_span_value]
                            # 将最后一个model epoch 进行 training 关闭操作 finished!
                            # 根据最后一个 model epoch 预测的结果，或者 统一 的 对 各个 dp res 进行 merge_beios_to_one_score 的操作（改内部的关键字） # almost finished
                            # 将 merge 后的 phrase-level prob mat 用于 testing process 计算

                        else:
                            all_predicted_values[representation].append(predicted_span_value)
                            # all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint,predicted_span, interv_dict, cal_dp=True))
                            all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key, processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num, cal_dp=True)
                            all_predicted_score_dict[representation].append(all_predicted_score_dict_ele)
                            all_predicted_score_dict_dp[representation].append(all_predicted_score_dict_dp_ele)
                            cls_all_predicted_values[representation].append(cls_predicted_span_value)

                        if representation not in all_spans:
                            all_spans.add(representation)

                    ### calculate the unique predicted entities score vec accroding to ground-truth index
                    if cal_unique_predict_scores:
                        for rep in unique_gold_representation_set:
                            mid_unique_gt_in_pd_score_dict_ele, mid_unique_gt_in_pd_score_dict_dp_ele = self.merge_beios_to_one_score(
                                    datapoint, rep, interv_dict, ori_pre_key=ori_pre_key,
                                    processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                                    cal_dp=True)
                            if rep not in unique_gt_in_pd_score_dict.keys():
                                # unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)] # for non-dp
                                unique_gt_in_pd_score_dict[rep],unique_gt_in_pd_score_dict_dp[rep] = [mid_unique_gt_in_pd_score_dict_ele], [mid_unique_gt_in_pd_score_dict_dp_ele]
                            else:
                                # unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)]) # for non-dp
                                unique_gt_in_pd_score_dict[rep].append(mid_unique_gt_in_pd_score_dict_ele)
                                unique_gt_in_pd_score_dict_dp[rep].append(mid_unique_gt_in_pd_score_dict_dp_ele)


                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            # convert true and predicted values to two span-aligned lists
            true_values_span_aligned = []
            predicted_values_span_aligned = []
            predicted_scoredict_span_aligned = [] # self-added
            predicted_scoredict_span_aligned_dp = [] # self-added
            cls_true_values_span_aligned = []
            cls_predicted_values_span_aligned = []
            for span in all_spans:
                # self-added: count shared_num
                if span in all_true_values and span in all_predicted_values:
                    self.shared_entity_num += 1
                if span in all_true_values and span not in all_predicted_values:
                    self.unique_gt_entity_num += 1

                list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else [wrong_span_label] # self-comment: need change for OOD detection metric
                cls_list_of_gold_values_for_span = cls_all_true_values[span] if span in cls_all_true_values else ["O"]
                # delete exluded labels if exclude_labels is given
                for excluded_label in exclude_labels:
                    if excluded_label in list_of_gold_values_for_span:
                        list_of_gold_values_for_span.remove(excluded_label)
                        cls_list_of_gold_values_for_span.remove(excluded_label)
                # if after excluding labels, no label is left, ignore the datapoint
                if not list_of_gold_values_for_span:
                    continue
                true_values_span_aligned.append(list_of_gold_values_for_span)
                cls_true_values_span_aligned.append(cls_list_of_gold_values_for_span)
                predicted_values_span_aligned.append(
                    all_predicted_values[span] if span in all_predicted_values else [wrong_span_label] # self-comment: need change for OOD detection metric
                )
                cls_predicted_values_span_aligned.append(
                    cls_all_predicted_values[span] if span in cls_all_predicted_values else ["O"]
                )


                if cal_unique_predict_scores:
                    if span in all_predicted_values:
                        to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
                        to_add_predicted_scoredict_span_aligned_dp = all_predicted_score_dict_dp[span]
                    else:
                        to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
                        to_add_predicted_scoredict_span_aligned_dp = unique_gt_in_pd_score_dict_dp[span]
                    predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
                    predicted_scoredict_span_aligned_dp.append(to_add_predicted_scoredict_span_aligned_dp)
                else:
                    predicted_scoredict_span_aligned.append(
                        all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
                    )
                    predicted_scoredict_span_aligned_dp.append(
                        # a variable with '_dp' means it is/includes a list of repeated results by dropout
                        all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp
                    )


                # predicted_scoredict_span_aligned.append(
                #     all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # (to do) need to reviese by calcualted_socore_dict
                # )
                # # add dp result at here: all_predicted_score_dict_dp[representation]
                # predicted_scoredict_span_aligned_dp.append(   # a variable with '_dp' means it is/includes a list of repeated results by dropout
                #     all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp # (to do) need to reviese by calcualted_socore_dict
                # )

                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                  # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # # make the evaluation dictionary
            # evaluation_label_dictionary = Dictionary(add_unk=False)
            # evaluation_label_dictionary.add_item("O") # self-comment: need change for OOD detection metric
            # for true_values in all_true_values.values():
            #     for label in true_values:
            #         evaluation_label_dictionary.add_item(label)
            # for predicted_values in all_predicted_values.values():
            #     for label in predicted_values:
            #         evaluation_label_dictionary.add_item(label)

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item(
                wrong_span_label)  # self-comment: need change for OOD detection metric
            for true_values in all_true_values.values():
                for label in true_values:
                    evaluation_label_dictionary.add_item(label)
            for predicted_values in all_predicted_values.values():
                for label in predicted_values:
                    evaluation_label_dictionary.add_item(label)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords

            default_evaluation_label_dictionary = Dictionary(add_unk=False)
            for label in default_score_dict.keys():
                default_evaluation_label_dictionary.add_item(label)  # (check) expect default_evaluation_label_dictionary in 1 + c keywords

        # check if this is a multi-label problem
        multi_label = False
        for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
            if len(true_instance) > 1 or len(predicted_instance) > 1:
                multi_label = True
                break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        y_true = []
        y_pred = []
        if multi_label:
            # multi-label problems require a multi-hot vector for each true and predicted label
            for true_instance in true_values_span_aligned:
                y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for true_value in true_instance:
                    y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
                y_true.append(y_true_instance.tolist())

            for predicted_values in predicted_values_span_aligned:
                y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
                for predicted_value in predicted_values:
                    y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
                y_pred.append(y_pred_instance.tolist())
        else:
            # single-label problems can do with a single index for each true and predicted label
            y_true = [
                evaluation_label_dictionary.get_idx_for_item(true_instance[0])
                for true_instance in true_values_span_aligned
            ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
            y_pred = [
                evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
                for predicted_instance in predicted_values_span_aligned
            ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)

            # process to transfer ood+ws -> ood
            # true: ws -> ood
            # predict: ws -> id
            fp_true_values_span_aligned = []
            fp_predicted_values_span_aligned = []
            for ele in true_values_span_aligned:
                mid_res = []
                for sub_ele in ele:
                    if sub_ele == wrong_span_label:
                        mid_res.append(OOD_label)
                    else:
                        mid_res.append(sub_ele)
                fp_true_values_span_aligned.append(mid_res)
            for ele in predicted_values_span_aligned:
                mid_res = []
                for sub_ele in ele:
                    if sub_ele == wrong_span_label:
                        mid_res.append(ID_label)
                    else:
                        mid_res.append(sub_ele)
                fp_predicted_values_span_aligned.append(mid_res)


            ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
                true_values_span_aligned=fp_true_values_span_aligned,
                predicted_values_span_aligned=fp_predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp, # this is different from PN version
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=None, # in ood setting, we need to remove the wrong_span_label
            )

            # below is for the wrong_span ture pred,
            ws_y_true, ws_y_pred, ws_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
                true_values_span_aligned=true_values_span_aligned,
                predicted_values_span_aligned=predicted_values_span_aligned,
                evaluation_label_dictionary=evaluation_label_dictionary,
                predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp, # this is different from PN version
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label=None, # in ood setting, we need to remove the wrong_span_label
            )

            # below is for the in-domain c-level semantic class ture pred,
            cls_y_true, cls_y_pred, cls_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
                true_values_span_aligned=cls_true_values_span_aligned,
                predicted_values_span_aligned=cls_predicted_values_span_aligned,
                evaluation_label_dictionary=default_evaluation_label_dictionary, # changed into c-class dict
                predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp,
                default_evaluation_label_dictionary=default_evaluation_label_dictionary,
                use_alpha=use_alpha,
                move_out_label='OOD', # in ood setting, we need to remove the wrong_span_label
            )


        # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        if use_alpha == False:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]

            y_pred_vec_alpha_or_sfmx_dp = [
                self.map_dict2vec_dp(default_evaluation_label_dictionary, predicted_scoredict_ins_dp, 'none') # 'none' can be 'mean'
                for predicted_scoredict_ins_dp in predicted_scoredict_span_aligned_dp
            ]  # use y_pred_vec_alpha_or_sfmx_dp for uncertainty estimation

        else:
            y_pred_vec_alpha_or_sfmx = [
                self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
                for predicted_scoredict_ins in predicted_scoredict_span_aligned
            ]




        # now, calculate evaluation numbers
        target_names = []
        labels = []

        counter = Counter(itertools.chain.from_iterable(cls_all_true_values.values()))
        counter.update(list(itertools.chain.from_iterable(cls_all_predicted_values.values())))

        for label_name, count in counter.most_common():
            if label_name == OOD_label or label_name == 'OOD':
                continue
            target_names.append(label_name)
            labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))

        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(cls_y_true, cls_y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        ood_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx_dp=ood_y_pred_vec_alpha_or_sfmx_dp,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='ood_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ood_unc_list)

        ws_unc_list = self.get_unc_score_list(
            y_true=ws_y_true,
            y_pred=ws_y_pred,
            y_pred_vec_alpha_or_sfmx_dp=ws_y_pred_vec_alpha_or_sfmx_dp,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='wrong_span_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(wrong_span_label)
        )
        writent_list.extend(ws_unc_list)

        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        print(write_file_name)

        self.train()  # restart training, after eval

        return result


    def ood_evaluate_token(
        self,
        data_points: Union[List[DT], Dataset],
        gold_label_type: str,
        out_path: Union[str, Path] = None,
        embedding_storage_mode: str = "none",
        mini_batch_size: int = 32,
        num_workers: Optional[int] = 8,
        main_evaluation_metric: Tuple[str, str] = ("micro avg", "f1-score"),
        exclude_labels: List[str] = [],
        gold_label_dictionary: Optional[Dictionary] = None,
        return_loss: bool = True,
        uncertainty_metrics: List[str]=[],
        test_dropout_num=1, #true_setting
        # test_dropout_num=3, #debug_uage
        use_alpha=False,
        sfmx_mode=None,
        leave_out_labels: List[str] = [],
        cal_unique_predict_scores: bool = True,
        use_var_metric=False,
        **kwargs,
    ) -> Result:
        import numpy as np
        import sklearn

        # make sure <unk> is contained in gold_label_dictionary, if given
        if gold_label_dictionary and not gold_label_dictionary.add_unk:
            raise AssertionError("gold_label_dictionary must have add_unk set to true in initialization.")

        # read Dataset into data loader, if list of sentences passed, make Dataset first
        if not isinstance(data_points, Dataset):
            data_points = FlairDatapointDataset(data_points)

        ## required in OOD
        assert cal_unique_predict_scores == True
        self.use_var_metric=use_var_metric
        self.shared_entity_num=0
        self.unique_gt_entity_num=0

        # self-added: initialize dict
        interv_dict = self.get_interv_dict(self.label_dictionary)

        default_score_dict = self.get_default_alpha_dict(interv_dict)
        default_score_dict_dp = self.get_default_alpha_dict_dp(interv_dict, test_dropout_num)

        with torch.no_grad():

            # loss calculation
            eval_loss = torch.zeros(1, device=flair.device)
            average_over = 0

            # variables for printing
            lines: List[str] = []

            # variables for computing scores
            all_spans: Set[str] = set()
            all_true_values = {}
            all_predicted_values = {}
            cls_all_true_values = {}
            cls_all_predicted_values = {}
            all_predicted_score_dict = {}
            all_predicted_score_dict_dp = {}
            if cal_unique_predict_scores:
                unique_gt_in_pd_score_dict = {}
                unique_gt_in_pd_score_dict_dp = {}
            ID_label, OOD_label = 'ID_label', 'OOD_label' # OOD and ID labels definition
            wrong_span_label = 'wrong_span_label'
            loader = DataLoader(data_points, batch_size=mini_batch_size, num_workers=0)

            # token:
            ood_y_true, ood_y_pred = [], []
            cls_y_true, cls_y_pred = [], []
            ood_y_pred_vec_alpha_or_sfmx = []
            ood_y_pred_vec_alpha_or_sfmx_dp = []

            sentence_id = 0

            for batch in Tqdm.tqdm(loader):

                # remove any previously predicted labels
                for datapoint in batch:
                    datapoint.remove_labels("predicted")

                for test_epoch in range(test_dropout_num):
                    if test_epoch != (test_dropout_num - 1):
                        self.train()
                    else:
                        self.eval()
                    # print(f'this is the test epoch {test_epoch}')
                    # predict for batch
                    loss_and_count = self.predict(
                        batch,
                        embedding_storage_mode=embedding_storage_mode,
                        mini_batch_size=mini_batch_size,
                        return_probabilities_for_all_classes=True, # self-added
                        label_name="predicted",
                        return_loss=return_loss,

                    )

                    # if test_epoch == (test_dropout_num - 1):
                    #     self.train()  # restart training, after eval

                if return_loss:
                    if isinstance(loss_and_count, tuple):
                        average_over += loss_and_count[1]
                        eval_loss += loss_and_count[0]
                    else:
                        eval_loss += loss_and_count


                for datapoint in batch:
                    # get the gold labels
                    # if cal_unique_predict_scores: unique_gold_representation_set = set()

                    token_num = len(datapoint)
                    mid_ood_y_true = [ID_label] * token_num
                    mid_ood_y_pred = [ID_label] * token_num
                    mid_cls_y_true = ['O'] * token_num
                    mid_cls_y_pred = ['O'] * token_num

                    for gold_label in datapoint.get_labels(gold_label_type):
                        # token:
                        total_representation, total_index = self.get_token_info(datapoint, sentence_id, gold_label.unlabeled_identifier)
                        for index in range(len(total_index)):
                            representation = str(sentence_id) + ": " + total_representation[index]
                            y_index = total_index[index]


                        # if cal_unique_predict_scores: unique_gold_representation_set.add(representation)

                        mid_cls_y_true[y_index] = gold_label.value  ## different from misclassification
                        if gold_label.value != 'OOD':
                            mid_ood_y_true[y_index] = ID_label
                        else:
                            mid_ood_y_true[y_index] = OOD_label

                        # # value = gold_label.value
                        # cls_value = gold_label.value ## different from misclassification
                        # if gold_label.value != 'OOD':
                        #     value = ID_label
                        # else:
                        #     value = OOD_label
                        #
                        # if gold_label_dictionary and gold_label_dictionary.get_idx_for_item(value) == 0:
                        #     value = "<unk>"

                        # if representation not in all_true_values:
                        #     all_true_values[representation] = [value]  # self-comment: need to check whether 'O' will be considered in the list
                        #     cls_all_true_values[representation] = [cls_value]
                        # else:
                        #     all_true_values[representation].append(value)
                        #     cls_all_true_values[representation].append(cls_value)
                        #
                        # if representation not in all_spans:
                        #     all_spans.add(representation)

                    ood_y_true.extend(mid_ood_y_true)
                    cls_y_true.extend(mid_cls_y_true)

                    # get the predicted labels
                    if len(datapoint.get_labels("predicted")) != 0:
                        # print(datapoint.get_labels("predicted"))
                        pass

                    # transfer datapoint with multiple dropout results into a mean result and extract the multiple dropout results
                    ensemble_pre_key, ori_pre_key, processed_ensb_pre_key = self.extract_multi_dp_prob(datapoint, test_dropout_num)

                    for predicted_span in datapoint.get_labels("predicted"): # self-comment: seems like there is no predicted tag or because all labels are 'O'
                        # token:
                        total_representation, total_index = self.get_token_info(datapoint, sentence_id,
                                                                                           predicted_span.unlabeled_identifier)

                        for index in range(len(total_index)):
                            representation = str(sentence_id) + ": " + total_representation[index]
                            y_index = total_index[index]

                        # remove the existing representation, if the representation is not unique in the ground-truth
                        # if cal_unique_predict_scores:
                        #     if representation in unique_gold_representation_set:
                        #         unique_gold_representation_set.remove(representation)

                        # cls_predicted_span_value = predicted_span.value
                        # predicted_span_value = ID_label  # predicted_span.value

                        # mid_cls_y_pred[y_index] = predicted_span.value
                        # mid_ood_y_pred[y_index] = ID_label
                        #
                        # # add to all_predicted_values
                        # if representation not in all_predicted_values:
                        #     all_predicted_values[representation] = [predicted_span_value] # self-comment: need recalculate after [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict)]
                        #     # all_predicted_score_dict[representation] = [self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, cal_dp=True)]
                        #     all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(
                        #         datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key,
                        #         processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                        #         cal_dp=True)
                        #
                        #     all_predicted_score_dict[representation], all_predicted_score_dict_dp[representation] = [all_predicted_score_dict_ele], [all_predicted_score_dict_dp_ele]
                        #     cls_all_predicted_values[representation] = [cls_predicted_span_value]
                        #     # 将最后一个model epoch 进行 training 关闭操作 finished!
                        #     # 根据最后一个 model epoch 预测的结果，或者 统一 的 对 各个 dp res 进行 merge_beios_to_one_score 的操作（改内部的关键字） # almost finished
                        #     # 将 merge 后的 phrase-level prob mat 用于 testing process 计算
                        #
                        # else:
                        #     all_predicted_values[representation].append(predicted_span_value)
                        #     # all_predicted_score_dict[representation].append(self.merge_beios_to_one_score(datapoint,predicted_span, interv_dict, cal_dp=True))
                        #     all_predicted_score_dict_ele, all_predicted_score_dict_dp_ele = self.merge_beios_to_one_score(datapoint, predicted_span, interv_dict, ori_pre_key=ori_pre_key, processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num, cal_dp=True)
                        #     all_predicted_score_dict[representation].append(all_predicted_score_dict_ele)
                        #     all_predicted_score_dict_dp[representation].append(all_predicted_score_dict_dp_ele)
                        #     cls_all_predicted_values[representation].append(cls_predicted_span_value)
                        #
                        # if representation not in all_spans:
                        #     all_spans.add(representation)

                    # ### calculate the unique predicted entities score vec accroding to ground-truth index
                    # if cal_unique_predict_scores:
                    #     for rep in unique_gold_representation_set:
                    #         mid_unique_gt_in_pd_score_dict_ele, mid_unique_gt_in_pd_score_dict_dp_ele = self.merge_beios_to_one_score(
                    #                 datapoint, rep, interv_dict, ori_pre_key=ori_pre_key,
                    #                 processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num,
                    #                 cal_dp=True)
                    #         if rep not in unique_gt_in_pd_score_dict.keys():
                    #             # unique_gt_in_pd_score_dict[rep] = [self.merge_beios_to_one_score(datapoint, rep, interv_dict)] # for non-dp
                    #             unique_gt_in_pd_score_dict[rep],unique_gt_in_pd_score_dict_dp[rep] = [mid_unique_gt_in_pd_score_dict_ele], [mid_unique_gt_in_pd_score_dict_dp_ele]
                    #         else:
                    #             # unique_gt_in_pd_score_dict[rep].append([self.merge_beios_to_one_score(datapoint, rep, interv_dict)]) # for non-dp
                    #             unique_gt_in_pd_score_dict[rep].append(mid_unique_gt_in_pd_score_dict_ele)
                    #             unique_gt_in_pd_score_dict_dp[rep].append(mid_unique_gt_in_pd_score_dict_dp_ele)

                    ### (to do) need to extract the dropout_versioned scores for each token!
                    ### mid_ood_y_pred_vec_alpha_or_sfmx-> mean of mid_ood_y_pred_vec_alpha_or_sfmx_dp (con@here)
                    mid_ood_y_pred_vec_alpha_or_sfmx, mid_ood_y_pred_vec_alpha_or_sfmx_dp  = self.extract_token_merged_score(datapoint, interv_dict, ori_pre_key=ori_pre_key, processed_ensb_pre_key=processed_ensb_pre_key, test_dropout_num=test_dropout_num, cal_dp=True)
                    ood_y_pred.extend(mid_ood_y_pred)
                    cls_y_pred.extend(mid_cls_y_pred)
                    ood_y_pred_vec_alpha_or_sfmx.extend(mid_ood_y_pred_vec_alpha_or_sfmx)
                    ood_y_pred_vec_alpha_or_sfmx_dp.extend(mid_ood_y_pred_vec_alpha_or_sfmx_dp)

                    sentence_id += 1

                store_embeddings(batch, embedding_storage_mode)

                # make printout lines
                if out_path:
                    lines.extend(self._print_predictions(batch, gold_label_type))

            new_cls_y_true, new_cls_y_pred = [], []
            for idx in range(len(cls_y_true)):
                if cls_y_true[idx] == OOD_label or cls_y_true[idx] == 'OOD':
                    continue
                new_cls_y_true.append(cls_y_true[idx])
                new_cls_y_pred.append(cls_y_pred[idx])
            cls_y_true, cls_y_pred = new_cls_y_true, new_cls_y_pred

            # # convert true and predicted values to two span-aligned lists
            # true_values_span_aligned = []
            # predicted_values_span_aligned = []
            # predicted_scoredict_span_aligned = [] # self-added
            # predicted_scoredict_span_aligned_dp = [] # self-added
            # cls_true_values_span_aligned = []
            # cls_predicted_values_span_aligned = []
            # for span in all_spans:
            #     # self-added: count shared_num
            #     if span in all_true_values and span in all_predicted_values:
            #         self.shared_entity_num += 1
            #     if span in all_true_values and span not in all_predicted_values:
            #         self.unique_gt_entity_num += 1
            #
            #     list_of_gold_values_for_span = all_true_values[span] if span in all_true_values else [wrong_span_label] # self-comment: need change for OOD detection metric
            #     cls_list_of_gold_values_for_span = cls_all_true_values[span] if span in cls_all_true_values else ["O"]
            #     # delete exluded labels if exclude_labels is given
            #     for excluded_label in exclude_labels:
            #         if excluded_label in list_of_gold_values_for_span:
            #             list_of_gold_values_for_span.remove(excluded_label)
            #             cls_list_of_gold_values_for_span.remove(excluded_label)
            #     # if after excluding labels, no label is left, ignore the datapoint
            #     if not list_of_gold_values_for_span:
            #         continue
            #     true_values_span_aligned.append(list_of_gold_values_for_span)
            #     cls_true_values_span_aligned.append(cls_list_of_gold_values_for_span)
            #     predicted_values_span_aligned.append(
            #         all_predicted_values[span] if span in all_predicted_values else [wrong_span_label] # self-comment: need change for OOD detection metric
            #     )
            #     cls_predicted_values_span_aligned.append(
            #         cls_all_predicted_values[span] if span in cls_all_predicted_values else ["O"]
            #     )
            #
            #
            #     if cal_unique_predict_scores:
            #         if span in all_predicted_values:
            #             to_add_predicted_scoredict_span_aligned = all_predicted_score_dict[span]
            #             to_add_predicted_scoredict_span_aligned_dp = all_predicted_score_dict_dp[span]
            #         else:
            #             to_add_predicted_scoredict_span_aligned = unique_gt_in_pd_score_dict[span]
            #             to_add_predicted_scoredict_span_aligned_dp = unique_gt_in_pd_score_dict_dp[span]
            #         predicted_scoredict_span_aligned.append(to_add_predicted_scoredict_span_aligned)
            #         predicted_scoredict_span_aligned_dp.append(to_add_predicted_scoredict_span_aligned_dp)
            #     else:
            #         predicted_scoredict_span_aligned.append(
            #             all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # need to revise # (to do) need to reviese by calcualted_socore_dict
            #         )
            #         predicted_scoredict_span_aligned_dp.append(
            #             # a variable with '_dp' means it is/includes a list of repeated results by dropout
            #             all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp
            #         )


                # predicted_scoredict_span_aligned.append(
                #     all_predicted_score_dict[span] if span in all_predicted_values else default_score_dict # (to do) need to reviese by calcualted_socore_dict
                # )
                # # add dp result at here: all_predicted_score_dict_dp[representation]
                # predicted_scoredict_span_aligned_dp.append(   # a variable with '_dp' means it is/includes a list of repeated results by dropout
                #     all_predicted_score_dict_dp[span] if span in all_predicted_values else default_score_dict_dp # (to do) need to reviese by calcualted_socore_dict
                # )

                # For [span] span-level uncertainty, need to calculate the span uncertainty score vector. # e.g. '524: Span[12:14]: "three chimneys"'
                  # merge 17 dimension to 7 dimension

            # write all_predicted_values to out_file if set
            if out_path:
                with open(Path(out_path), "a+", encoding="utf-8") as outfile:
                    outfile.write("".join(lines))

            # # make the evaluation dictionary
            # evaluation_label_dictionary = Dictionary(add_unk=False)
            # evaluation_label_dictionary.add_item("O") # self-comment: need change for OOD detection metric
            # for true_values in all_true_values.values():
            #     for label in true_values:
            #         evaluation_label_dictionary.add_item(label)
            # for predicted_values in all_predicted_values.values():
            #     for label in predicted_values:
            #         evaluation_label_dictionary.add_item(label)

            # make the evaluation dictionary
            evaluation_label_dictionary = Dictionary(add_unk=False)
            evaluation_label_dictionary.add_item(
                wrong_span_label)  # self-comment: need change for OOD detection metric
            # for true_values in all_true_values.values():
            #     for label in true_values:
            #         evaluation_label_dictionary.add_item(label)
            # for predicted_values in all_predicted_values.values():
            #     for label in predicted_values:
            #         evaluation_label_dictionary.add_item(label)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords
            #
            # default_evaluation_label_dictionary = Dictionary(add_unk=False)
            # for label in default_score_dict.keys():
            #     default_evaluation_label_dictionary.add_item(label)  # (check) expect default_evaluation_label_dictionary in 1 + c keywords
            for true_values in ood_y_true:  # token
                evaluation_label_dictionary.add_item(true_values)
            for predicted_values in all_predicted_values.values():
                evaluation_label_dictionary.add_item(predicted_values)  # (check) expect evaluation_label_dictionary in [ws, id, ood] keywords

            default_evaluation_label_dictionary = Dictionary(add_unk=False)
            for label in default_score_dict.keys():
                default_evaluation_label_dictionary.add_item(label) # (check) expect default_evaluation_label_dictionary in 1 + c keywords

        # check if this is a multi-label problem
        multi_label = False
        # for true_instance, predicted_instance in zip(true_values_span_aligned, predicted_values_span_aligned):
        #     if len(true_instance) > 1 or len(predicted_instance) > 1:
        #         multi_label = True
        #         break

        log.info(f"Evaluating as a multi-label problem: {multi_label}")

        # # compute numbers by formatting true and predicted such that Scikit-Learn can use them
        # y_true = []
        # y_pred = []
        # if multi_label:
        #     # multi-label problems require a multi-hot vector for each true and predicted label
        #     for true_instance in true_values_span_aligned:
        #         y_true_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
        #         for true_value in true_instance:
        #             y_true_instance[evaluation_label_dictionary.get_idx_for_item(true_value)] = 1
        #         y_true.append(y_true_instance.tolist())
        #
        #     for predicted_values in predicted_values_span_aligned:
        #         y_pred_instance = np.zeros(len(evaluation_label_dictionary), dtype=int)
        #         for predicted_value in predicted_values:
        #             y_pred_instance[evaluation_label_dictionary.get_idx_for_item(predicted_value)] = 1
        #         y_pred.append(y_pred_instance.tolist())
        # else:
        #     # single-label problems can do with a single index for each true and predicted label
        #     y_true = [
        #         evaluation_label_dictionary.get_idx_for_item(true_instance[0])
        #         for true_instance in true_values_span_aligned
        #     ] # y_true is the label ID list for the ture tokens (concatenationg all sentences)
        #     y_pred = [
        #         evaluation_label_dictionary.get_idx_for_item(predicted_instance[0])
        #         for predicted_instance in predicted_values_span_aligned
        #     ]  # y_pred is the label ID list for the predicted tokens (concatenationg all sentences)
        #
        #     # process to transfer ood+ws -> ood
        #     # true: ws -> ood
        #     # predict: ws -> id
        #     fp_true_values_span_aligned = []
        #     fp_predicted_values_span_aligned = []
        #     for ele in true_values_span_aligned:
        #         mid_res = []
        #         for sub_ele in ele:
        #             if sub_ele == wrong_span_label:
        #                 mid_res.append(OOD_label)
        #             else:
        #                 mid_res.append(sub_ele)
        #         fp_true_values_span_aligned.append(mid_res)
        #     for ele in predicted_values_span_aligned:
        #         mid_res = []
        #         for sub_ele in ele:
        #             if sub_ele == wrong_span_label:
        #                 mid_res.append(ID_label)
        #             else:
        #                 mid_res.append(sub_ele)
        #         fp_predicted_values_span_aligned.append(mid_res)

        y_pred_vec_alpha_or_sfmx_dp = []

        for index in range(len(ood_y_pred_vec_alpha_or_sfmx)): # need revision (to do) (continue@here)
            if use_alpha == False:
                y_pred_vec_alpha_or_sfmx_dp.append(
                    self.map_dict2vec_dp(default_evaluation_label_dictionary, ood_y_pred_vec_alpha_or_sfmx_dp[index], 'none'))
            else:
                y_pred_vec_alpha_or_sfmx_dp.append(
                    self.map_dict2vec_dp(default_evaluation_label_dictionary, ood_y_pred_vec_alpha_or_sfmx_dp[index],'none'))



        #     ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
        #         true_values_span_aligned=fp_true_values_span_aligned,
        #         predicted_values_span_aligned=fp_predicted_values_span_aligned,
        #         evaluation_label_dictionary=evaluation_label_dictionary,
        #         predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp, # this is different from PN version
        #         default_evaluation_label_dictionary=default_evaluation_label_dictionary,
        #         use_alpha=use_alpha,
        #         move_out_label=None, # in ood setting, we need to remove the wrong_span_label
        #     )
        #
        #     # below is for the wrong_span ture pred,
        #     ws_y_true, ws_y_pred, ws_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
        #         true_values_span_aligned=true_values_span_aligned,
        #         predicted_values_span_aligned=predicted_values_span_aligned,
        #         evaluation_label_dictionary=evaluation_label_dictionary,
        #         predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp, # this is different from PN version
        #         default_evaluation_label_dictionary=default_evaluation_label_dictionary,
        #         use_alpha=use_alpha,
        #         move_out_label=None, # in ood setting, we need to remove the wrong_span_label
        #     )
        #
        #     # below is for the in-domain c-level semantic class ture pred,
        #     cls_y_true, cls_y_pred, cls_y_pred_vec_alpha_or_sfmx_dp = self.get_true_pred_vectors(
        #         true_values_span_aligned=cls_true_values_span_aligned,
        #         predicted_values_span_aligned=cls_predicted_values_span_aligned,
        #         evaluation_label_dictionary=default_evaluation_label_dictionary, # changed into c-class dict
        #         predicted_scoredict_span_aligned_dp=predicted_scoredict_span_aligned_dp,
        #         default_evaluation_label_dictionary=default_evaluation_label_dictionary,
        #         use_alpha=use_alpha,
        #         move_out_label='OOD', # in ood setting, we need to remove the wrong_span_label
        #     )
        #
        #
        # # consider use y_true, y_pred, and adding their softmax vectors for the AUROC and AUPR calculation
        # if use_alpha == False:
        #     y_pred_vec_alpha_or_sfmx = [
        #         self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can be 'mean'
        #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     ]
        #
        #     y_pred_vec_alpha_or_sfmx_dp = [
        #         self.map_dict2vec_dp(default_evaluation_label_dictionary, predicted_scoredict_ins_dp, 'none') # 'none' can be 'mean'
        #         for predicted_scoredict_ins_dp in predicted_scoredict_span_aligned_dp
        #     ]  # use y_pred_vec_alpha_or_sfmx_dp for uncertainty estimation
        #
        # else:
        #     y_pred_vec_alpha_or_sfmx = [
        #         self.map_dict2vec(default_evaluation_label_dictionary, predicted_scoredict_ins, 'none') # 'none' can only be 'none'?
        #         for predicted_scoredict_ins in predicted_scoredict_span_aligned
        #     ]

        # # now, calculate evaluation numbers
        # target_names = []
        # labels = []
        #
        # counter = Counter(itertools.chain.from_iterable(cls_all_true_values.values()))
        # counter.update(list(itertools.chain.from_iterable(cls_all_predicted_values.values())))
        #
        # for label_name, count in counter.most_common():
        #     if label_name == OOD_label or label_name == 'OOD':
        #         continue
        #     target_names.append(label_name)
        #     labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))

        target_names = set(cls_y_true)
        target_names = list(target_names)
        labels = []
        for label_name in target_names:
            labels.append(default_evaluation_label_dictionary.get_idx_for_item(label_name))

        # token: transfer from str to id
        cls_y_true, cls_y_pred = self.transfer_str_to_id(cls_y_true, cls_y_pred, default_evaluation_label_dictionary)
        ood_y_true, ood_y_true = self.transfer_str_to_id(ood_y_true, ood_y_true, evaluation_label_dictionary)





        # there is at least one gold label or one prediction (default)
        if len(all_true_values) + len(all_predicted_values) > 1:
            classification_report = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                digits=4,
                target_names=target_names,
                zero_division=0,
                labels=labels,
            )

            classification_report_dict = sklearn.metrics.classification_report(
                cls_y_true,
                cls_y_pred,
                target_names=target_names,
                zero_division=0,
                output_dict=True,
                labels=labels,
            )

            accuracy_score = round(sklearn.metrics.accuracy_score(cls_y_true, cls_y_pred), 4)
            macro_f_score = round(classification_report_dict["macro avg"]["f1-score"], 4)

            # if there is only one label, then "micro avg" = "macro avg"
            if len(target_names) == 1:
                classification_report_dict["micro avg"] = classification_report_dict["macro avg"]

            if "micro avg" in classification_report_dict:
                # micro average is only computed if zero-label exists (for instance "O")
                precision_score = round(classification_report_dict["micro avg"]["precision"], 4)
                recall_score = round(classification_report_dict["micro avg"]["recall"], 4)
                micro_f_score = round(classification_report_dict["micro avg"]["f1-score"], 4)
            else:
                # if no zero-label exists (such as in POS tagging) micro average is equal to accuracy
                precision_score = round(classification_report_dict["accuracy"], 4)
                recall_score = round(classification_report_dict["accuracy"], 4)
                micro_f_score = round(classification_report_dict["accuracy"], 4)

            # same for the main score
            if "micro avg" not in classification_report_dict and main_evaluation_metric[0] == "micro avg":
                main_score = classification_report_dict["accuracy"]
            else:
                main_score = classification_report_dict[main_evaluation_metric[0]][main_evaluation_metric[1]]

        else:
            # issue error and default all evaluation numbers to 0.
            log.error(
                "ACHTUNG! No gold labels and no all_predicted_values found! "
                "Could be an error in your corpus or how you "
                "initialize the trainer!"
            )
            accuracy_score = precision_score = recall_score = micro_f_score = macro_f_score = main_score = 0.0
            classification_report = ""
            classification_report_dict = {}

        detailed_result = (
            f"\nResults at {datetime.now()}"
            f"\n- F-score (micro) {micro_f_score}"
            f"\n- F-score (macro) {macro_f_score}\n"
            # f"\n- Accuracy {accuracy_score}"
            # "\n\nBy class:\n" + classification_report ## This is a detailed classification report
        )

        # line for log file
        log_header = "PRECISION\tRECALL\tF1\tACCURACY"
        log_line = f"{precision_score}\t" f"{recall_score}\t" f"{micro_f_score}\t" f"{accuracy_score}"

        if average_over > 0:
            eval_loss /= average_over

        writent_list = [detailed_result, log_header, log_line]

        result = Result(
            main_score=main_score,
            log_line=log_line,
            log_header=log_header,
            detailed_results=detailed_result,
            classification_report=classification_report_dict,
            loss=eval_loss.item(),
        )

        ood_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx_dp=y_pred_vec_alpha_or_sfmx_dp,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='ood_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ood_unc_list)

        ws_unc_list = self.get_unc_score_list(
            y_true=ood_y_true,
            y_pred=ood_y_pred,
            y_pred_vec_alpha_or_sfmx_dp=y_pred_vec_alpha_or_sfmx_dp,
            use_alpha=use_alpha,
            uncertainty_metrics=uncertainty_metrics,
            sfmx_mode=sfmx_mode,
            task_keyword='wrong_span_',
            auroc_aupr_pos_id=evaluation_label_dictionary.get_idx_for_item(OOD_label)
        )
        writent_list.extend(ws_unc_list)

        if cal_unique_predict_scores:
            write_file_name = str(out_path.absolute())[:-8] + 'unc_cal.txt'
        else:
            write_file_name = str(out_path.absolute())[:-8] + 'unc.txt'


        with open(write_file_name, 'a+') as ff:
            ff.writelines(writent_list)
        print(write_file_name)

        self.train()  # restart training, after eval

        return result

    def transfer_str_to_id(self, true_values_span_aligned, predicted_scoredict_span_aligned, evaluation_label_dictionary):
        ood_y_true = []
        ood_y_pred = []
        for ind in range(len(true_values_span_aligned)):
            # ele = true_values_span_aligned[ind]
            # predicted_scoredict_ins = predicted_scoredict_span_aligned[ind]
            ood_y_true.append(evaluation_label_dictionary.get_idx_for_item(true_values_span_aligned[ind]))
            ood_y_pred.append(evaluation_label_dictionary.get_idx_for_item(predicted_scoredict_span_aligned[ind]))
        return ood_y_pred, ood_y_pred
    

    def get_token_info(self, datapoint, sentence_id, unlabeled_identifier):
        reprensentation_list = []
        index_list = []
        try:
            un_identifier = unlabeled_identifier  # e.g: 'Span[1:2]: "asian"'
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            comma_idx = un_identifier.index(':')
        except:
            un_identifier = str(unlabeled_identifier)
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            new_str = un_identifier[lf_braket_idx:rg_braket_idx]
            new_comma_idx = new_str.index(':')
            comma_idx = new_comma_idx + lf_braket_idx
        low_idx = int(un_identifier[lf_braket_idx + 1:comma_idx])
        high_idx = int(un_identifier[comma_idx + 1:rg_braket_idx])

        for i in range(low_idx, high_idx):
            cur_rep = str(sentence_id) + ": [" + str(i) + "]" + datapoint[i].text
            reprensentation_list.append(cur_rep)
            index_list.append(i)
        return reprensentation_list, index_list


    def get_true_pred_vectors(self,
                              true_values_span_aligned,  # list of ground_truth labels in string
                              predicted_values_span_aligned, # list of predicted labels in string
                              evaluation_label_dictionary, # a dict of interested of dictionary (could be different from c-semantic-class dict)
                              predicted_scoredict_span_aligned_dp, # a list of dropout_version c-semantic class vector
                              default_evaluation_label_dictionary, # a dict of c-semantic class dict
                              use_alpha, # whether use alpha or not
                              move_out_label=None # the case of moving out label (e.g. OOD-> remove wrong_span)
                              ):
        if move_out_label is not None:
            assert move_out_label=="OOD" or move_out_label in evaluation_label_dictionary.get_items()
        ood_y_true = []
        ood_y_pred = []
        ood_y_pred_vec_alpha_or_sfmx_dp = []
        for ind in range(len(true_values_span_aligned)):
            ele = true_values_span_aligned[ind]
            predicted_scoredict_ins_dp = predicted_scoredict_span_aligned_dp[ind]
            if move_out_label is None or (move_out_label is not None and ele[0] != move_out_label):
                ood_y_true.append(evaluation_label_dictionary.get_idx_for_item(true_values_span_aligned[ind][0]))
                ood_y_pred.append(evaluation_label_dictionary.get_idx_for_item(predicted_values_span_aligned[ind][0]))
                if use_alpha == False:
                    ood_y_pred_vec_alpha_or_sfmx_dp.append(
                        self.map_dict2vec_dp(default_evaluation_label_dictionary, predicted_scoredict_ins_dp, 'none'))
                else:
                    ood_y_pred_vec_alpha_or_sfmx_dp.append(
                        self.map_dict2vec_dp(default_evaluation_label_dictionary, predicted_scoredict_ins_dp, 'none'))

        return ood_y_true, ood_y_pred, ood_y_pred_vec_alpha_or_sfmx_dp

    def get_unc_score_list(self,
                           y_true, # the ground-truth label list
                           y_pred, # the predicted label list
                           y_pred_vec_alpha_or_sfmx_dp, # a list of dropout-version c-class probability vector
                           use_alpha, # whether use alpha (e.g PN-based) or not (e.g. ensemble).
                           uncertainty_metrics, # a list of considered uncertainty metrics
                           sfmx_mode=None, # the softmax (normalization) mode used in aleatoric unc
                           task_keyword=None, # the name of task ['ood_', 'wrong_span_', '']
                           auroc_aupr_pos_id=None, # if postive id exist, use the postive id for evalue; else use the (y_true==y_pred) for the binary id
                           ):
        writent_list = []
        if task_keyword not in ['', 'ood_', 'wrong_span_']:
            raise ValueError(f"the task_keywod={task_keyword} is not in ['', 'ood_', 'wrong_span_']")
        assert use_alpha==False

        if task_keyword == 'wrong_span_':
            print(f'auroc_aupr_pos_id={auroc_aupr_pos_id}')
            # assert auroc_aupr_pos_id not in y_true
            writent_list.append(f'total(three parts) entities are {len(y_true)}\n')
            print(f'total(three parts) entities are {len(y_true)}')
            ws_num_jsq = 0
            for ele1 in y_true:
                if ele1 == auroc_aupr_pos_id:
                    ws_num_jsq += 1
            writent_list.append(f'total(wrong_span part) entities are {ws_num_jsq}\n')
            print(f'total(wrong_span part) entities are {ws_num_jsq}')
            writent_list.append(f'total(shared part) entities are {self.shared_entity_num}\n')
            print(f'total(shared part) entities are {self.shared_entity_num}')
            writent_list.append(f'total(unique_gt part) entities are {self.unique_gt_entity_num}\n')
            print(f'total(unique_gt part) entities are {self.unique_gt_entity_num}')

        y_pred_vec_alpha_or_sfmx_dp = np.array(y_pred_vec_alpha_or_sfmx_dp).transpose(2, 0, 1)
        if 'entr_sfmx' in uncertainty_metrics:
            entr_unc, class_entr_unc = entropy_dropout(y_pred_vec_alpha_or_sfmx_dp)
            entr_unc = entr_unc.squeeze(axis=1)
            entr_unc = entr_unc.tolist()
            # entr_unc, class_entr_unc = entropy_dropout(y_pred_vec_alpha_or_sfmx_dp)
            entr_auroc, entr_aupr, entr_wrlist = get_auroc_aupr(y_true, y_pred, entr_unc, keyword=task_keyword+'entr_mean', auroc_aupr_pos_id=auroc_aupr_pos_id)
            # it is actually 'entr_sfmx'
            writent_list.extend(entr_wrlist)
        if 'alea' in uncertainty_metrics:
            # for misclassification
            alea_unc, class_alea_unc = aleatoric_dropout(y_pred_vec_alpha_or_sfmx_dp)

            alea_unc = alea_unc.squeeze(axis=1)
            alea_unc = alea_unc.tolist()

            alea_auroc, alea_aupr, alea_wrlist = get_auroc_aupr(y_true, y_pred, alea_unc, keyword=task_keyword+'alea_sfmx', auroc_aupr_pos_id=auroc_aupr_pos_id)

            writent_list.extend(alea_wrlist)

        if 'epis' in uncertainty_metrics:
            dr_eps_class = class_entr_unc - class_alea_unc
            epis_unc = np.sum(dr_eps_class, axis=1, keepdims=True)

            epis_unc = epis_unc.squeeze(axis=1)
            epis_unc = epis_unc.tolist()

            epis_auroc, epis_aupr, epis_wrlist = get_auroc_aupr(y_true, y_pred, epis_unc, keyword=task_keyword+'epis', auroc_aupr_pos_id=auroc_aupr_pos_id)
            writent_list.extend(epis_wrlist)

        # writen for read usage only
        for manual_keyword in ['alea_mean', 'diss', 'vacu']:
            writent_list.append("\n")
            writent_list.append(f"{task_keyword+manual_keyword} auroc is 0\n")
            writent_list.append(f"{task_keyword+manual_keyword} aupr is 0\n")




        return writent_list


    @abstractmethod
    def predict(
        self,
        sentences: Union[List[DT], DT],
        mini_batch_size: int = 32,
        return_probabilities_for_all_classes: bool = False,
        verbose: bool = False,
        label_name: Optional[str] = None,
        return_loss=False,
        embedding_storage_mode="none",
    ):
        """
        Predicts the class labels for the given sentences. The labels are directly added to the sentences.  # noqa: E501
        :param sentences: list of sentences
        :param mini_batch_size: mini batch size to use
        :param return_probabilities_for_all_classes : return probabilities for all classes instead of only best predicted  # noqa: E501
        :param verbose: set to True to display a progress bar
        :param return_loss: set to True to return loss
        :param label_name: set this to change the name of the label type that is predicted  # noqa: E501
        :param embedding_storage_mode: default is 'none' which is always best. Only set to 'cpu' or 'gpu' if you wish to not only predict, but also keep the generated embeddings in CPU or GPU memory respectively. 'gpu' to store embeddings in GPU memory.  # noqa: E501
        """
        raise NotImplementedError

    def _print_predictions(self, batch, gold_label_type):
        lines = []
        for datapoint in batch:
            # check if there is a label mismatch
            g = [label.labeled_identifier for label in datapoint.get_labels(gold_label_type)]
            p = [label.labeled_identifier for label in datapoint.get_labels("predicted")]
            g.sort()
            p.sort()
            correct_string = " -> MISMATCH!\n" if g != p else ""
            # print info
            eval_line = (
                f"{datapoint.to_original_text()}\n"
                f" - Gold: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels(gold_label_type))}\n"
                f" - Pred: {', '.join(label.value if label.data_point == datapoint else label.labeled_identifier for label in datapoint.get_labels('predicted'))}\n{correct_string}\n"
            )
            lines.append(eval_line)
        return lines

    def cal_max_score_in_interv_ori(self, token, interv_dict):
        token_cat_num = len(token.tags_proba_dist['predicted'])
        token_info = token.tags_proba_dist['predicted']
        token_val_scr_dict = {}

        token_merged_score_dict = {}

        for i in range(token_cat_num):
            token_val_scr_dict[token_info[i].value] = token_info[i].score

        # interv_dict = {
        #     'Location': ['B-Location', 'I-Location', 'E-Location', 'S-Location'],
        # }

        interv_dict_key_list = list(interv_dict.keys())
        for j in range(len(interv_dict_key_list)):
            cur_interv_score_list = []
            cur_key = interv_dict_key_list[j]
            for k in range(len(interv_dict[cur_key])):
                cur_interv_score_list.append(token_val_scr_dict[interv_dict[cur_key][k]])
            cur_interv_max_score = max(cur_interv_score_list)   # process to get the score for an interv
            token_merged_score_dict[cur_key] = cur_interv_max_score

        return token_merged_score_dict




    def get_interv_dict(self, label_dictionary):
        micro_label_list = label_dictionary.get_items()
        macro_label_list = set()
        for ele in micro_label_list:
            if ele != 'O':
                macro_label_list.add(ele[2:])
            else:
                macro_label_list.add(ele)
        macro_label_list = list(macro_label_list)

        interv_dict = {}

        for mac_ele in macro_label_list:
            interv_dict[mac_ele] = []
            for mic_ele in micro_label_list:
                if mac_ele in mic_ele:
                    interv_dict[mac_ele].append(mic_ele)

        return interv_dict

    def get_default_score_dict(self, interv_dict):
        default_score_dict = {}
        for key in interv_dict.keys():
            if key == 'O':
                val = 1.0
            else:
                val = 1.0
            default_score_dict[key] = val
        return default_score_dict

    def get_default_alpha_dict(self, interv_dict):
        default_score_dict = {}
        for key in interv_dict.keys():
            val = 1.0
            default_score_dict[key] = val
        return default_score_dict

    def get_default_alpha_dict_dp(self, interv_dict, test_dp_times):
        default_score_dict = {}
        for key in interv_dict.keys():
            val = 1.0
            default_score_dict[key] = [val] * test_dp_times
        return default_score_dict

    def map_dict2vec(self, evaluation_label_dictionary, predicted_scoredict_ins, method='none'):
        assert method == 'none'
        y_pred_instance_vec = np.zeros(len(evaluation_label_dictionary), dtype=float)
        # print(predicted_scoredict_ins)
        try:
            if len(predicted_scoredict_ins) == 1:
                predicted_scoredict_ins = predicted_scoredict_ins[0]
        except:
            pass
        for key in list(predicted_scoredict_ins.keys()): # check alpha use None or not
            # print(evaluation_label_dictionary)
            # print(key) # few samples are too little to be seen.
            y_pred_instance_vec[evaluation_label_dictionary.get_idx_for_item(key)] = predicted_scoredict_ins[key]
        if method == 'mean':
            y_pred_instance_vec = y_pred_instance_vec/y_pred_instance_vec.sum()
        elif method == 'none':
            pass
        else:
            raise ValueError('the method choice is wrong!')
        return y_pred_instance_vec.tolist()

    def map_dict2vec_dp(self, evaluation_label_dictionary, predicted_scoredict_ins, method='none'):
        assert method == 'none'
        y_pred_instance_vec = [0] * len(evaluation_label_dictionary)
        # print(predicted_scoredict_ins)
        try:
            if len(predicted_scoredict_ins) == 1:
                predicted_scoredict_ins = predicted_scoredict_ins[0]
        except:
            pass
        for key in list(predicted_scoredict_ins.keys()): # check alpha use None or not
            # print(evaluation_label_dictionary)
            # print(key) # few samples are too little to be seen.
            y_pred_instance_vec[evaluation_label_dictionary.get_idx_for_item(key)] = predicted_scoredict_ins[key]
        if method == 'mean': # need revision
            y_pred_instance_vec = y_pred_instance_vec/y_pred_instance_vec.sum()
        elif method == 'none':
            pass
        else:
            raise ValueError('the method choice is wrong!')
        y_pred_instance_vec = np.array(y_pred_instance_vec).tolist()
        # y_pred_instance_vec = np.array(y_pred_instance_vec)
        return y_pred_instance_vec

    def build_label_dict(self, str_label_list):
        str_label_set = set(str_label_list)
        ref_str_label_list = list(str_label_set)
        label2id = {}
        id2label = {}
        for i in range(len(ref_str_label_list)):
            label2id[ref_str_label_list[i]] = i
            id2label[i] = ref_str_label_list[i]
        return label2id, id2label


    def extract_multi_dp_prob(self, datapoint, test_dp_num):
        # transfer datapoint with multiple dropout results into a mean result and extract the multiple dropout results
        extracted_multi_dp_prob = []

        ensemble_pre_key = 'ensemble_predicted'
        ori_pre_key = 'predicted'
        processed_ensb_pre_key = 'preocessed_ensemble_predicted'


        for token in datapoint.tokens:
            token_cat_num = len(token.tags_proba_dist[ensemble_pre_key][0])
            extracted_multi_dp_prob_list = []
            for k in range(token_cat_num):
                mid_extracted_multi_dp_prob_list = [token.tags_proba_dist[ensemble_pre_key][j][k].score for j in range(test_dp_num)]
                mid_extracted_multi_dp_prob_val = np.array(mid_extracted_multi_dp_prob_list).mean()
                token.tags_proba_dist[ori_pre_key][k].score = mid_extracted_multi_dp_prob_val   # merge multi_dp_res to one and save
                extracted_multi_dp_prob_list.append(mid_extracted_multi_dp_prob_list)

            token.add_tags_proba_dist(processed_ensb_pre_key, extracted_multi_dp_prob_list) # prepare multi_dp_res as one list

        return ensemble_pre_key, ori_pre_key, processed_ensb_pre_key


    def cal_max_score_in_interv(self, token, interv_dict, ori_pre_key):
        token_cat_num = len(token.tags_proba_dist[ori_pre_key])
        token_info = token.tags_proba_dist[ori_pre_key]
        token_val_scr_dict = {}

        # token_merged_score_dict = {}

        for i in range(token_cat_num):
            token_val_scr_dict[token_info[i].value] = token_info[i].score

        # interv_dict = {
        #     'Location': ['B-Location', 'I-Location', 'E-Location', 'S-Location'],
        # }

        token_merged_score_dict = self.cal_max_score_in_interv_once(token_val_scr_dict, interv_dict)
        # interv_dict_key_list = list(interv_dict.keys())
        # for j in range(len(interv_dict_key_list)):
        #     cur_interv_score_list = []
        #     cur_key = interv_dict_key_list[j]
        #     for k in range(len(interv_dict[cur_key])):
        #         cur_interv_score_list.append(token_val_scr_dict[interv_dict[cur_key][k]])
        #     cur_interv_max_score = max(cur_interv_score_list)   # process to get the score for an interv
        #     token_merged_score_dict[cur_key] = cur_interv_max_score


        return token_merged_score_dict

    def cal_max_score_in_interv_once(self, token_val_scr_dict, interv_dict):
        token_merged_score_dict = {}
        interv_dict_key_list = list(interv_dict.keys())
        for j in range(len(interv_dict_key_list)):
            cur_interv_score_list = []
            cur_key = interv_dict_key_list[j]
            for k in range(len(interv_dict[cur_key])):
                cur_interv_score_list.append(token_val_scr_dict[interv_dict[cur_key][k]])
            cur_interv_max_score = max(cur_interv_score_list)   # process to get the score for an interv
            token_merged_score_dict[cur_key] = cur_interv_max_score
        return token_merged_score_dict

    def cal_max_score_in_interv_dp(self, token, interv_dict, ori_pre_key, processed_ensb_pre_key, test_dropout_num):
        token_cat_num = len(token.tags_proba_dist[processed_ensb_pre_key])
        ori_token_info = token.tags_proba_dist[ori_pre_key]

        token_dp_merged_score_dict = {}


        for k in range(test_dropout_num):
            token_info = token.tags_proba_dist[processed_ensb_pre_key]
            token_val_scr_dict = {}


            for i in range(token_cat_num):
                mid_list = token_info[i]
                token_val_scr_dict[ori_token_info[i].value] = mid_list[k]


            token_merged_score_dict = self.cal_max_score_in_interv_once(token_val_scr_dict, interv_dict)
            for dict_key in token_merged_score_dict.keys():
                if dict_key not in token_dp_merged_score_dict.keys():
                    token_dp_merged_score_dict[dict_key] = [token_merged_score_dict[dict_key]]
                else:
                    token_dp_merged_score_dict[dict_key].append(token_merged_score_dict[dict_key])
        # 期待 的 形式 token_dp_merged_score_dict 是 phrase-level 键值 加
        return token_dp_merged_score_dict


    def merge_beios_to_one_score(self, datapoint, predicted_span, interv_dict, ori_pre_key, processed_ensb_pre_key=None, test_dropout_num=1, cal_dp=False):
        # un_identifier = predicted_span.unlabeled_identifier  # 'Span[1:2]: "asian"'
        # lf_braket_idx = un_identifier.index('[')
        # rg_braket_idx = un_identifier.index(']')
        # comma_idx = un_identifier.index(':')
        try:
            un_identifier = predicted_span.unlabeled_identifier  # e.g: 'Span[1:2]: "asian"'
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            comma_idx = un_identifier.index(':')
        except:
            un_identifier = str(predicted_span)
            lf_braket_idx = un_identifier.index('[')
            rg_braket_idx = un_identifier.index(']')
            new_str = un_identifier[lf_braket_idx:rg_braket_idx]
            new_comma_idx = new_str.index(':')
            comma_idx = new_comma_idx + lf_braket_idx

        low_idx = int(un_identifier[lf_braket_idx+1:comma_idx])
        high_idx = int(un_identifier[comma_idx + 1:rg_braket_idx])
        token_merged_score_dict = None
        for i in range(low_idx, high_idx):
            if token_merged_score_dict is None:
                token_merged_score_dict = self.cal_max_score_in_interv(datapoint.tokens[i], interv_dict, ori_pre_key)
                for dict_key in interv_dict.keys():
                    token_merged_score_dict[dict_key] = [token_merged_score_dict[dict_key]]
            else:
                next_token_score_dict = self.cal_max_score_in_interv(datapoint.tokens[i], interv_dict, ori_pre_key)
                for dict_key in interv_dict.keys():
                    token_merged_score_dict[dict_key].append(next_token_score_dict[dict_key])

        for key in token_merged_score_dict.keys():
            try:
                if self.use_var_metric==True:
                    # print(f'use_var_metric={self.use_var_metric}')
                    # token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key])) / (1 + np.std(np.array(token_merged_score_dict[key])))
                    token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key])) / ((1 + np.tanh(len(token_merged_score_dict[key]))) * 2)
                else:
                    token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key]))  # process to get the merged scores for multi tokens by mean
            except:
                token_merged_score_dict[key] = np.mean(np.array(token_merged_score_dict[key]))  # process to get the merged scores for multi tokens by mean

        if cal_dp == False:
            return token_merged_score_dict
        else:
            # additionaly cal the dp results
            token_merged_score_dict_dp = None
            for i in range(low_idx, high_idx):
                if token_merged_score_dict_dp is None:
                    token_merged_score_dict_dp = self.cal_max_score_in_interv_dp(datapoint.tokens[i], interv_dict, ori_pre_key, processed_ensb_pre_key, test_dropout_num)
                    for dict_key in interv_dict.keys():
                        token_merged_score_dict_dp[dict_key] = [token_merged_score_dict_dp[dict_key]]
                else:
                    next_token_score_dict = self.cal_max_score_in_interv_dp(datapoint.tokens[i], interv_dict, ori_pre_key, processed_ensb_pre_key, test_dropout_num)
                    for dict_key in interv_dict.keys():
                        token_merged_score_dict_dp[dict_key].append(next_token_score_dict[dict_key])

            for key in token_merged_score_dict_dp.keys():
                # need to revise
                token_merged_score_dict_dp[key] = np.mean(np.array(token_merged_score_dict_dp[key]), axis=0)  # process to get the merged scores for multi tokens by mean
            return token_merged_score_dict, token_merged_score_dict_dp
        # better to have a list to define the order of key labels, and uses the order to have a score vector.
        # datapoint.tokens[0].tags_proba_dist['predicted'][1].score