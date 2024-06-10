from flair.data import Dictionary, Label, Sentence, Span
from typing import Dict, List, Optional, Tuple, Union
import torch
import flair.nn
from flair.datasets import DataLoader


def extract_labels(corpus, init_from_state_dict, tag_dictionary, tag_format, allow_unk_predictions):
    # refer to flair/trainers/trainer.py
    train_data = corpus.train
    batch_loader = DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,  # never shuffle the first epoch
        num_workers=0,
        sampler=None,
    )
    total_number_of_batches = len(batch_loader)
    print(f'the total num of train_data is {total_number_of_batches}')

    for batch_no, batch in enumerate(batch_loader):
        # if necessary, make batch_steps
        batch_steps = [batch]
        
        # forward and backward for batch
        for batch_step in batch_steps:
            sentences = batch_step
            if len(sentences) == 0:
                return torch.tensor(0.0, dtype=torch.float, device=flair.device, requires_grad=True), 0
            sentences = sorted(sentences, key=len, reverse=True)
            gold_labels = prepare_label_tensor(sentences, init_from_state_dict, tag_dictionary, tag_format, allow_unk_predictions)


