from collections import defaultdict
from typing import Dict, List, Tuple


def get_spans_from_bio(bioes_tags: List[str], bioes_scores=None) -> List[Tuple[List[int], float, str]]:
    # add a dummy "O" to close final prediction
    bioes_tags.append("O")
    # return complex list
    found_spans = []
    # internal variables
    current_tag_weights: Dict[str, float] = defaultdict(lambda: 0.0)
    previous_tag = "O-"
    current_span: List[int] = []
    current_span_scores: List[float] = []
    for idx, bioes_tag in enumerate(bioes_tags):

        # non-set tags are OUT tags
        if bioes_tag == "" or bioes_tag == "O" or bioes_tag == "_":
            bioes_tag = "O-"

        # anything that is not OUT is IN
        in_span = False if bioes_tag == "O-" else True

        # does this prediction start a new span?
        starts_new_span = False

        # begin and single tags start new spans
        if bioes_tag[0:2] in ["B-", "S-"]:
            starts_new_span = True

        # in IOB format, an I tag starts a span if it follows an O or is a different span
        if bioes_tag[0:2] == "I-" and previous_tag[2:] != bioes_tag[2:]:
            starts_new_span = True

        # single tags that change prediction start new spans
        if bioes_tag[0:2] in ["S-"] and previous_tag[2:] != bioes_tag[2:]:
            starts_new_span = True

        # if an existing span is ended (either by reaching O or starting a new span)
        if (starts_new_span or not in_span) and len(current_span) > 0:
            # determine score and value
            span_score = sum(current_span_scores) / len(current_span_scores)  # self comment: after one span is over, start cal mean of the span values
            span_value = max(current_tag_weights.keys(), key=current_tag_weights.__getitem__)

            # append to result list
            found_spans.append((current_span, span_score, span_value))

            # reset for-loop variables for new span
            current_span = []
            current_span_scores = []
            current_tag_weights = defaultdict(lambda: 0.0)

        if in_span:
            current_span.append(idx)
            current_span_scores.append(bioes_scores[idx] if bioes_scores else 1.0)
            weight = 1.1 if starts_new_span else 1.0
            current_tag_weights[bioes_tag[2:]] += weight

        # remember previous tag
        previous_tag = bioes_tag

    return found_spans
