import pickle

import torch


def clean_protocol(protocol):
    protocol = protocol.lower()
    protocol_split = protocol.split("\n")
    filter_out_empty_fn = lambda x: len(x.strip()) > 0
    strip_fn = lambda x: x.strip()
    protocol_split = list(filter(filter_out_empty_fn, protocol_split))
    protocol_split = list(map(strip_fn, protocol_split))
    return protocol_split


def split_protocol(protocol):
    protocol_split = clean_protocol(protocol)
    inclusion_idx, exclusion_idx = len(protocol_split), len(protocol_split)
    for idx, sentence in enumerate(protocol_split):
        if "inclusion" in sentence:
            inclusion_idx = idx
            break
    for idx, sentence in enumerate(protocol_split):
        if "exclusion" in sentence:
            exclusion_idx = idx
            break
    if inclusion_idx + 1 < exclusion_idx + 1 < len(protocol_split):
        inclusion_criteria = protocol_split[inclusion_idx:exclusion_idx]
        exclusion_criteria = protocol_split[exclusion_idx:]
        if not (len(inclusion_criteria) > 0 and len(exclusion_criteria) > 0):
            print(len(inclusion_criteria), len(exclusion_criteria), len(protocol_split))
            exit()
        return inclusion_criteria, exclusion_criteria  ## list, list
    else:
        return (protocol_split,)


def load_sentence_2_vec(data_path="data/sentence2embedding.pkl"):
    sentence_2_vec = pickle.load(open(data_path, "rb"))
    return sentence_2_vec


def protocol2feature(protocol, sentence_2_vec):
    result = split_protocol(protocol)
    inclusion_criteria, exclusion_criteria = result[0], result[-1]
    inclusion_feature = [
        sentence_2_vec[sentence].view(1, -1)
        for sentence in inclusion_criteria
        if sentence in sentence_2_vec
    ]
    exclusion_feature = [
        sentence_2_vec[sentence].view(1, -1)
        for sentence in exclusion_criteria
        if sentence in sentence_2_vec
    ]
    if inclusion_feature == []:
        inclusion_feature = torch.zeros(1, 768)
    else:
        inclusion_feature = torch.cat(inclusion_feature, 0)
    if exclusion_feature == []:
        exclusion_feature = torch.zeros(1, 768)
    else:
        exclusion_feature = torch.cat(exclusion_feature, 0)
    return inclusion_feature, exclusion_feature
