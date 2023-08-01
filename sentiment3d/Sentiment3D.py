import json
import warnings
from types import MethodType

import numpy as np
import torch
from codenamize import codenamize
from transformers import pipeline


class Sentiment3D:
    """
    Sentiment3D is a callable class that takes a string or a list of strings and returns valence,
    arousal, and confidence (VAC) values for the string, or a list of VAC tuples given a list of
    strings. The VAC values can be interpreted as the positivity or negativity of the input in the
    usual sense of 'sentiment' (valence), the level of excitedness/calmness (arousal), and the
    degree of assuredness/confidence or lack thereof (confidence)

    Note:
        Sentiment3D uses the zero-shot classifier facebook/bart-large-mnli from hugginface.co.
    """

    def __init__(self, anchor_spec="anchor_spec.json", model_dir=None, device=None, batch_size=20):
        """Instantiate class instance

        :param anchor_spec: A dict specifying the anchor point classes or a string to a json file
               with the anchor point classes that will loaded to a dict. This dict should have nine
               keys of the form '+++', '++-', '+-+', '+--', '-++', '-+-', '--+', '---', and "000",
               where "+++" is the +1, +1, +1 corner, "---" is the -1, -1, -1 corner, etc. The
               value at each key is a list of words (more generally, classes) to use in the
               classifier to get weights for the corner associated with the given key.
        :param model_dir: local directory for the BART model cache. Set to None to not cache the model.
        :param device: Device used to run the model. The default of None will auto-detect a GPU and
                       use it, falling back to CPU if no GPU is found. To force CPU use, set to -1.
        """
        if isinstance(anchor_spec, dict):
            # save a copy to ensure that it isn't mutated if caller mutates anchor_spec
            self.anchor_spec = anchor_spec.copy()
        else:
            with open(anchor_spec) as fp:
                self.anchor_spec = json.load(fp)

        self.model_dir = model_dir
        self.classes = list(set(sum([self.anchor_spec[k] for k in self.anchor_spec.keys()], [])))
        self.model_str = json.dumps(dict(anchor_spec=self.anchor_spec, model_type="NineAnchorSentiment"))
        self.model_name = codenamize(self.model_str)
        self.batch_size = batch_size
        print(f"model name: {self.model_name}")
        if not device:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = -1

        self.device = device
        if model_dir:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model=model_dir,
                    tokenizer=model_dir,
                    device=device,
                )
            except:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=device,
                )
                self.classifier.save_pretrained(model_dir)
        else:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=device,
            )

        # add postprocess to return entail_logits
        def postprocess(self, model_outputs, multi_label=False):
            candidate_labels = [outputs["candidate_label"] for outputs in model_outputs]
            sequences = [outputs["sequence"] for outputs in model_outputs]
            logits = np.concatenate([output["logits"].numpy() for output in model_outputs])
            N = logits.shape[0]
            n = len(candidate_labels)
            num_sequences = N // n
            reshaped_outputs = logits.reshape((num_sequences, n, -1))

            if multi_label or len(candidate_labels) == 1:
                # softmax over the entailment vs. contradiction dim for each label independently
                entailment_id = self.entailment_id
                contradiction_id = -1 if entailment_id == 0 else 0
                entail_contr_logits = reshaped_outputs[..., [contradiction_id, entailment_id]]
                scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(-1, keepdims=True)
                scores = scores[..., 1]
            else:
                # softmax the "entailment" logits over all candidate labels
                entail_logits = reshaped_outputs[..., self.entailment_id]
                scores = np.exp(entail_logits) / np.exp(entail_logits).sum(-1, keepdims=True)

            top_inds = list(reversed(scores[0].argsort()))
            return {
                "sequence": sequences[0],
                "labels": [candidate_labels[i] for i in top_inds],
                "scores": scores[0, top_inds].tolist(),
                "entail_logits": reshaped_outputs[..., self.entailment_id][0, top_inds],
            }

        self.classifier.postprocess = MethodType(postprocess, self.classifier)

    def get_utterance_class_scores(self, utterance):
        """
        Utterance can be a string or a list of strings. Passing a list of strings
        should be much faster than looping when doing many utterances.
        """
        # torch may flood you with PipelineChunkIterator warnings that seem to be harmless.
        # https://github.com/huggingface/transformers/issues/23003
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cl_out = self.classifier(utterance, self.classes, batch_size=self.batch_size)

        if isinstance(cl_out, list):
            scores = [dict(zip(c["labels"], c["scores"])) for c in cl_out]
        else:
            scores = dict(zip(cl_out["labels"], cl_out["scores"]))
        return scores

    def get_utterance_sentiment(self, utterance, scores=None, return_scores=False):
        """
        Utterance can be a string or a list of strings. Passing a list of strings
        should be much faster than looping when doing many utterances.

        :return sent: The estimated sentement values (valence, arousal, confidence).

        """

        def _sent(scores, return_scores):
            valence, arousal, confidence = 0, 0, 0
            total_weight = 0
            for k, v in self.anchor_spec.items():
                if k != "000":
                    if "1" in k:
                        signs = [-1 if t else 1 for t in k.split("1")]
                    else:
                        signs = [-1 if t == "-" else 1 for t in k]
                    k_weight = sum([scores[c] for c in v])
                    valence += signs[0] * k_weight
                    arousal += signs[1] * k_weight
                    confidence += signs[2] * k_weight
                    total_weight += k_weight
            sent = {
                "valence": valence / total_weight,
                "arousal": arousal / total_weight,
                "confidence": confidence / total_weight,
            }
            if return_scores:
                sent["scores"] = scores
            return sent

        scores = scores if scores else self.get_utterance_class_scores(utterance)
        if not isinstance(scores, list):
            sent = _sent(scores, return_scores=return_scores)
        else:
            sent = [_sent(s, return_scores=return_scores) for s in scores]
        return sent

    def __call__(self, utterance):
        return self.get_utterance_sentiment(utterance)
