import tempfile
import os
import re
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen, Request
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import logit, expit

# camelot is used to scrape the ANEW pdf tables, so if you're going to use that, be sure to install it
# pip install camelot-py opencv-python-headless ghostscript

SENTCOLS = ["valence", "arousal", "dominance"]
NEWCOLS = ["valence", "arousal", "confidence"]
STDCOLS = [f"{c}_std" for c in SENTCOLS]


# Functions to pull raw human ratings. We do not do any data manipulation here,
# except normalize some column names
def get_anew_df(url="https://e-lub.net/media/anew.pdf"):
    """
    Bradley and Lang (1999). Affective Norms for English Words (ANEW): Instruction

    Bradley, M.M., & Lang, P.J. (1999). Affective norms for English words (ANEW):
    Instruction Manual and Affective Ratings. Technical Report C-1, The Center for
    Research in Psychophysiology, University of Florida.
    """
    import camelot

    pages = [5, 18]
    tables = camelot.read_pdf(url, pages=f"{pages[0]}-{pages[1]}", flavor="stream")
    cols = ["word", "wordnum"] + SENTCOLS + ["freq"]
    names = [f"{c}{i}" for i in range(2) for c in cols]
    name_map = {f"{c}{i}": c for i in range(2) for c in cols}

    dfs = []
    with tempfile.TemporaryDirectory() as dirname:
        tables.export(os.path.join(dirname, "tmp"))
        for n in range(pages[0], pages[1] + 1):
            fn = os.path.join(dirname, f"tmp-page-{n}-table-1")
            tmpdf = pd.read_csv(fn, header=3, names=names)
            # there are two tables side-by-side that we want to separate
            dfs.append(tmpdf.iloc[:, :6].rename(columns=name_map))
            dfs.append(tmpdf.iloc[:, 6:].rename(columns=name_map))

    tmpdf = pd.concat(dfs)
    tmpdf.dropna(inplace=True)

    # separate mean and std
    # for c in cols:
    #    df[c] =
    res = []
    for row in tmpdf.itertuples():
        val = {"word": row.word}
        for c in HUMCOLS:
            mn, sd = row._asdict()[c].split("  ")
            val.update({c: float(mn.strip()), f"{c}_std": float(sd.strip(" ()"))})
        res.append(val)
    return pd.DataFrame(res)


def get_nrc(url="https://saifmohammad.com/WebDocs/Lexicons/NRC-VAD-Lexicon.zip"):
    """
    Publicly Released: 10 July 2011
    Created By: Dr. Saif M. Mohammad, Dr. Peter Turney
    Home Page: http://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm

    Crowdsourcing a Word-Emotion Association Lexicon, Saif Mohammad and Peter Turney,
    Computational Intelligence, 29 (3), 436-465, 2013.

    Emotions Evoked by Common Words and Phrases: Using Mechanical Turk to Create an
    Emotion Lexicon, Saif Mohammad and Peter Turney, In Proceedings of the NAACL-HLT
    2010 Workshop on Computational Approaches to Analysis and Generation of Emotion
    in Text, June 2010, LA, California.
    """
    req = Request(url)
    req.add_header(
        "user-agent",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
    )
    resp = urlopen(req)
    myzip = ZipFile(BytesIO(resp.read()))
    with myzip.open("NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt") as fp:
        # set keep_default_na=False so words like "null" are not interpreted as NaNs
        df = pd.read_csv(
            fp,
            sep="\t",
            keep_default_na=False,
            low_memory=False,
            names=["word", "valence", "arousal", "dominance"],
        )
    return df


def get_warriner(url="http://crr.ugent.be/papers/Ratings_Warriner_et_al.csv"):
    """
    Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal,
    and dominance for 13,915 English lemmas. Behavior Research Methods, 45, 1191-1207.

    Alternate source: https://github.com/JULIELab/XANEW
    """
    cols = ["word"] + SENTCOLS
    # We're only renaming the columns that we might use
    cmap = {
        "Word": "word",
        "V.Mean.Sum": "valence",
        "V.SD.Sum": "valence_std",
        "A.Mean.Sum": "arousal",
        "A.SD.Sum": "arousal_std",
        "D.Mean.Sum": "dominance",
        "D.SD.Sum": "dominance_std",
    }
    resp = urlopen(url)
    # set keep_default_na=False so words like "null" are not interpreted as NaNs
    df = pd.read_csv(resp, index_col=0, keep_default_na=False, low_memory=False).rename(
        columns=cmap
    )
    return df


def get_wan_from_sources(outfn="./human_wan.csv"):
    cols = ["word"] + SENTCOLS

    anew = get_anew_df()
    nrc = get_nrc()
    warr = get_warriner()

    warr = warr.loc[:, cols + STDCOLS]
    # Warriner et. al. used the standard 1-9 scale
    warr.loc[:, SENTCOLS] = (warr.loc[:, SENTCOLS] - 1) / 4 - 1
    warr.loc[:, STDCOLS] = warr.loc[:, STDCOLS] / 4
    warr["source"] = "warriner"

    # NRC data are scaled 0-1:
    # "...the scores range from 0 (lowest V/A/D) to 1 (highest V/A/D)."
    nrc.loc[:, SENTCOLS] = nrc.loc[:, SENTCOLS] * 2 - 1
    nrc["source"] = "nrc"

    # Bradley & Lang used the standard 1-9 scale
    anew.loc[:, SENTCOLS] = (anew.loc[:, SENTCOLS] - 1) / 4 - 1
    anew.loc[:, STDCOLS] = anew.loc[:, STDCOLS] / 4
    anew["source"] = "anew"

    df = pd.concat((warr, nrc, anew)).reset_index(drop=True)
    df.to_csv(outfn, index=None)


def load_wan_ratings(file_path=None):
    """
    Load the Warriner, ANEW and NRC ("wan") rating data
    """
    if not file_path:
        file_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data", "human_wan.csv"
        )
    df = pd.read_csv(file_path, keep_default_na=False)
    for c in SENTCOLS + STDCOLS:
        df.loc[:, c] = pd.to_numeric(df.loc[:, c])
    return df


def get_reliable_words(df=None, std_quantile=0.5, max_dist=0.5):
    if not df:
        df = load_wan_ratings()
    # for the big test set, we drop anew as it's very small
    bigdf = df.loc[df.source != "anew"].dropna(subset=SENTCOLS)
    whdf = bigdf.pivot(index="word", columns="source")
    whdf.columns = ["_".join(c) for c in whdf.columns]

    # drop cases where we don't have overlap between nrc and warriner
    whdf.dropna(subset=["valence_nrc", "valence_warriner"], inplace=True)
    # Drop the empty NRD std cols
    whdf.dropna(axis=1, inplace=True)

    nrc_cols = [f"{c}_nrc" for c in SENTCOLS]
    war_cols = [f"{c}_warriner" for c in SENTCOLS]
    whdf["dist"] = np.linalg.norm(
        whdf.loc[:, war_cols].values - whdf.loc[:, nrc_cols].values, axis=1
    )
    std_cols = [f"{s}_std_warriner" for s in SENTCOLS]
    # dist_thresh = whdf.dist.quantile(max_dist)
    scoredf = pd.DataFrame(index=whdf.index)
    scoredf["dist"] = whdf.dist <= max_dist
    if not isinstance(std_quantile, float):
        std_thresh = whdf.loc[:, std_cols].quantile(std_quantile).iloc[0]
        for c in std_cols:
            scoredf[c] = whdf.loc[:, c] <= std_thresh[c]
    else:
        whdf["std_dist"] = np.linalg.norm(whdf.loc[:, std_cols].values, axis=1)
        std_thresh = whdf.loc[:, "std_dist"].quantile(std_quantile)
        scoredf["std_dist"] = whdf.loc[:, "std_dist"] <= std_thresh

    idx = scoredf.all(axis=1)
    scoredf = scoredf.loc[~idx]
    extras = {"scoredf": scoredf, "std_thresh": std_thresh}
    return whdf.loc[idx].copy(), extras


def sentiment_from_scores(model, scores):
    def _sent(scores):
        valence, arousal, confidence = 0, 0, 0
        total_weight = 0
        for k, v in model.items():
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
        return sent

    if not isinstance(scores, list):
        sent = _sent(scores)
    else:
        sent = [_sent(s) for s in scores]
    return sent


def sentiment_from_logits(model, utterances, logit_df=None):
    """
    Compute sentiment scores given pre-computed logits, a model, and a list of utterances
    """
    if not logit_df:
        logit_df = pd.read_csv("data/logits.csv", keep_default_na=False)
    anchors = [x for lst in model.values() for x in lst]
    us, ls = set(utterances).intersection(), set(logit_df.utterance)
    missing = us - ls
    # Doing this with set intersection breaks the order, so just loop it
    utt_clean = [u for u in utterances if u not in missing]
    if len(missing) > 0:
        print(f"{missing} not in logits (n={len(missing)}).")
    el = logit_df.loc[logit_df.utterance.isin(utt_clean), anchors]
    # the same anchor word may appear in more than one anchor point, so we drop the duplicated columns
    scores = (
        (np.exp(el) / np.exp(el).sum(axis=0))
        .loc[:, ~el.columns.duplicated()]
        .to_dict(orient="records")
    )
    sentdf = pd.DataFrame(sentiment_from_scores(model, scores))
    sentdf["utterance"] = utt_clean
    sentdf.set_index("utterance", inplace=True)
    return sentdf, anchors


def get_pvalues_tscore(r, n):
    """
    Given a dataframe of Pearson product-moment correlation coefficients
    (e.g., as from df.corr()) and the number of samples used to compute r,
    retuns a dataframe of p-values by using r and n to estimatte t-scores:
      https://stats.stackexchange.com/questions/320510/t-test-for-pearson-correlation-coeffcient
    """
    # convert r to t
    tval = r / (np.sqrt((1 - r**2) / (n - 2)))
    # look up p value for t
    p = 2 * stats.t.sf(abs(tval), n - 2)
    return pd.DataFrame(index=r.index, columns=r.columns, data=p)


def get_pvalues(r, n):
    """
    Given a dataframe of Pearson product-moment correlation coefficients
    (e.g., as from df.corr()) and the number of samples used to compute r,
    retuns a dataframe of p-values based on scipy.stats.pearsonr. See:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Note that if the dataframe used to compute r contains nans then the number
    of samples may vary from cell to cell. In this case, you should pass a dataframe
    or array of n that is the same size as r and contains the actual n for each cell.
    If r is square (e.g., as from df.corr), n can be a vector with length identical
    to the number of columns in r. E.g.,
    df = pd.DataFrame()
    r = df.corr()
    notnan = (~df.isna()).astype(int)
    n = np.dot(notnan.T, notnan)
    p = get_pvalues(r, n)
    """
    p = r.copy()
    dist = stats.beta(n / 2 - 1, n / 2 - 1, loc=-1, scale=2)
    p.loc[:, :] = 2 * dist.cdf(-r.abs())
    return p


def df_corr(df):
    """
    Simple wrapper around panda's DataFrame.corr funtion that also computes p-values.

    returns the pearson correlation in r, the 2-tailed p-value in p, and the
    number of samples (i.e., not nan) for each pairwise comparison in n.
    """
    r = df.corr()
    notnan = (~df.isna()).astype(int)
    n = np.dot(notnan.T, notnan)
    p = get_pvalues(r, n)
    return r, p, pd.DataFrame(index=r.index, columns=r.columns, data=n)


def get_corr(df, xcol, ycol):
    tmpdf = df.copy().dropna()
    pr, pp = stats.pearsonr(tmpdf[xcol], tmpdf[ycol])
    sr, sp = stats.spearmanr(tmpdf[xcol], tmpdf[ycol])
    kr, kp = stats.kendalltau(tmpdf[xcol], tmpdf[ycol])
    n = tmpdf.shape[0]
    return {
        "r": pr,
        "p": pp,
        "n": n,
        "spearmanr": sr,
        "spearmanp": sp,
        "kendallr": kr,
        "kendallp": kp,
    }


def get_stats(df, cols=["NRC", "Warr"]):
    idx, r, p = [], [], []
    full_stats = {}
    for s in NEWCOLS:
        s = s.capitalize()
        vac = f"{s} VAC"
        idx.append(vac)
        for ds in cols:
            hr = f"{s} {ds}"
            # r = corr.loc[vac, hr]
            corr = get_corr(df, vac, hr)
            full_stats[(vac, hr)] = corr
            r.append(corr["r"])
            p.append(corr["p"])
    n = corr["n"]
    rdf = pd.DataFrame(
        index=idx, columns=cols, data=np.array(r).reshape(len(NEWCOLS), len(cols))
    )
    pdf = pd.DataFrame(
        index=idx, columns=cols, data=np.array(p).reshape(len(NEWCOLS), len(cols))
    )
    return rdf, pdf, n, full_stats


def separate_utterances(df):
    END_OF_UTTERANCE_PUNCTUATION = ".!?"
    regexp = f"([^{END_OF_UTTERANCE_PUNCTUATION}]+[{END_OF_UTTERANCE_PUNCTUATION}])"

    utterances = []
    for row in df.itertuples():
        speaker = row.speaker
        talk_turn = re.sub(
            r"\([^)]*\)", "", row.talk_turn
        )  # Remove everything between parentheses
        turn_label = row.turn_label
        turn_number = row.turn_number

        sentences = re.findall(regexp, talk_turn)

        last_sentence = sentences[0]
        utterances.append((turn_label, last_sentence.strip(), speaker, turn_number))

        for sentence in sentences[1:]:
            if last_sentence.endswith("Dr."):
                last_sentence += " " + sentence.strip()
                utterances[-1] = (
                    turn_label,
                    last_sentence.strip(),
                    speaker,
                    turn_number,
                )
            else:
                utterances.append((turn_label, sentence.strip(), speaker, turn_number))
                last_sentence = sentence.strip()

    return pd.DataFrame(
        utterances, columns=["turn_label", "utterance", "speaker", "turn_number"]
    )


def patch_model(nas):
    # Monkey-patch postprocess to return entail_logits
    from types import MethodType

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
            entail_contr_logits = reshaped_outputs[
                ..., [contradiction_id, entailment_id]
            ]
            scores = np.exp(entail_contr_logits) / np.exp(entail_contr_logits).sum(
                -1, keepdims=True
            )
            scores = scores[..., 1]
        else:
            # softmax the "entailment" logits over all candidate labels
            entail_logits = reshaped_outputs[..., self.entailment_id]
            scores = np.exp(entail_logits) / np.exp(entail_logits).sum(
                -1, keepdims=True
            )

        top_inds = list(reversed(scores[0].argsort()))
        return {
            "sequence": sequences[0],
            "labels": [candidate_labels[i] for i in top_inds],
            "scores": scores[0, top_inds].tolist(),
            "entail_logits": reshaped_outputs[..., self.entailment_id][0, top_inds],
        }

    nas.classifier.postprocess = MethodType(postprocess, nas.classifier)
