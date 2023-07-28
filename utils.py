import csv
import os
import re
import tempfile
from io import BytesIO
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy import stats

from sentiment3d import Sentiment3D

# camelot is used to scrape the ANEW pdf tables, so if you're going to use that, be sure to install it
# pip install camelot-py opencv-python-headless ghostscript

SENTCOLS = ["valence", "arousal", "dominance"]
NEWCOLS = ["valence", "arousal", "confidence"]
STDCOLS = [f"{c}_std" for c in SENTCOLS]
COLMAP = {
    "valence_nrc": "Valence NRC",
    "arousal_nrc": "Arousal NRC",
    "confidence_nrc": "Confidence NRC",
    "valence_warriner": "Valence Warr",
    "arousal_warriner": "Arousal Warr",
    "confidence_warriner": "Confidence Warr",
    "valence_anew": "Valence ANEW",
    "arousal_anew": "Arousal ANEW",
    "confidence_anew": "Confidence ANEW",
    "valence": "Valence VAC",
    "arousal": "Arousal VAC",
    "confidence": "Confidence VAC",
}


def map_cols(df):
    df.columns = [c.replace("dominance", "confidence") for c in df.columns]
    cols = [c for c in COLMAP.keys() if c in df.columns]
    df = df[cols].rename(columns=COLMAP)
    return df


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
    res = []
    for row in tmpdf.itertuples():
        val = {"word": row.word}
        for c in SENTCOLS:
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
    df = pd.read_csv(resp, index_col=0, keep_default_na=False, low_memory=False).rename(columns=cmap)
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
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "human_wan.csv")
    # pandas will read the word "null" as a missing value unless we set keep_default_na=False.
    df = pd.read_csv(file_path, keep_default_na=False)
    for c in SENTCOLS + STDCOLS:
        df.loc[:, c] = pd.to_numeric(df.loc[:, c])
    return df


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
        file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "logits.csv")
        logit_df = pd.read_csv(file_path, keep_default_na=False)

    anchors = [x for lst in model.values() for x in lst]
    us, ls = set(utterances).intersection(), set(logit_df.utterance)
    missing = us - ls
    utt_clean = [u for u in utterances if u not in missing]
    if len(missing) > 0:
        print(f"{missing} not in logits (n={len(missing)}).")
    el = logit_df.loc[logit_df.utterance.isin(utt_clean), anchors]
    # the same anchor word may appear in more than one anchor point, so we drop the duplicated columns
    scores = (np.exp(el) / np.exp(el).sum(axis=0)).loc[:, ~el.columns.duplicated()].to_dict(orient="records")
    sentdf = pd.DataFrame(sentiment_from_scores(model, scores))
    sentdf["utterance"] = utt_clean
    sentdf.set_index("utterance", inplace=True)
    return sentdf, anchors


def get_pvalues_tscore(r, n):
    """
    Given a dataframe of Pearson product-moment correlation coefficients
    (e.g., as from df.corr()) and the number of samples used to compute r,
    returns a dataframe of p-values by using r and n to estimatte t-scores:
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
    returns a dataframe of p-values based on scipy.stats.pearsonr. See:
      https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html

    Note that if the dataframe used to compute r contains nans then the number
    of samples may vary from cell to cell. In this case, you should pass a dataframe
    or array of n that is the same size as r and contains the actual n for each cell.
    E.g.,
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
    Simple wrapper around panda's DataFrame.corr function that also computes p-values.

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
    rdf = pd.DataFrame(index=idx, columns=cols, data=np.array(r).reshape(len(NEWCOLS), len(cols)))
    pdf = pd.DataFrame(index=idx, columns=cols, data=np.array(p).reshape(len(NEWCOLS), len(cols)))
    return rdf, pdf, n, full_stats


def separate_utterances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse talk turns in a dataframe into individual utterances using punctuation.

    Args:
        df: pd.DataFrame
            The dataframe containing talk turns.

    Returns:
        pd.DataFrame: A new dataframe with separated utterances.

    """
    END_OF_UTTERANCE_PUNCTUATION = ".!?"
    regexp = f"([^{END_OF_UTTERANCE_PUNCTUATION}]+[{END_OF_UTTERANCE_PUNCTUATION}])"

    utterances = []
    for row in df.itertuples():
        speaker = row.speaker
        talk_turn = re.sub(r"\([^)]*\)", "", row.talk_turn)  # Remove everything between parentheses
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

    return pd.DataFrame(utterances, columns=["turn_label", "utterance", "speaker", "turn_number"])


def generate_logits(utterances: list, model: dict, batch_size: int = 50) -> pd.DataFrame:
    """
    Generates a dataframelogits for sentiment analysis of utterances in a dataframe.

    Args:
        utterances (list): A list of utterances for which to generate logits.
        model (dict): A dictionary containing sentiment anchor words for the model.
        batch_size: The size of the MNLI inference batches

    Returns:
        pd.DataFrame: A dataframe containing the logits for sentiment analysis.
    """

    class_words = [model[k] for k in model]
    candidate_anchor_words = [item for sublist in class_words for item in sublist]

    utterance_batches = [
        utterances[b * batch_size : (b + 1) * batch_size] for b in range(len(utterances) // batch_size + 1)
    ]

    nas = Sentiment3D(anchor_spec=model, model_dir="facebook-bart-large-mnli")

    for i, utterance_batch in enumerate(utterance_batches):
        batch_logits = []
        inferences = nas.classifier(utterance_batch, candidate_anchor_words)

        for inference in inferences:
            sequence_logits = list(zip(inference["labels"], inference["entail_logits"]))
            sequence = inference["sequence"]

            for sl in sequence_logits:
                batch_logits.append((sequence, *sl))

        with open("data/logits_batch-append.csv", "a") as out:
            csv_out = csv.writer(out)
            csv_out.writerows(batch_logits)

    logits_df = pd.read_csv("data/logits_batch-append.csv", header=None, keep_default_na=False)
    logits_df.columns = ["utterance", "anchor", "logit"]
    logits_df = logits_df.pivot_table(index="utterance", columns="anchor", values="logit")
    logits_df.to_csv("data/logits.csv")
    return logits_df
