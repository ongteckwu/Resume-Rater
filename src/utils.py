import os
import pandas as pd
import re
import spacy
import docx

from typing import *
from .custom_filter import customFilter


def loadDefaultNLP(is_big: bool = True) -> Any:
    """
    Function to load the default SpaCy nlp model into self.nlp
    :param is_big: if True, uses a large vocab set, else a small one
    :returns: nlp: a SpaCy nlp model
    """

    def segment_on_newline(doc):
        for token in doc[:-1]:
            if token.text.endswith("\n"):
                doc[token.i + 1].is_sent_start = True
        return doc

    if is_big:
        nlp = spacy.load("en_core_web_lg")
    else:
        nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(segment_on_newline, before="parser")
    return nlp


def countWords(line: str) -> int:
    """
    Counts the numbers of words in a line
    :param line: line to count
    :return count: num of lines
    """
    count = 0
    is_space = False
    for c in line:
        is_not_char = not c.isspace()
        if is_space and is_not_char:
            count += 1
        is_space = not is_not_char
    return count


def getAllTokensAndChunks(doc) -> Tuple[Set[Any], Set[Any]]:
    """
    Converts a spacy doc into tokens and chunks. Tokens and chunks pass through a customFilter first
    :param doc: a SpaCy doc
    :returns: seen_chunks_words: set of strings seen
    :returns: all_tokens_chunks: set of all tokens and chunks found
    """
    # used to test duplicate words/chunks
    seen_chunks_words = set()
    # collate all words/chunks
    all_tokens_chunks = set()
    # generate all 1-gram tokens
    for token in doc:
        w = token.lemma_.lower()
        if (w not in seen_chunks_words) and customFilter(token):
            all_tokens_chunks.add(token)
            seen_chunks_words.add(w)

    # generate all n-gram tokens
    for chunk in doc.noun_chunks:
        c = chunk.lemma_.lower()
        if (
            len(chunk) > 1
            and (c not in seen_chunks_words)
            and all(customFilter(token) for token in chunk)
        ):
            all_tokens_chunks.add(chunk)
            seen_chunks_words.add(c)

    return seen_chunks_words, all_tokens_chunks


def findDocumentsRecursive(base_dir: str) -> Optional[List[str]]:
    """
    Recursively get all documents from `base_dir`
    :param base_dir: base directory of documents
    :returns out: a list of full file names of the documents
    """
    out: List[str] = []

    # check if base_dir is a proper dir
    if not os.path.isdir(base_dir):
        return None

    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path):
            out.extend(findDocumentsRecursive(full_path))
        else:
            for end in (".pdf", ".docx"):
                if full_path.endswith(end):
                    out.append(full_path)
    return out


def generateDFFromData(
    data: Dict[Any, Any],
    filename: str,
    save_csv: bool = False
) -> pd.DataFrame:
    """
    Generates DF for model creation
    :param data: dictionary of data
    :param filename: what to save model as
    :param save_csv: whether to save the model as csv
    :returns data_df: the model df
    """
    data_df = pd.DataFrame(data=data)
    data_df.sort_values(by=["score"], ascending=False, inplace=True)
    data_df.reset_index(inplace=True)
    if save_csv:
        data_df.to_csv(filename)
    return data_df


def getDocxText(filename: str) -> str:
    """
    Get the text from a docx file
    :param filename: docx file
    :returns fullText: text of file
    """
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        txt = para.text
        fullText.append(txt)
    return "\n".join(fullText)


def getPDFText(filename: str, parser) -> str:
    """
    Get the text from a pdf file
    :param filename: pdf file
    :param parser: pdf parser
    :returns fullText: text of file
    """
    raw = parser.from_file(filename)
    new_text = raw["content"]
    if "title" in raw["metadata"]:
        title = raw["metadata"]["title"]
        new_text = new_text.replace(title, "")
    return new_text


def loadDocumentIntoSpacy(f: str, parser, spacy_nlp) -> Optional[Tuple[Any, str]]:
    """
    Convert file into spacy Document
    :param f: filename
    :param parser: pdf_parser
    :param spacy_nlp: nlp model
    :returns nlp_doc: nlp doc
    :returns new_text: text of file
    """
    if f.endswith(".pdf"):
        new_text = getPDFText(f, parser)
    elif f.endswith(".docx"):
        new_text = getDocxText(f)
    else:
        return None, None

    # new_text = "\n".join(
    #     [line.strip() for line in new_text.split("\n") if len(line) > 1]
    # )
    new_text = re.sub("\n{3,}", "\n", new_text)
    new_text = str(bytes(new_text, "utf-8").replace(b"\xe2\x80\x93", b""), "utf-8")
    # convert to spacy doc
    return spacy_nlp(new_text), new_text
