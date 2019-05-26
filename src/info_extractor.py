# TODO: accept docx as well
import re
import pandas as pd
import os
import sys

from collections import Counter, defaultdict
from datetime import datetime
from dateutil import relativedelta
from .utils import loadDocumentIntoSpacy, countWords, loadDefaultNLP
from typing import *

WORDS_LIST = {
    "Work": ["(Work|WORK)", "(Experience(s?)|EXPERIENCE(S?))", "(History|HISTORY)"],
    "Education": ["(Education|EDUCATION)", "(Qualifications|QUALIFICATIONS)"],
    "Skills": [
        "(Skills|SKILLS)",
        "(Proficiency|PROFICIENCY)",
        "LANGUAGE",
        "CERTIFICATION",
    ],
    "Projects": ["(Projects|PROJECTS)"],
    "Activities": ["(Leadership|LEADERSHIP)", "(Activities|ACTIVITIES)"],
}


class InfoExtractor:
    """
    Extracts key information from resumes
    """

    def __init__(self, spacy_nlp_model, parser):
        self.nlp = spacy_nlp_model
        self.parser = parser

    def extractFromFile(self, filename):
        doc, text = loadDocumentIntoSpacy(filename, self.parser, self.nlp)
        self.extractFromText(doc, text, filename)

    def extractFromText(self, doc, text, filename):
        name = InfoExtractor.findName(doc, filename)
        if name is None:
            name = ""
        email = InfoExtractor.findEmail(doc)
        if email is None:
            email = ""
        number = InfoExtractor.findNumber(doc)
        if number is None:
            number = ""
        city = InfoExtractor.findCity(doc)
        if city is None:
            city = ""
        categories = InfoExtractor.extractCategories(text)
        workAndEducation = InfoExtractor.findWorkAndEducation(
            categories, doc, text, name
        )
        totalWorkExperience = InfoExtractor.getTotalExperienceFormatted(
            workAndEducation["Work"]
        )
        totalEducationExperience = InfoExtractor.getTotalExperienceFormatted(
            workAndEducation["Education"]
        )
        allSkills = ", ".join(InfoExtractor.extractSkills(doc))
        print("Name: %s" % name)
        print("Email: %s" % email)
        print("Number: %s" % number)
        print("City/Country: %s" % city)
        print("\nWork Experience:")
        print(totalWorkExperience)
        for w in workAndEducation["Work"]:
            print(" - " + w)
        print("\nEducation:")
        print(totalEducationExperience)
        for e in workAndEducation["Education"]:
            print(" - " + e)
        print("\nSkills:")
        print(allSkills)

    @staticmethod
    def extractSkills(doc) -> List[str]:
        """
        Helper function to extract skills from spacy nlp text

        :param doc: object of `spacy.tokens.doc.Doc`
        :return: list of skills extracted
        """
        tokens = [token.text for token in doc if not token.is_stop]
        data = pd.read_csv(
            os.path.join(os.path.dirname(__file__), "constants/skills.csv")
        )
        skills = list(data.columns.values)
        skillset = []
        # check for one-grams
        for token in tokens:
            if token.lower() in skills:
                skillset.append(token)

        # check for bi-grams and tri-grams
        for token in doc.noun_chunks:
            token = token.text.lower().strip()
            if token in skills:
                skillset.append(token)
        return [i.capitalize() for i in set([i.lower() for i in skillset])]

    @staticmethod
    def extractCategories(text) -> Dict[str, List[Tuple[int, int]]]:
        """
        Helper function to extract categories like EDUCATION and EXPERIENCE from text
        :param text: text
        :return: Dict[str, List[Tuple[int, int]]]: {category: list((size_of_category, page_count))}
        """
        data = defaultdict(list)
        page_count = 0
        prev_count = 0
        prev_line = None
        prev_k = None
        for line in text.split("\n"):
            line = re.sub(r"\s+?", " ", line).strip()
            for (k, wl) in WORDS_LIST.items():
                # for each word in the list
                for w in wl:
                    # if category has not been found and not a very long line
                    # - long line likely not a category
                    if countWords(line) < 10:
                        match = re.findall(w, line)
                        if match:
                            size = page_count - prev_count
                            # append previous
                            if prev_k is not None:
                                data[prev_k].append((size, prev_count, prev_line))
                            prev_count = page_count
                            prev_k = k
                            prev_line = line
            page_count += 1

        # last item
        if prev_k is not None:
            size = page_count - prev_count - 1 # -1 cuz page_count += 1 on prev line
            data[prev_k].append((size, prev_count, prev_line))

        # choose the biggest category (reduce false positives)
        for k in data:
            if len(data[k]) >= 2:
                data[k] = [max(data[k], key=lambda x: x[0])]
        return data

    @staticmethod
    def findWorkAndEducation(categories, doc, text, name) -> Dict[str, List[str]]:
        inv_data = {v[0][1]: (v[0][0], k) for k, v in categories.items()}
        line_count = 0
        exp_list = defaultdict(list)
        name = name.lower()

        current_line = None
        is_dot = False
        is_space = True
        continuation_sent = []
        first_line = None
        unique_char_regex = "[^\sA-Za-z0-9\.\/\(\)\,\-\|]+"

        for line in text.split("\n"):
            line = re.sub(r"\s+", " ", line).strip()
            match = re.search(r"^.*:", line)
            if match:
                line = line[match.end() :].strip()

            # get first non-space line for filtering since
            # sometimes it might be a page header
            if line and first_line is None:
                first_line = line

            # update line_countfirst since there are `continue`s below
            line_count += 1
            if (line_count - 1) in inv_data:
                current_line = inv_data[line_count - 1][1]
            # contains a full-blown state-machine for filtering stuff
            elif current_line == "Work":
                if line:
                    # if name is inside, skip
                    if name == line:
                        continue
                    # if like first line of resume, skip
                    if line == first_line:
                        continue
                    # check if it's not a list with some unique character as list bullet
                    has_dot = re.findall(unique_char_regex, line[:5])
                    # if last paragraph is a list item
                    if is_dot:
                        # if this paragraph is not a list item and the previous line is a space
                        if not has_dot and is_space:
                            if line[0].isupper() or re.findall(r"^\d+\.", line[:5]):
                                exp_list[current_line].append(line)
                                is_dot = False

                    else:
                        if not has_dot and (
                            line[0].isupper() or re.findall(r"^\d+\.", line[:5])
                        ):
                            exp_list[current_line].append(line)
                            is_dot = False
                    if has_dot:
                        is_dot = True
                    is_space = False
                else:
                    is_space = True
            elif current_line == "Education":
                if line:
                    # if not like first line
                    if line == first_line:
                        continue
                    line = re.sub(unique_char_regex, '', line[:5]) + line[5:]
                    if len(line) < 12:
                        continuation_sent.append(line)
                    else:
                        if continuation_sent:
                            continuation_sent.append(line)
                            line = " ".join(continuation_sent)
                            continuation_sent = []
                        exp_list[current_line].append(line)

        return exp_list

    @staticmethod
    def findNumber(doc) -> Optional[str]:
        """
        Helper function to extract number from nlp doc
        :param doc: SpaCy Doc of text
        :return: int:number if found, else None
        """
        for sent in doc.sents:
            num = re.findall(r"\(?\+?\d+\)?\d+(?:[- \)]+\d+)*", sent.text)
            if num:
                for n in num:
                    if len(n) >= 8 and (
                        not re.findall(r"^[0-9]{2,4} *-+ *[0-9]{2,4}$", n)
                    ):
                        return n
        return None

    @staticmethod
    def findEmail(doc) -> Optional[str]:
        """
        Helper function to extract email from nlp doc
        :param doc: SpaCy Doc of text
        :return: str:email if found, else None
        """
        for token in doc:
            if token.like_email:
                return token.text
        return None

    @staticmethod
    def findCity(doc) -> Optional[str]:
        counter = Counter()
        """
        Helper function to extract most likely City/Country from nlp doc
        :param doc: SpaCy Doc of text
        :return: str:city/country if found, else None
        """
        for ent in doc.ents:
            if ent.label_ == "GPE":
                counter[ent.text] += 1

        if len(counter) >= 1:
            return counter.most_common(1)[0][0]
        return None

    @staticmethod
    def findName(doc, filename) -> Optional[str]:
        """
        Helper function to extract name from nlp doc
        :param doc: SpaCy Doc of text
        :param filename: used as backup if NE cannot be found
        :return: str:NAME_PATTERN if found, else None
        """
        to_chain = False
        all_names = []
        person_name = None

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                if not to_chain:
                    person_name = ent.text.strip()
                    to_chain = True
                else:
                    person_name = person_name + " " + ent.text.strip()
            elif ent.label_ != "PERSON":
                if to_chain:
                    all_names.append(person_name)
                    person_name = None
                    to_chain = False
        if all_names:
            return all_names[0]
        else:
            try:
                base_name_wo_ex = os.path.splitext(os.path.basename(filename))[0]
                return base_name_wo_ex + " (from filename)"
            except:
                return None

    @staticmethod
    def getNumberOfMonths(datepair) -> int:
        """
        Helper function to extract total months of experience from a resume

        :param date1: Starting date
        :param date2: Ending date
        :return: months of experience from date1 to date2
        """
        # if years
        # if years
        date2_parsed = False
        if datepair.get("fh", None) is not None:
            gap = datepair["fh"]
        else:
            gap = ""
        try:
            present_vocab = ("present", "date", "now")
            if "syear" in datepair:
                date1 = datepair["fyear"]
                date2 = datepair["syear"]

                if date2.lower() in present_vocab:
                    date2 = datetime.now()
                    date2_parsed = True

                try:
                    if not date2_parsed:
                        date2 = datetime.strptime(str(date2), "%Y")
                    date1 = datetime.strptime(str(date1), "%Y")
                except:
                    pass
            elif "smonth_num" in datepair:
                date1 = datepair["fmonth_num"]
                date2 = datepair["smonth_num"]

                if date2.lower() in present_vocab:
                    date2 = datetime.now()
                    date2_parsed = True

                for stype in ("%m" + gap + "%Y", "%m" + gap + "%y"):
                    try:
                        if not date2_parsed:
                            date2 = datetime.strptime(str(date2), stype)
                        date1 = datetime.strptime(str(date1), stype)
                        break
                    except:
                        pass
            else:
                date1 = datepair["fmonth"]
                date2 = datepair["smonth"]

                if date2.lower() in present_vocab:
                    date2 = datetime.now()
                    date2_parsed = True

                for stype in (
                    "%b" + gap + "%Y",
                    "%b" + gap + "%y",
                    "%B" + gap + "%Y",
                    "%B" + gap + "%y",
                ):
                    try:
                        if not date2_parsed:
                            date2 = datetime.strptime(str(date2), stype)
                        date1 = datetime.strptime(str(date1), stype)
                        break
                    except:
                        pass

            months_of_experience = relativedelta.relativedelta(date2, date1)
            months_of_experience = (
                months_of_experience.years * 12 + months_of_experience.months
            )
            return months_of_experience
        except Exception as e:
            return 0

    @staticmethod
    def getTotalExperience(experience_list) -> int:
        """
        Wrapper function to extract total months of experience from a resume

        :param experience_list: list of experience text extracted
        :return: total months of experience
        """
        exp_ = []
        for line in experience_list:
            line = line.lower().strip()
            # have to split search since regex OR does not capture on a first-come-first-serve basis
            experience = re.search(
                r"(?P<fyear>\d{4})\s*(\s|-|to)\s*(?P<syear>\d{4}|present|date|now)",
                line,
                re.I,
            )
            if experience:
                d = experience.groupdict()
                exp_.append(d)
                continue

            experience = re.search(
                r"(?P<fmonth>\w+(?P<fh>.)\d+)\s*(\s|-|to)\s*(?P<smonth>\w+(?P<sh>.)\d+|present|date|now)",
                line,
                re.I,
            )
            if experience:
                d = experience.groupdict()
                exp_.append(d)
                continue

            experience = re.search(
                r"(?P<fmonth_num>\d+(?P<fh>.)\d+)\s*(\s|-|to)\s*(?P<smonth_num>\d+(?P<sh>.)\d+|present|date|now)",
                line,
                re.I,
            )
            if experience:
                d = experience.groupdict()
                exp_.append(d)
                continue
        experience_num_list = [InfoExtractor.getNumberOfMonths(i) for i in exp_]
        total_experience_in_months = sum(experience_num_list)
        return total_experience_in_months

    @staticmethod
    def getTotalExperienceFormatted(exp_list) -> str:
        months = InfoExtractor.getTotalExperience(exp_list)
        if months < 12:
            return str(months) + " months"
        years = months // 12
        months = months % 12
        return str(years) + " years " + str(months) + " months"


if __name__ == "__main__":
    filename = sys.argv[1]
    from tika import parser

    nlp = loadDefaultNLP(is_big=False)
    infoExtractor = InfoExtractor(nlp, parser)
    infoExtractor.extractFromFile(filename)
