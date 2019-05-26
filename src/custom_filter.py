def customFilter(token):
    customized_stop_words = [
        "work",
        "experience",
        "education",
        "mobile",
        "email",
        "number",
        "num",
        "professional",
        "career",
        "history",
        "histories",
        "skill",
        "skills",
        "activity",
        "activities",
        "curriculum",
        "tool",
        "tools",
        "language",
        "languages",
        "profile",
        "qualification",
        "qualifications",
        "certificate",
        "certificates",
        "certifications",
        "certification",
        "information",
        "intern",
        "volunteer",
        "award",
        "awards",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    token_lower = token.lemma_.lower()
    return all(
        [
            not token.is_space,  # not space
            not token.is_punct,  # not punct
            not token.is_bracket,  # not bracket
            not token.is_quote,  # not quote
            not token.is_currency,  # not currency
            not token.like_num,  # not number
            not token.like_url,  # not url
            not token.like_email,  # not email
            not token.is_oov,  # not out of vocab
            not token.is_stop,  # not a stopword
            token.is_alpha,  # is alphabetical
            token.has_vector,
            token_lower not in customized_stop_words,
            token.pos_ not in ["ADV", "ADJ", "INTJ", "PART", "PRON", "X"],
            token.ent_type_
            not in [
                "PERSON",
                "ORG",
                "DATE",
                "CARDINAL",
                "TIME",
            ],  # not amongst these categories
        ]
    )
