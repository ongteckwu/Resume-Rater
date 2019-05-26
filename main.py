import argparse
import random

from src.model import RatingModel
from src.utils import *
from src.info_extractor import InfoExtractor

RANDOM_INT_STR = str(random.randint(100000, 1000000))
parser = argparse.ArgumentParser(description="Train or test documents on models")

### GENERAL
parser.add_argument(
    "--type",
    dest="type",
    type=str,
    default="fixed",
    nargs="?",
    help="the type of model to use",
)
### FOR TEST
parser.add_argument(
    "path_to_resume",
    metavar="path_to_resume",
    nargs="?",
    default=None,
    help="the path to a resume",
)
parser.add_argument(
    "--model_path",
    dest="model_path",
    type=str,
    default=None,  # since there are two
    nargs="?",
    help="the pre-trained model to use",
)
parser.add_argument(
    "--no_info",
    dest="no_info",
    action="store_true",
    help="don't show extracted info and don't open document"
)
### FOR TRAIN
parser.add_argument(
    "--train",
    dest="train",
    type=str,
    default=None,
    nargs="?",
    help="the training directory for training",
)
parser.add_argument(
    "--model_name",
    dest="model_name",
    type=str,
    default="model_" + RANDOM_INT_STR,
    nargs="?",
    help="the model name to save as",
)

parser.add_argument(
    "--keywords",
    dest="keywords",
    type=str,
    nargs="*",
    help="keywords to use for training a fixed model",
)


args = vars(parser.parse_args())


class MainArgParseException(Exception):
    pass


# Testing--
_type = args["type"]
if _type not in ("fixed", "lda"):
    raise MainArgParseException("--type should be either 'fixed' or 'lda'")

if args["train"] is None:
    path_to_resume = args["path_to_resume"]
    if path_to_resume is None:
        raise MainArgParseException("No path/to/resume provided")

    train_type = args["type"]
    model_path = args["model_path"]
    # if no model path specified, use model name to derive it instead
    if model_path is None:
        model_name = args["model_name"]
        dirname = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            dirname, "src/models/model_" + _type, model_name + ".json"
        )

    r = RatingModel(_type, model_path)
    if args["no_info"]:
        infoExtractor = None
    else:
        infoExtractor = InfoExtractor(r.nlp, r.parser)
    r.test(path_to_resume, infoExtractor)
else:
    training_dir = args["train"]
    model_name = args["model_name"]
    keywords = args["keywords"]
    r = RatingModel()
    if _type == "fixed":
        if len(keywords) < 1:
            raise MainArgParseException("No keywords supplied in --keywords")
        r.train(training_dir, _type, model_name, keywords)
    else:
        r.train(training_dir, _type, model_name)
