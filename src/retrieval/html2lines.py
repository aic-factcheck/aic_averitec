import sys
from time import sleep
import trafilatura
from trafilatura.meta import reset_caches
from trafilatura.settings import DEFAULT_CONFIG
import spacy


nlp = spacy.load("en_core_web_lg")


DEFAULT_CONFIG.MAX_FILE_SIZE = 50000
MIN_CHAR = 50
MAX_CHAR = 5000


def get_page(url):
    page = None
    for _ in range(3):
        try:
            page = trafilatura.fetch_url(url, config=DEFAULT_CONFIG)
            assert page is not None
            print("Fetched " + url, file=sys.stderr)
            break
        except:
            sleep(3)
    return page


def url2lines(url):
    page = get_page(url)

    if page is None:
        return []

    res = html2lines(page, url)
    return res


def line_correction(lines, max_size=100):
    out_lines = []
    for line in lines:
        if len(line) < MIN_CHAR:
            continue

        if len(line) > max_size:
            doc = nlp(
                line[:MAX_CHAR]
            )  # We split lines into sentences, but for performance we take only the first 5k characters per line
            stack = ""
            for sent in doc.sents:
                if len(stack) > 0:
                    stack += " "
                stack += str(sent).strip()
                if len(stack) > max_size:
                    out_lines.append(stack)
                    stack = ""

            if (
                len(stack) > MIN_CHAR
            ):  # Ensure every lines in the out_lines suffice the MIN_CHAR restriction
                out_lines.append(stack)
        else:
            out_lines.append(line)

    return out_lines


def html2lines(page, url):
    out_lines = []

    if len(page.strip()) == 0 or page is None:
        return out_lines

    text = trafilatura.extract(page, url=url, config=DEFAULT_CONFIG)
    metadata = trafilatura.extract_metadata(page, default_url=url, config=DEFAULT_CONFIG).as_dict()
    reset_caches()

    if text is None:
        return out_lines

    return text.split(
        "\n"
    ), text, metadata  # We just spit out the entire page, so need to reformat later.
