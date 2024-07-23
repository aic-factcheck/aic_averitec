import os
import argparse
import csv
from time import sleep
import time
import json
import numpy as np
import fitz
import pandas as pd
import requests
from scraping.html2lines import url2lines, line_correction

csv.field_size_limit(100000000)

MAX_RETRIES = 3
TIMEOUT = 5  # time limit for request


def scrape_text_from_url(url, temp_name):
    response = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                sleep(3)  # Wait before retrying

    if (
        response is None or response.status_code == 503
    ):  # trafilatura does not handle retry with 503, often waiting 24 hours as overwritten by the html
        return []

    if url.endswith(".pdf"):
        with open(f"pdf_dir/{temp_name}.pdf", "wb") as f:
            f.write(response.content)

        extracted_text = ""
        doc = fitz.open(f"pdf_dir/{temp_name}.pdf")
        for page in doc:  # iterate the document pages
            extracted_text += page.get_text() if page.get_text() else ""

        return line_correction(extracted_text.split("\n")), extracted_text, {}

    lines, text, metadata = url2lines(url)
    return line_correction(lines), text, metadata


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Scraping text from URLs.")
    parser.add_argument(
        "-i",
        "--tsv_input_file",
        type=str,
        help="The path of the input files containing URLs from Google search.",
    )
    parser.add_argument(
        "-o",
        "--json_output_dir",
        type=str,
        default="output",
        help="The output JSON file to save the scraped data.",
    )
    parser.add_argument(
        "--overwrite_out_file",
        action="store_true",
    )

    args = parser.parse_args()

    assert (
        os.path.splitext(args.tsv_input_file)[-1] == ".tsv"
    ), "The input should be a tsv file."

    os.makedirs(args.json_output_dir, exist_ok=True)

    total_scraped, empty, total_failed = 0, 0, 0

    print(f"Processing files {args.tsv_input_file}")

    st = time.time()

    claim_id = os.path.splitext(os.path.basename(args.tsv_input_file))[0]
    json_output_path = os.path.join(args.json_output_dir, f"{claim_id}.json")

    lines_skipped = 0
    if os.path.exists(json_output_path):
        if args.overwrite_out_file:
            os.remove(json_output_path)
        else:
            with open(json_output_path, "r", encoding="utf-8") as json_file:
                existing_data = json_file.readlines()
                lines_skipped = len(existing_data)
                print(f"    Skipping {lines_skipped} lines in {json_output_path}")

    # Some tsv files will fail to be loaded, try different libs to to load them
    try:
        df = pd.read_csv(args.tsv_input_file, sep="\t", header=None)
        data = df.values
        print("Data loaded successfully with Pandas.")

    except Exception as e:
        print("Error loading with csv:", e)
        try:
            data = np.genfromtxt(
                args.tsv_input_file, delimiter="\t", dtype=None, encoding=None
            )
            print("Data loaded successfully with NumPy.")
        except Exception as e:
            print("Error loading with NumPy:", e)
            try:
                data = []
                with open(args.tsv_input_file, "r", newline="") as tsvfile:
                    reader = csv.reader(tsvfile, delimiter="\t")
                    for row in reader:
                        data.append(row)
                print("Data loaded successfully with csv.")
            except Exception as e:
                print("Error loading with csv:", e)
                data = None

    if len(data) == lines_skipped:
        print("    No more lines need to be processed!")
    else:
        with open(json_output_path, "a", encoding="utf-8") as json_file:
            for index, row in enumerate(data):
                if index < lines_skipped:
                    continue
                url = row[2]
                json_data = {
                    "claim_id": claim_id,
                    "type": row[1],
                    "query": row[3],
                    "url": url,
                    "url2text": [],
                }
                print(f"Scraping text for url_{index}: {url}!")
                try:
                    scrape_result_lines, scrapte_result_text, scrape_result_metadata = scrape_text_from_url(url, claim_id)
                    json_data["url2text"] = scrape_result_lines
                    json_data["url2page"] = scrapte_result_text
                    json_data["url2metadata"] = scrape_result_metadata

                    if len(json_data["url2text"]) > 0:
                        total_scraped += 1
                    else:
                        empty += 1

                except Exception as e:
                    total_failed += 1

            json_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            json_file.flush()

        print(f"Output for {args.tsv_input_file} saved to {json_output_path}")
        elapsed_time = time.time() - st
        elapsed_minutes = int(elapsed_time // 60)
        elapsed_seconds = int(elapsed_time % 60)
        print(f"Time elapsed: {elapsed_minutes}min {elapsed_seconds}sec")
        print(f"{total_scraped} scraped, {empty} empty, {total_failed} failed")
