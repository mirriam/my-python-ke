import pandas as pd
import requests
from bs4 import BeautifulSoup
import base64
import time
import re
import hashlib
import nltk
from requests.exceptions import RequestException
import json
import os
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import logging
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import language_tool_python
import torch
from huggingface_hub import login

# Set CUDA_LAUNCH_BLOCKING for debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='debug.log')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.handlers = [h for h in logger.handlers if not isinstance(h, logging.StreamHandler)]
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# Initialize language tool
tool = language_tool_python.LanguageTool('en-US')

# Initialize model and tokenizer
device = torch.device("cpu")  # Always CPU
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()
model.to(device)

# Initialize sentence transformer on CPU
similarity_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

# Constants
MAX_TOTAL_TOKENS = 3000
MAX_RETURN_SEQUENCES = 4
WP_URL = "https://kenya.mimusjobs.com/wp-json/wp/v2/job-listings"
WP_COMPANY_URL = "https://kenya.mimusjobs.com/wp-json/wp/v2/company"
WP_MEDIA_URL = "https://kenya.mimusjobs.com/wp-json/wp/v2/media"
WP_USERNAME = "admin"
WP_APP_PASSWORD = "Xljs I1VY 7XL0 F45N 3Wsv 5qcv"
PROCESSED_IDS_FILE = "kenya_processed_job_ids.csv"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36'
}
JOB_TYPE_MAPPING = {
    "Full-time": "full-time",
    "Part-time": "part-time",
    "Contract": "contract",
    "Temporary": "temporary",
    "Freelance": "freelance",
    "Internship": "internship",
    "Volunteer": "volunteer"
}

def sanitize_text(text, is_url=False, is_email=False):
    """Sanitize input text by removing unwanted characters and normalizing."""
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    if not text:
        return ""
    if is_url or is_email:
        text = re.sub(r'[\r\t\f\v]', '', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text).strip()
        return text
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'—+', '', text)
    text = re.sub(r'←+', '', text)
    text = re.sub(r'[^\x20-\x7E\n\u00C0-\u017F]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text).strip()
    return text

def clean_description(text):
    """Clean paraphrased text using LanguageTool for grammar and style."""
    try:
        matches = tool.check(text)
        corrected_text = language_tool_python.utils.correct(text, matches)
        return corrected_text
    except Exception as e:
        logger.error(f"Error in grammar correction: {str(e)}")
        return text

def is_good_paraphrase(original: str, candidate: str) -> float:
    """Calculate cosine similarity between original and paraphrased text."""
    try:
        embeddings = similarity_model.encode([original, candidate], convert_to_tensor=True)
        sim_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return sim_score
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0

def is_grammatically_correct(text):
    """Check if text is grammatically correct with minimal issues."""
    matches = tool.check(text)
    return len(matches) < 3

def extract_nouns(text):
    """Extract nouns (NN, NNS, NNP, NNPS) from text using NLTK POS tagging."""
    try:
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        nouns = [word for word, pos in tagged if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        return nouns
    except Exception as e:
        logger.error(f"Error extracting nouns from {text}: {str(e)}")
        return []

def contains_nouns(paraphrase, required_nouns):
    """Check if the paraphrase contains all required nouns."""
    if not required_nouns:
        return True
    paraphrase_lower = paraphrase.lower()
    return all(noun.lower() in paraphrase_lower for noun in required_nouns)

def extract_capitalized_words(text):
    """Extract words with specific capitalization (e.g., proper nouns, acronyms)."""
    words = re.findall(r'\b[A-Z][a-zA-Z]*\b', text)
    return {word: word for word in words if len(word) > 1}

def restore_capitalization(paraphrased, capitalized_words):
    """Restore original capitalization of specified words in the paraphrased text."""
    paraphrased_lower = paraphrased.lower()
    result = paraphrased
    for lower_word, orig_word in capitalized_words.items():
        pattern = r'\b' + re.escape(lower_word) + r'\b'
        result = re.sub(pattern, orig_word, result, flags=re.IGNORECASE)
    return result

def score_paraphrase(original, paraphrased, target_wc):
    """Calculate score for paraphrased text based on similarity and length."""
    sim = is_good_paraphrase(original, paraphrased)
    wc = len(paraphrased.split())
    length_penalty = abs(wc - target_wc) / max(target_wc, 1)
    return (sim + (1 - length_penalty)) / 2, sim, wc

def paraphrase_strict_title(title, max_attempts=3, max_sub_attempts=2):
    def has_repetitions(text):
        tokens = text.lower().split()
        seen = set()
        for i in range(len(tokens) - 2):
            ngram = tuple(tokens[i:i + 3])
            if ngram in seen:
                return True
            seen.add(ngram)
        return False

    def contains_banned_phrase(text, banned_list):
        text_lower = text.lower()
        critical_phrases = [
            "Rewrite the following", "Paraphrased title", "Professionally rewrite",
            "Keep it short", "Use different phrasing", "Short (5–12 words)",
            "Paraphrase", "Paraphrased", "Paraphrasing", "Paraphrased version",
            "Summary", "Summarised", "Summarized", "Summarizing", "Summarising",
            "None.", "None", "none", ".", ":"
        ]
        for phrase in critical_phrases:
            if phrase.lower() in text_lower:
                start_idx = text_lower.find(phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(text), start_idx + len(phrase) + 20)
                context_snippet = text[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(text):
                    context_snippet = context_snippet + "..."
                return True, phrase, context_snippet
        return False, None, None

    clean_title = sanitize_text(title)
    if not clean_title:
        logger.error("Input title is empty after sanitization.")
        return title

    nouns = extract_nouns(clean_title)
    capitalized_words = extract_capitalized_words(clean_title)
    nouns_str = ", ".join(nouns) if nouns else "none"
    logger.debug(f"Extracted nouns from title '{clean_title}': {nouns}")
    logger.debug(f"Extracted capitalized words from title '{clean_title}': {list(capitalized_words.values())}")

    prompt = (
        f"Rewrite the following job title professionally, using different phrasing while preserving the meaning. "
        f"Keep it short (5–12 words) and avoid duplicating words. "
        f"Preserve the following nouns exactly as they are: {nouns_str}.\n{clean_title}"
    )

    encoding = tokenizer.encode_plus(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOTAL_TOKENS
    ).to(device)

    available_output_tokens = 60
    target_word_count = len(clean_title.split())
    min_wc = max(1, int(target_word_count * 0.6))
    max_wc = min(12, int(target_word_count * 1.4))

    best_paraphrase = None
    best_score = -1
    best_attempt = ""
    best_metadata = ""

    for attempt in range(max_attempts):
        sub_attempt = 0
        valid_paraphrase_found = False

        while not valid_paraphrase_found and sub_attempt < max_sub_attempts:
            try:
                with torch.no_grad():
                    output = model.generate(
                        input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask'],
                        max_new_tokens=available_output_tokens,
                        do_sample=True,
                        top_k=40,
                        top_p=0.95,
                        temperature=0.8 + 0.1 * sub_attempt,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                        num_return_sequences=MAX_RETURN_SEQUENCES
                    )

                decoded_outputs = [
                    tokenizer.decode(seq, skip_special_tokens=True).strip()
                    for seq in output
                ]

                for idx, d in enumerate(decoded_outputs):
                    paraphrased = d.replace(prompt, "").strip() if prompt in d else d.strip()
                    paraphrased = clean_description(paraphrased)
                    paraphrased = restore_capitalization(paraphrased, capitalized_words)

                    if not paraphrased or len(paraphrased.split()) < 1:
                        logger.info(f"⛔ Rejected due to empty or too short: \"{paraphrased}\"")
                        continue

                    is_banned, banned_phrase, context_snippet = contains_banned_phrase(paraphrased, [])
                    if is_banned:
                        logger.info(f"⛔ Rejected due to banned phrase '{banned_phrase}' in context: '{context_snippet}'")
                        print(f"⛔ Rejected due to banned phrase '{banned_phrase}' in context: '{context_snippet}'")
                        continue
                    if has_repetitions(paraphrased):
                        logger.info(f"⛔ Rejected due to repeated phrases: \"{paraphrased}\"")
                        continue
                    if not is_grammatically_correct(paraphrased):
                        logger.info(f"⛔ Rejected due to grammar: \"{paraphrased}\"")
                        continue
                    if not contains_nouns(paraphrased, nouns):
                        logger.info(f"⛔ Rejected due to missing nouns: \"{paraphrased}\" (required: {nouns})")
                        continue

                    score, sim, wc = score_paraphrase(title, paraphrased, target_word_count)
                    first_diff = not paraphrased.lower().startswith(title.lower())

                    print(f"📝 Attempt {attempt + 1}.{sub_attempt + 1}, Option {idx + 1}")
                    print(f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}")
                    print(f"→ Paraphrased: {paraphrased}\n")

                    is_valid = (
                        min_wc <= wc <= max_wc
                        and sim >= (0.65 if target_word_count > 5 else 0.6)
                        and first_diff
                    )

                    if is_valid:
                        print(f"✅ Picked from attempt {attempt + 1}.{sub_attempt + 1}, option {idx + 1}")
                        print(f"→ {paraphrased}\n")
                        return paraphrased

                    if first_diff and score > best_score:
                        best_score = score
                        best_paraphrase = paraphrased
                        best_attempt = f"{attempt + 1}.{sub_attempt + 1}, option {idx + 1}"
                        best_metadata = (
                            f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}\n"
                            f"→ Paraphrased: {paraphrased}"
                        )

                sub_attempt += 1
                time.sleep(0.5 * (2 ** sub_attempt))

            except Exception as e:
                logger.error(f"Error during attempt {attempt + 1}, sub-attempt {sub_attempt + 1}: {str(e)}")
                sub_attempt += 1
                time.sleep(0.5 * (2 ** sub_attempt))

        time.sleep(1)

    if best_paraphrase:
        print(f"✅ Picked fallback from attempt {best_attempt}")
        print(best_metadata + "\n")
        return best_paraphrase

    print("❌ Fallback to original title.\n")
    return clean_title

def paraphrase_strict_company(text, max_attempts=2, max_sub_attempts=2):
    def contains_prompt(para):
        prompt_phrases = [
            "Rephrase the following company details paragraph",
            "Rephrase the company details",
            "Paragraph professionally, preserving all key details",
            "Rewrite the following",
            "Rephrase the paragraph below",
            "Rephrase the following company details",
            "Preserving all key details",
            "Tone and structure",
            "Keep the length approximately the same",
            "Do your company information paragraph need improvements",
            "Paraphrase", "Paraphrased", "Paraphrasing", "Paragraph", "Company details"
        ]
        para_lower = para.lower()
        for phrase in prompt_phrases:
            if phrase.lower() in para_lower:
                start_idx = para_lower.find(phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(para), start_idx + len(phrase) + 20)
                context_snippet = para[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(para):
                    context_snippet = context_snippet + "..."
                return True, phrase, context_snippet
        return False, None, None

    clean_text = sanitize_text(text)
    if not clean_text:
        logger.error("Input text is empty after sanitization.")
        return text

    capitalized_words = extract_capitalized_words(clean_text)
    logger.debug(f"Extracted capitalized words from text: {list(capitalized_words.values())}")

    paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
    final_paraphrased = []

    for idx, para in enumerate(paragraphs):
        print(f"\n🔹 Paraphrasing Paragraph {idx + 1}/{len(paragraphs)}")

        prompt = (
            f"Rephrase the following company details paragraph professionally, preserving all key details, tone, and structure. "
            f"Keep the length approximately the same and avoid repeating the input format:\n{para}"
        )

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_token_len = len(prompt_tokens)

        if prompt_token_len > MAX_TOTAL_TOKENS - 200:
            logger.warning(f"Prompt for paragraph {idx + 1} too long, truncating to fit.")
            para = " ".join(para.split()[:int((MAX_TOTAL_TOKENS - 200) / 4)])
            prompt = (
                f"Rephrase the following company details paragraph professionally, preserving all key details, tone, and structure. "
                f"Keep the length approximately the same:\n{para}"
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_token_len = len(prompt_tokens)

        available_output_tokens = max(200, MAX_TOTAL_TOKENS - prompt_token_len)
        target_word_count = len(para.split())
        tolerance = 0.25
        min_wc = int(target_word_count * (1 - tolerance))
        max_wc = int(target_word_count * (1 + tolerance))

        encoding = tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOTAL_TOKENS
        ).to(device)

        best_paraphrase = None
        best_score = -1
        best_attempt = ""
        best_metadata = ""

        for attempt in range(max_attempts):
            sub_attempt = 0
            valid_paraphrase_found = False

            while not valid_paraphrase_found and sub_attempt < max_sub_attempts:
                try:
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'],
                            max_new_tokens=available_output_tokens,
                            do_sample=True,
                            top_k=40,
                            top_p=0.95,
                            temperature=0.8 + 0.1 * sub_attempt,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            num_return_sequences=MAX_RETURN_SEQUENCES
                        )

                    decoded_outputs = [
                        tokenizer.decode(seq, skip_special_tokens=True).strip()
                        for seq in output
                    ]

                    for idx2, d in enumerate(decoded_outputs):
                        paraphrased = d.replace(prompt, "").strip() if prompt in d else d.strip()
                        paraphrased = clean_description(paraphrased)
                        paraphrased = restore_capitalization(paraphrased, capitalized_words)

                        if not paraphrased or len(paraphrased.split()) < 1:
                            logger.info(f"⛔ Paragraph {idx + 1} rejected due to empty or too short: \"{paraphrased}\"")
                            continue

                        is_banned, banned_phrase, context_snippet = contains_prompt(paraphrased)
                        if is_banned:
                            logger.info(f"⛔ Paragraph {idx + 1} rejected due to banned phrase '{banned_phrase}' in context: '{context_snippet}'")
                            print(f"⛔ Paragraph {idx + 1} rejected due to banned phrase '{banned_phrase}' in context: '{context_snippet}'")
                            continue
                        if not is_grammatically_correct(paraphrased):
                            logger.info(f"⛔ Paragraph {idx + 1} rejected due to grammar: \"{paraphrased}\"")
                            continue

                        score, sim, wc = score_paraphrase(para, paraphrased, target_word_count)
                        first_diff = not paraphrased.lower().startswith(para.lower())

                        print(f"📝 Paragraph {idx + 1}, Attempt {attempt + 1}.{sub_attempt + 1}, Option {idx2 + 1}")
                        print(f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}")
                        print(f"→ Paraphrased: {paraphrased}\n")

                        is_valid = (
                            min_wc <= wc <= max_wc
                            and sim >= 0.65
                            and first_diff
                        )

                        if is_valid:
                            print(f"✅ Paragraph {idx + 1} picked from attempt {attempt + 1}.{sub_attempt + 1}, option {idx2 + 1}")
                            print(f"→ {paraphrased}\n")
                            final_paraphrased.append(paraphrased)
                            valid_paraphrase_found = True
                            break

                        if first_diff and score > best_score:
                            best_score = score
                            best_paraphrase = paraphrased
                            best_attempt = f"{attempt + 1}.{sub_attempt + 1}, option {idx2 + 1}"
                            best_metadata = (
                                f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}\n"
                                f"→ Paraphrased: {paraphrased}"
                            )

                    if valid_paraphrase_found:
                        break

                    sub_attempt += 1
                    time.sleep(0.5 * (2 ** sub_attempt))

                except Exception as e:
                    logger.error(f"Error during paragraph {idx + 1}, attempt {attempt + 1}, sub-attempt {sub_attempt + 1}: {str(e)}")
                    sub_attempt += 1
                    time.sleep(0.5 * (2 ** sub_attempt))

            if valid_paraphrase_found:
                break

            time.sleep(1)

        if not valid_paraphrase_found:
            if best_paraphrase:
                print(f"✅ Paragraph {idx + 1} picked fallback from attempt {best_attempt}")
                print(best_metadata + "\n")
                final_paraphrased.append(best_paraphrase)
            else:
                print(f"❌ Paragraph {idx + 1} fallback to original paragraph.")
                final_paraphrased.append(para)

    result = "\n".join(final_paraphrased)
    if not result.strip():
        logger.warning("No valid paraphrased text produced, returning original.")
        return text
    return result

def paraphrase_strict_tagline(company_tagline, max_attempts=5):
    clean_text = sanitize_text(company_tagline)
    if not clean_text:
        logger.error(f"Input text is empty after sanitization: {company_tagline}")
        print("Error: Input text is empty after sanitization.")
        return company_tagline

    capitalized_words = extract_capitalized_words(clean_text)
    logger.debug(f"Extracted capitalized words from tagline: {list(capitalized_words.values())}")

    target_word_count = max(len(clean_text.split()), 8)
    min_word_count = 4
    max_word_count = 15

    rejected_phrases = [
        "Paraphrased tagline", "Rewrite the following", "Original tagline",
        "Professionally rewritten", "Crisp and impactful", "Summary:",
        "Short and professional", "Keep it short", "###", "Tagline:",
        "Output:", "Company summary", "Paraphrased version", "Rephrased version",
        "Paraphrase", "Paraphrased", "Paraphrasing", "Summarized", "Summarised",
        "Summarizing", "Summarising", "Summary"
    ]

    def contains_rejected_phrase(text):
        lower = text.lower()
        for bad_phrase in rejected_phrases:
            if bad_phrase.lower() in lower:
                start_idx = lower.find(bad_phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(text), start_idx + len(bad_phrase) + 20)
                context_snippet = text[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(text):
                    context_snippet = context_snippet + "..."
                return True, bad_phrase, context_snippet
        return False, None, None

    def first_sentence_diff(original, paraphrased):
        orig_first = original.split(".")[0].strip().lower()
        para_first = paraphrased.split(".")[0].strip().lower()
        return not para_first.startswith(orig_first)

    input_prompt = (
        f"Rewrite the following tagline into a crisp, professional, and meaningful summary. "
        f"Keep it short and impactful (5–12 words):\n\n"
        f"### Original ###\n{clean_text}\n\n### Paraphrased Tagline ###"
    )

    input_tokens = tokenizer.encode(input_prompt, add_special_tokens=True)
    max_length = min(len(input_tokens) + 50, 512)

    encoding = tokenizer.encode_plus(
        input_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    best_paraphrase = None
    best_score = -1
    best_meta = {"attempt": -1, "similarity": 0.0, "word_count": 0, "first_diff": False}

    for attempt in range(max_attempts):
        try:
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask'],
                    max_new_tokens=25,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.9,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=2,
                    num_return_sequences=6,
                    eos_token_id=tokenizer.eos_token_id
                )

            decoded_outputs = [
                tokenizer.decode(seq, skip_special_tokens=True).strip()
                for seq in outputs
            ]

            paraphrases = []
            for d in decoded_outputs:
                if "### Paraphrased Tagline ###" in d:
                    parts = d.split("### Paraphrased Tagline ###")
                    paraphrased = clean_description(parts[1]) if len(parts) >= 2 else clean_description(d)
                else:
                    paraphrased = clean_description(d)
                paraphrased = restore_capitalization(paraphrased, capitalized_words)
                paraphrases.append(paraphrased)

            for paraphrased in paraphrases:
                is_banned, banned_phrase, context_snippet = contains_rejected_phrase(paraphrased)
                if is_banned:
                    logger.info(f"⛔ Rejected tagline due to banned phrase '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\"")
                    print(f"⛔ Rejected tagline due to banned phrase '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\"")
                    continue

                if not is_grammatically_correct(paraphrased):
                    logger.info(f"⛔ Rejected tagline due to grammar: \"{paraphrased}\"")
                    continue

                word_count = len(paraphrased.split())
                if word_count < min_word_count or word_count > max_word_count:
                    continue

                similarity = is_good_paraphrase(clean_text, paraphrased)
                length_score = 1 - abs(target_word_count - word_count) / target_word_count
                score = similarity * 0.7 + length_score * 0.3
                first_diff = first_sentence_diff(clean_text, paraphrased)

                print(f"Attempt {attempt + 1}: \"{paraphrased}\" | Words: {word_count} | Similarity: {similarity:.2f} | Score: {score:.2f} | First sentence different: {first_diff}")

                if first_diff and score > best_score:
                    best_paraphrase = paraphrased
                    best_meta = {
                        "attempt": attempt + 1,
                        "similarity": similarity,
                        "word_count": word_count,
                        "first_diff": first_diff
                    }

        except Exception as e:
            logger.error(f"Error during paraphrasing attempt {attempt + 1}: {str(e)}")

        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)

    if best_paraphrase:
        logger.info(
            f"✅ Picked tagline from attempt {best_meta['attempt']} "
            f"(words: {best_meta['word_count']}, similarity: {best_meta['similarity']:.2f}, score: {best_score:.2f}, first sentence different: {best_meta['first_diff']})"
        )
        print(
            f"\n✅ Picked tagline from attempt {best_meta['attempt']} "
            f"(words: {best_meta['word_count']}, similarity: {best_meta['similarity']:.2f}, score: {best_score:.2f}, first sentence different: {best_meta['first_diff']})"
        )
        return best_paraphrase

    logger.warning("No valid tagline candidates produced. Returning original.")
    return clean_text

def paraphrase_strict_description(text, max_attempts=2, max_sub_attempts=2):
    def contains_prompt(para):
        prompt_phrases = [
            "Rephrase the following job description paragraph",
            "Rephrase the job description",
            "Paragraph professionally, preserving all key details",
            "Rewrite the following",
            "Rephrase the paragraph below",
            "Rephrase the following job description",
            "Preserving all key details",
            "Tone and structure",
            "Keep the length approximately the same",
            "Job description paragraph professionally",
            "Paraphrase", "Paraphrased", "Paraphrase the following",
            "Paraphrase the job description", "Paraphrasing",
            "Job description", "Job description paragraph"
        ]
        para_lower = para.lower()
        for phrase in prompt_phrases:
            if phrase.lower() in para_lower:
                start_idx = para_lower.find(phrase.lower())
                context_start = max(0, start_idx - 20)
                context_end = min(len(para), start_idx + len(phrase) + 20)
                context_snippet = para[context_start:context_end]
                if context_start > 0:
                    context_snippet = "..." + context_snippet
                if context_end < len(para):
                    context_snippet = context_snippet + "..."
                return True, phrase, context_snippet
        return False, None, None

    clean_text = sanitize_text(text)
    if not clean_text:
        logger.error("Input text is empty after sanitization.")
        return text

    capitalized_words = extract_capitalized_words(clean_text)
    logger.debug(f"Extracted capitalized words from text: {list(capitalized_words.values())}")

    paragraphs = [p.strip() for p in clean_text.split('\n') if p.strip()]
    final_paraphrased = []

    for idx, para in enumerate(paragraphs):
        print(f"\n🔹 Paraphrasing Paragraph {idx + 1}/{len(paragraphs)}")

        prompt = (
            f"Rephrase the following job description paragraph professionally, preserving all key details, tone, and structure. "
            f"Keep the length approximately the same and avoid repeating the input format:\n{para}"
        )

        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_token_len = len(prompt_tokens)

        if prompt_token_len > MAX_TOTAL_TOKENS - 200:
            logger.warning(f"Prompt for paragraph {idx + 1} too long, truncating to fit.")
            para = " ".join(para.split()[:int((MAX_TOTAL_TOKENS - 200) / 4)])
            prompt = (
                f"Rephrase the following job description paragraph professionally, preserving all key details, tone, and structure. "
                f"Keep the length approximately the same:\n{para}"
            )
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            prompt_token_len = len(prompt_tokens)

        available_output_tokens = max(200, MAX_TOTAL_TOKENS - prompt_token_len)
        target_word_count = len(para.split())
        tolerance = 0.25
        min_wc = int(target_word_count * (1 - tolerance))
        max_wc = int(target_word_count * (1 + tolerance))

        encoding = tokenizer.encode_plus(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOTAL_TOKENS
        ).to(device)

        best_paraphrase = None
        best_score = -1
        best_attempt = ""
        best_metadata = ""

        for attempt in range(max_attempts):
            sub_attempt = 0
            valid_paraphrase_found = False

            while not valid_paraphrase_found and sub_attempt < max_sub_attempts:
                try:
                    with torch.no_grad():
                        output = model.generate(
                            input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'],
                            max_new_tokens=available_output_tokens,
                            do_sample=True,
                            top_k=40,
                            top_p=0.95,
                            temperature=0.8 + 0.1 * sub_attempt,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            num_return_sequences=MAX_RETURN_SEQUENCES
                        )

                    decoded = [tokenizer.decode(seq, skip_special_tokens=True).strip() for seq in output]

                    for option_index, d in enumerate(decoded):
                        paraphrased = d.replace(prompt, "").strip() if prompt in d else d.strip()
                        paraphrased = clean_description(paraphrased)
                        paraphrased = restore_capitalization(paraphrased, capitalized_words)

                        if not paraphrased or len(paraphrased.split()) < 5:
                            logger.info(f"⛔ Rejected due to empty or too short: \"{paraphrased}\"")
                            continue
                        is_banned, banned_phrase, context_snippet = contains_prompt(paraphrased)
                        if is_banned:
                            print(f"❌ Rejected due to prompt echo (phrase: '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\")")
                            logger.info(f"❌ Rejected due to prompt echo (phrase: '{banned_phrase}' in context: '{context_snippet}' in output: \"{paraphrased}\")")
                            continue

                        score, sim, wc = score_paraphrase(para, paraphrased, target_word_count)
                        first_diff = not paraphrased.lower().startswith(para.lower())

                        print(f"📝 Attempt {attempt + 1}.{sub_attempt + 1}, Option {option_index + 1}")
                        print(f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}")
                        print(f"→ Paraphrased: {paraphrased}\n")

                        is_valid = (
                            min_wc <= wc <= max_wc
                            and sim >= (0.65 if target_word_count > 10 else 0.6)
                            and first_diff
                            and is_grammatically_correct(paraphrased)
                        )

                        if is_valid:
                            print(f"✅ Picked from attempt {attempt + 1}.{sub_attempt + 1}, option {option_index + 1}")
                            final_paraphrased.append(clean_description(paraphrased))
                            valid_paraphrase_found = True
                            break

                        if first_diff and score > best_score:
                            best_score = score
                            best_paraphrase = paraphrased
                            best_attempt = f"{attempt + 1}.{sub_attempt + 1}, option {option_index + 1}"
                            best_metadata = (
                                f"↪ Words: {wc}, Sim: {sim:.2f}, Score: {score:.2f}, First different: {first_diff}\n"
                                f"→ Paraphrased: {paraphrased}"
                            )

                    if not valid_paraphrase_found:
                        sub_attempt += 1
                        time.sleep(0.5 * (2 ** sub_attempt))

                except Exception as e:
                    logger.error(f"Error during attempt {attempt + 1}, sub-attempt {sub_attempt + 1} for paragraph {idx + 1}: {str(e)}")
                    sub_attempt += 1
                    time.sleep(0.5 * (2 ** sub_attempt))

            if valid_paraphrase_found:
                break
            time.sleep(1)

        if not valid_paraphrase_found:
            if best_paraphrase:
                print(f"✅ Picked fallback from attempt {best_attempt}")
                print(best_metadata + "\n")
                final_paraphrased.append(clean_description(best_paraphrase))
            else:
                print(f"❌ Paragraph {idx + 1} fallback to original.\n")
                final_paraphrased.append(para)

    return "\n\n".join(final_paraphrased)

def paraphrase_title_and_description(title, description, index, max_attempts=5):
    try:
        rewritten_title = paraphrase_strict_title(title, max_attempts=max_attempts)
        rewritten_description = paraphrase_strict_description(description, max_attempts=2)
        combined = f"{rewritten_title}\n\n{rewritten_description}"
        return combined, rewritten_title, rewritten_description
    except Exception as e:
        logger.error(f"Error paraphrasing for index {index}: {str(e)}")
        return title + "\n\n" + description, title, description

def validate_application_method(application, is_email=False):
    """Validate application method (URL or email)."""
    if not application:
        return False
    if is_email:
        return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', application))
    try:
        parsed = urlparse(application)
        if not parsed.scheme or not parsed.netloc:
            return False
        excluded_domains = ['mysalaryscale.com', 'myjobmag.co.ke', 'linkedin.com', 'twitter.com', 'facebook.com']
        if any(domain in parsed.netloc.lower() for domain in excluded_domains):
            return False
        return True
    except ValueError:
        return False

def clean_application_url(url):
    """Clean application URL by removing query parameters and fragments."""
    try:
        parsed = urlparse(url)
        cleaned = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
        return cleaned
    except ValueError as e:
        logger.error(f"Error cleaning URL {url}: {str(e)}")
        return url

def add_three_months_to_date(date_str):
    """Add three months to a date string and return in the same format."""
    try:
        date_str = re.sub(r'^Posted:\s*', '', date_str.strip())
        date_obj = datetime.strptime(date_str, '%b %d, %Y')
        new_date = date_obj + timedelta(days=90)
        return new_date.strftime('%b %d, %Y')
    except ValueError as e:
        logger.error(f"Error adding three months to date {date_str}: {str(e)}")
        return datetime.now().strftime('%b %d, %Y')

def extract_job_title(title):
    """Extract the job title by removing company name and other noise."""
    if not title:
        return ""
    title = sanitize_text(title)
    title = re.sub(r'\s+at\s+.*$', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+\(.*?\)$', '', title)
    title = re.sub(r'\s+\-.*$', '', title)
    return title.strip()

def save_company_to_wordpress(index, company_data):
    try:
        company_name = company_data.get('company_name', 'Unknown Company')
        if not company_name or company_name == 'Unknown Company':
            logger.warning(f"Skipping company post for index {index}: Invalid company name")
            return None, None

        # Generate slug from company name
        slug = company_name.lower().replace(' ', '-').replace('&', 'and')
        slug = re.sub(r'[^a-z0-9-]', '', slug)

        # Check if company post already exists by slug
        headers = {
            'Authorization': 'Basic ' + base64.b64encode(f"{WP_USERNAME}:{WP_APP_PASSWORD}".encode()).decode('utf-8'),
            'User-Agent': HEADERS['User-Agent']
        }
        response = requests.get(f"{WP_COMPANY_URL}?slug={slug}", headers=headers, timeout=10)
        response.raise_for_status()
        existing_posts = response.json()
        if existing_posts:
            logger.info(f"Company {company_name} already exists with slug {slug}. Skipping post.")
            print(f"Company {company_name} already exists. Post ID: {existing_posts[0]['id']}, URL: {existing_posts[0]['link']}")
            return existing_posts[0]['id'], existing_posts[0]['link']

        # Prepare company post data
        company_post_data = {
            "title": company_name,
            "content": company_data.get('company_details', ''),
            "status": "publish",
            "slug": slug,
            "meta": {
                "industry": company_data.get('company_industry', ''),
                "founded": company_data.get('company_founded', ''),
                "type": company_data.get('company_type', ''),
                "website": company_data.get('company_website', ''),
                "address": company_data.get('company_address', '')
            }
        }

        # Post company to WordPress
        response = requests.post(WP_COMPANY_URL, headers=headers, json=company_post_data, timeout=10)
        response.raise_for_status()
        post = response.json()
        post_id = post.get('id')
        post_url = post.get('link')
        logger.info(f"Posted company {company_name} to WordPress. Post ID: {post_id}, URL: {post_url}")

        # Handle company logo if available
        company_logo = company_data.get('company_logo', [''])[0]
        if company_logo:
            try:
                logo_response = requests.get(company_logo, headers=HEADERS, timeout=10)
                logo_response.raise_for_status()
                logo_filename = f"{slug}-logo{os.path.splitext(company_logo)[1]}"
                media_headers = headers.copy()
                media_headers['Content-Type'] = 'image/jpeg' if company_logo.endswith('.jpg') or company_logo.endswith('.jpeg') else 'image/png'
                media_headers['Content-Disposition'] = f'attachment; filename={logo_filename}'
                media_response = requests.post(WP_MEDIA_URL, headers=media_headers, data=logo_response.content, timeout=10)
                media_response.raise_for_status()
                media = media_response.json()
                media_id = media.get('id')
                requests.post(f"{WP_COMPANY_URL}/{post_id}", headers=headers, json={"featured_media": media_id}, timeout=10)
                logger.info(f"Uploaded logo for company {company_name}. Media ID: {media_id}")
            except RequestException as e:
                logger.error(f"Error uploading logo for company {company_name}: {str(e)}")

        return post_id, post_url
    except RequestException as e:
        logger.error(f"Error posting company {company_name} to WordPress: {str(e)}")
        print(f"Error posting company {company_name}: {str(e)}")
        return None, None

def save_article_to_wordpress(index, job_data, rewritten_title, rewritten_description, application):
    try:
        headers = {
            'Authorization': 'Basic ' + base64.b64encode(f"{WP_USERNAME}:{WP_APP_PASSWORD}".encode()).decode('utf-8'),
            'User-Agent': HEADERS['User-Agent']
        }
        job_type_slug = JOB_TYPE_MAPPING.get(job_data.get('Job Type', ''), 'full-time')
        post_data = {
            "title": rewritten_title,
            "content": rewritten_description,
            "status": "publish",
            "meta": {
                "job_type": job_type_slug,
                "location": job_data.get('Location', ''),
                "qualifications": job_data.get('Qualifications', ''),
                "experience": job_data.get('Experience', ''),
                "field": job_data.get('Field', ''),
                "date_posted": job_data.get('Date Posted', ''),
                "deadline": job_data.get('Deadline', ''),
                "application": application,
                "company": job_data.get('Company', 'Unknown Company'),
                "company_logo": job_data.get('Company Logo', ''),
                "company_website": job_data.get('Company Website', ''),
                "job_url": job_data.get('Job URL', ''),
                "job_id": job_data.get('Job ID', '')
            }
        }
        response = requests.post(WP_URL, headers=headers, json=post_data, timeout=10)
        response.raise_for_status()
        post = response.json()
        post_id = post.get('id')
        post_url = post.get('link')
        logger.info(f"Posted job {index} to WordPress. Post ID: {post_id}, URL: {post_url}")
        return post_id, post_url
    except RequestException as e:
        logger.error(f"Error posting job {index} to WordPress: {str(e)}")
        print(f"Error posting job {index}: {str(e)}")
        return None, None

def scrape_job_details(job_url):
    try:
        resp = requests.get(job_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        job_title_elem = soup.select_one('h2.mag-b') or soup.select_one('h1')
        job_title = job_title_elem.text.replace("Method of Application", "").strip() if job_title_elem else ""
        parts = job_title.split(" at ")
        trimmed_parts = [part.strip() for part in parts]
        job_title_clean = trimmed_parts[0] if trimmed_parts else job_title
        company_name = trimmed_parts[1] if len(trimmed_parts) > 1 else None
        if not company_name:
            logo_elem = soup.select_one('div.company-logo > img')
            company_name = logo_elem.get('alt', '').strip() if logo_elem and logo_elem.get('alt') else None
        if not company_name:
            company_name_elem = soup.select_one('#wrap-comp-jobs > div.company-jobs > h1') or soup.select_one('h1.company-name') or soup.select_one('div.company-info > h2')
            company_name = company_name_elem.text.replace("Recruitment", "").strip() if company_name_elem else "Unknown Company"
        job_type = soup.select_one('#printable > ul > li:nth-child(1) > span.jkey-info').text.strip() if soup.select_one('#printable > ul > li:nth-child(1) > span.jkey-info') else ""
        job_qualifications = soup.select_one('#printable > ul > li:nth-child(2) > span.jkey-info').text.strip() if soup.select_one('#printable > ul > li:nth-child(2) > span.jkey-info') else ""
        job_experiences = soup.select_one('#printable > ul > li:nth-child(3) > span.jkey-info').text.strip() if soup.select_one('#printable > ul > li:nth-child(3) > span.jkey-info') else ""
        job_locations = soup.select_one('ul.job-info > li:nth-child(4) > span.jkey-info').text.strip() if soup.select_one('ul.job-info > li:nth-child(4) > span.jkey-info') else ""
        if not job_locations:
            job_locations = soup.select_one('#printable > ul > li:nth-child(4) > span.jkey-info').text.strip() if soup.select_one('#printable > ul > li:nth-child(4) > span.jkey-info') else "Remote"
        logger.debug(f"Extracted location: {job_locations}")
        job_fields = soup.select_one('#printable > ul > li:nth-child(5) > span.jkey-info').text.strip() if soup.select_one('#printable > ul > li:nth-child(5) > span.jkey-info') else ""
        date_posted_str = soup.select_one('#posted-date').text.strip() if soup.select_one('#posted-date') else ""
        try:
            datetime.strptime(re.sub(r'^Posted:\s*', '', date_posted_str.strip()), '%b %d, %Y')
            new_date_string = add_three_months_to_date(date_posted_str)
        except ValueError:
            print(f"Invalid date format: {date_posted_str}")
            return None, None
        deadline_elem = soup.select_one('div.read-left-section > ul > li.read-head > div > div:nth-child(2)')
        deadline = deadline_elem.text.strip().replace("Deadline:", "").replace("Not specified", new_date_string).strip() if deadline_elem else new_date_string
        job_description = soup.select_one('div.job-details').text.strip() if soup.select_one('div.job-details') else ""
        application_detail = soup.select_one('#printable > div.mag-b.bm-b-30 > p').text.strip() if soup.select_one('#printable > div.mag-b.bm-b-30 > p') else ""
        job_description = job_description + (f"\n\nApplication Instructions: {application_detail}" if application_detail else "")
        if company_name == "Unknown Company" and job_description:
            company_match = re.search(r'(?:at|for|with)\s+([A-Z][\w\s&-]+)\b', job_description, re.IGNORECASE)
            company_name = company_match.group(1).strip() if company_match else "Unknown Company"
        if company_name == "Unknown Company":
            logger.warning(f"Failed to extract company name for job URL: {job_url}")
        application_text = soup.select_one('#printable > div.mag-b.bm-b-30') or soup.select_one('div.application-details') or soup.select_one('div.job-apply')
        application_text = application_text.text.strip() if application_text else ""
        extracted_email = None
        if application_text:
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', application_text)
            extracted_email = email_match.group(0) if email_match and validate_application_method(email_match.group(0), is_email=True) else ""
        application_url_elem = soup.select_one('#printable > div.mag-b.bm-b-30 > a') or soup.select_one('a.apply-button') or soup.select_one('a[href*="apply"]')
        application_url = application_url_elem.get('href', '') if application_url_elem else ""
        if application_url:
            if application_url.startswith('/'):
                application_url = 'https://www.myjobmag.co.ke' + application_url
            application_url = clean_application_url(application_url)
            if not validate_application_method(application_url):
                application_url = ""
                logger.warning(f"Invalid application URL after cleaning: {application_url}")
        application = application_url if application_url else extracted_email if extracted_email else ""
        if not application:
            logger.warning(f"No valid application method extracted for job URL: {job_url}")
        company_urls = ['https://www.myjobmag.co.ke' + a.get('href') for a in soup.select('#printable > a') if a.get('href')]
        company_data = {}
        if company_urls:
            try:
                company_resp = requests.get(company_urls[0], headers=HEADERS, timeout=10)
                company_resp.raise_for_status()
                company_soup = BeautifulSoup(company_resp.text, 'html.parser')
                company_data['company_name'] = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > h1').text.replace("Recruitment", "").strip() if company_soup.select_one('#wrap-comp-jobs > div.company-jobs > h1') else company_name
                company_data['company_logo'] = ['https://www.myjobmag.co.ke' + img.get('src') for img in company_soup.select('#wrap-comp-jobs > div.company-jobs > div.company-logo > img') if img.get('src') and (img.get('src').lower().endswith('.png') or img.get('src').lower().endswith('.jpg') or img.get('src').lower().endswith('.jpeg'))]
                company_data['company_industry'] = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(1) > span.comp-info-desc > a').text.strip() if company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(1) > span.comp-info-desc > a') else ""
                company_data['company_founded'] = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(2) > span.comp-info-desc').text.strip() if company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(2) > span.comp-info-desc') else ""
                company_data['company_type'] = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(3) > span.comp-info-desc').text.strip() if company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(3) > span.comp-info-desc') else ""
                company_website = ""
                website_elem = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(4) > span.comp-info-desc > a')
                if website_elem and website_elem.get('href'):
                    company_website = website_elem.get('href').strip()
                else:
                    website_text_elem = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(4) > span.comp-info-desc')
                    if website_text_elem:
                        company_website = website_text_elem.text.strip()
                excluded_domains = ['mysalaryscale.com', 'myjobmag.co.ke', 'linkedin.com', 'twitter.com', 'facebook.com']
                if company_website:
                    company_website = clean_application_url(company_website)
                    if any(domain in company_website for domain in excluded_domains) or not validate_application_method(company_website):
                        company_website = ""
                company_data['company_website'] = company_website
                company_data['company_address'] = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(5) > span.comp-info-desc').text.strip() if company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-right > ul > li:nth-child(5) > span.comp-info-desc') else ""
                company_data['company_details'] = company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-left').text.strip() if company_soup.select_one('#wrap-comp-jobs > div.company-jobs > div.company-details-left') else ""
            except RequestException as e:
                logger.error(f"Error scraping company details from {company_urls[0]}: {str(e)}")
                company_data = {
                    'company_name': company_name,
                    'company_logo': [],
                    'company_industry': '',
                    'company_founded': '',
                    'company_type': '',
                    'company_website': '',
                    'company_address': '',
                    'company_details': ''
                }
        else:
            company_data = {
                'company_name': company_name,
                'company_logo': [],
                'company_industry': '',
                'company_founded': '',
                'company_type': '',
                'company_website': '',
                'company_address': '',
                'company_details': ''
            }
        job_data = {
            "Job Title": sanitize_text(job_title_clean),
            "Company": company_name,
            "Job Type": job_type,
            "Location": job_locations,
            "Qualifications": job_qualifications,
            "Experience": job_experiences,
            "Field": job_fields,
            "Date Posted": date_posted_str,
            "Deadline": deadline,
            "Job Description": sanitize_text(job_description),
            "Application": application,
            "Company Logo": company_data.get('company_logo', [''])[0],
            "Company Website": company_data.get('company_website', ''),
            "Company Details": company_data.get('company_details', ''),
            "Job URL": job_url,
            "Job ID": hashlib.md5(job_url.encode()).hexdigest()
        }
        logger.debug(f"Scraped job details: {job_data}")
        logger.debug(f"Scraped company details: {company_data}")
        return job_data, company_data
    except RequestException as e:
        logger.error(f"Error scraping job details from {job_url}: {str(e)}")
        return None, None

def load_kenya_processed_job_ids():
    if not os.path.exists(PROCESSED_IDS_FILE):
        logger.info(f"{PROCESSED_IDS_FILE} does not exist. Initializing empty sets.")
        return set(), set(), set()
    try:
        df = pd.read_csv(PROCESSED_IDS_FILE)
        required_columns = ['Job ID', 'Job URL', 'Company Name', 'URL Page', 'Job Number']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        job_ids = set(df['Job ID'].fillna('').astype(str).tolist())
        job_urls = set(df['Job URL'].fillna('').astype(str).tolist())
        company_names = set(df['Company Name'].fillna('').astype(str).tolist())
        logger.info(f"Loaded {len(job_ids)} Job IDs, {len(job_urls)} URLs, and {len(company_names)} companies")
        return job_ids, job_urls, company_names
    except Exception as e:
        logger.error(f"Error reading {PROCESSED_IDS_FILE}: {str(e)}")
        print(f"Error reading {PROCESSED_IDS_FILE}: {str(e)}. Using empty sets.")
        return set(), set(), set()

def save_processed_job_id(job_id, job_url, company_name, url_page, job_number):
    try:
        job_id = str(job_id)
        job_url = sanitize_text(str(job_url), is_url=True)
        company_name = sanitize_text(str(company_name))
        url_page = str(url_page)
        job_number = str(job_number)
        new_row = pd.DataFrame({
            'Job ID': [job_id],
            'Job URL': [job_url],
            'Company Name': [company_name],
            'URL Page': [url_page],
            'Job Number': [job_number]
        })
        if os.path.exists(PROCESSED_IDS_FILE):
            df = pd.read_csv(PROCESSED_IDS_FILE)
            if not df.empty and job_id not in df['Job ID'].astype(str).values:
                df = pd.concat([df, new_row], ignore_index=True)
                df.to_csv(PROCESSED_IDS_FILE, index=False)
            elif df.empty:
                new_row.to_csv(PROCESSED_IDS_FILE, index=False)
        else:
            new_row.to_csv(PROCESSED_IDS_FILE, index=False)
        logger.info(f"Saved Job ID {job_id}, URL {job_url}, Company {company_name}, Page {url_page}, Job Number {job_number}")
    except Exception as e:
        logger.error(f"Error saving Job ID {job_id}: {str(e)}")
        print(f"Error saving Job ID {job_id}: {str(e)}")
        raise

def crawl_and_process():
    kenya_processed_job_ids, processed_job_urls, processed_companies = load_kenya_processed_job_ids()
    print(f"Loaded {len(kenya_processed_job_ids)} previously processed Job IDs, {len(processed_job_urls)} URLs, and {len(processed_companies)} companies")
    
    for i in range(1, 6):  # Scrape pages 1 to 5
        url = f'https://www.myjobmag.co.ke/page/{i}'
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, 'html.parser')
            job_links = ['https://www.myjobmag.co.ke' + a.get('href') for a in soup.select('li.mag-b > h2 > a') if a.get('href')]
            print(f"Collected {len(job_links)} job URLs from page {i}")
            for index, job_url in enumerate(job_links):
                job_number = index + 1
                print(f"\nProcessing job {job_number} from page {i}: {job_url}")
                if job_url in processed_job_urls:
                    print(f"Skipping job {job_number}: URL {job_url} already processed.")
                    continue
                job_data, company_data = scrape_job_details(job_url)
                if not job_data or not company_data:
                    print(f"Failed to scrape job details from {job_url}")
                    continue
                job_data['URL Page'] = str(i)
                job_data['Job Number'] = str(job_number)
                print(f"\nRaw Scraped Data for Job {job_number} (Job ID: {job_data.get('Job ID', '')})")
                print("-" * 50)
                for key, value in job_data.items():
                    print(f"{key}: {value}")
                print("-" * 50)
                job_id = str(job_data.get("Job ID", ""))
                job_title = job_data.get("Job Title", "")
                job_description = job_data.get("Job Description", "")
                application = job_data.get("Application", "")
                company_name = job_data.get("Company", "Unknown Company")
                if not job_id or pd.isna(job_id):
                    print(f"Skipping job {job_number}: Empty or invalid Job ID.")
                    continue
                if job_id in kenya_processed_job_ids:
                    print(f"Skipping job {job_number}: Job ID {job_id} already processed.")
                    continue
                if not job_title or pd.isna(job_title):
                    print(f"Skipping job {job_number}: Empty or invalid job title.")
                    continue
                if not job_description or pd.isna(job_description):
                    print(f"Skipping job {job_number}: Empty or invalid job description.")
                    continue
                if company_name == "Unknown Company" or company_name in processed_companies:
                    print(f"Skipping job {job_number}: Company {company_name} is either unknown or already processed.")
                    save_processed_job_id(job_id, job_url, company_name, i, job_number)
                    continue
                company_post_id, company_post_url = save_company_to_wordpress(index, company_data)
                if company_post_id:
                    print(f"Successfully posted company {company_name} to WordPress. Post ID: {company_post_id}, URL: {company_post_url}")
                else:
                    print(f"Failed to post company {company_name} to WordPress.")
                extracted_title = extract_job_title(job_title)
                print(f"\nParaphrasing Job Title and Description for Job ID: {job_id}")
                print("-" * 30)
                print(f"Extracted Job Title: {extracted_title}")
                combined_paraphrased, rewritten_title, rewritten_description = paraphrase_title_and_description(
                    extracted_title,
                    job_description,
                    index,
                    max_attempts=5
                )
                post_id, post_url = save_article_to_wordpress(index, job_data, rewritten_title, rewritten_description, application)
                if post_id:
                    print(f"Successfully posted job {job_number} (Job ID: {job_id}, URL: {job_url}) to WordPress. Post ID: {post_id}, URL: {post_url}")
                    save_processed_job_id(job_id, job_url, company_name, i, job_number)
                else:
                    print(f"Failed to post job {job_number} (Job ID: {job_id}, URL: {job_url}) to WordPress.")
                    save_processed_job_id(job_id, job_url, company_name, i, job_number)
                    time.sleep(10)
                if job_number % 10 == 0:
                    logger.info("Pausing for 30 seconds to avoid server overload")
                    time.sleep(30)
        except Exception as e:
            print(f"Error crawling page {url}: {str(e)}")
            logger.error(f"Error crawling page {url}: {str(e)}")
            continue

def main():
    cycle_count = 0
    while True:
        cycle_count += 1
        print(f"\nStarting cycle {cycle_count} of job processing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        try:
            crawl_and_process()  # Scrape pages 1 to 5
            print(f"Completed cycle {cycle_count}. Waiting 2 hours before restarting...")
            time.sleep(2 * 60 * 60)  # Wait 2 hours
        except Exception as e:
            logger.error(f"Error in cycle {cycle_count}: {str(e)}")
            print(f"Error in cycle {cycle_count}: {str(e)}")
            print("Retrying after 2 hours...")
            time.sleep(2 * 60 * 60)  # Wait 2 hours before retrying

if __name__ == "__main__":
    main()
