import math

import neuralcoref
import torch
import coreferee, spacy
from newspaper import ArticleException
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from graph import KB
from information import get_article


# the tokenizer transforms input text to an array of tokens, ready to be processed
tokenizer = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
# load the actual model
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")
# English transformer pipeline (roberta-base)
nlp = spacy.load('en_core_web_sm')
# add coreference resolution to our pipeline
neuralcoref.add_to_pipe(nlp)


def extract_relations_from_model_output(text):
    # function copied from the authors of the model

    relations = []
    relation, subject, relation, object_ = '', '', '', ''
    current = 'x'
    # eliminate residual html tags after stripping away the whitespaces
    text = text.strip().replace("<s>", "").replace("<pad>", "").replace("</s>", "")
    for token in text.split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                relations.append({
                    'head': subject.strip(),
                    'type': relation.strip(),
                    'tail': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        relations.append({
            'head': subject.strip(),
            'type': relation.strip(),
            'tail': object_.strip()
        })
    return relations


def from_text_to_kb(text, article_url, span_length=128, article_title=None, article_publish_date=None):

    doc = nlp(text)

    print("--- text: ---")
    print(text)
    print("--- chains: ---")
    doc._.coref_chains.print()
    print("--- clusters: ---")
    print(doc._.coref_clusters)
    print("--- resolved: ---")
    print(doc._.coref_resolved)
    # tokenize whole text to be processed
    inputs = tokenizer([text], return_tensors="pt", truncation=True)

    # compute span boundaries
    num_tokens = len(inputs["input_ids"][0])
    num_spans = math.ceil(num_tokens / span_length)
    # distributes the exceeding characters into overlaps between spans
    overlap = math.ceil((num_spans * span_length - num_tokens) / max(num_spans - 1, 1))
    spans_boundaries = []

    # offsets spans' start by the cumulative effect of overlaps
    offset = 0
    for i in range(num_spans):
        spans_boundaries.append([offset + span_length * i, offset + span_length * (i + 1)])
        offset -= overlap

    # takes the first tensor (since we analyze one chunk of text at a time) of input_ids and crops the span
    tensor_ids = [inputs["input_ids"][0][boundary[0]:boundary[1]] for boundary in spans_boundaries]
    # similarly crops the span of the attention mask
    tensor_masks = [inputs["attention_mask"][0][boundary[0]:boundary[1]] for boundary in spans_boundaries]
    # the number of sequences that we want to be returned
    num_return_sequences = 3
    # bundles together the current span's input_ids and attention mask, together with other parameters
    inputs = {
        "input_ids": torch.stack(tensor_ids),
        "attention_mask": torch.stack(tensor_masks),
        "max_length": 256,
        "length_penalty": 0,
        "num_beams": 3,
        "num_return_sequences": num_return_sequences
    }
    # generate all the relationships from the various spans
    generated_tokens = model.generate(**inputs)

    # decode relations
    decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

    # create kb
    kb = KB()
    for i, sentence_pred in enumerate(decoded_preds):
        # since we independently generate num_return_sequences for each span
        current_span_index = i // num_return_sequences

        # structures the relation from a textual output
        relations = extract_relations_from_model_output(sentence_pred)
        for relation in relations:
            relation["meta"] = {
                "url": article_url,
                "spans": [spans_boundaries[current_span_index]]
            }
            kb.add_relation(relation, article_title, article_publish_date)

    return kb


def from_url_to_kb(url):
    # get parsed article
    article = get_article(url)

    config = {
        "article_title": article.title,
        "article_publish_date": article.publish_date
    }
    kb = from_text_to_kb(article.text, article.url, **config)
    return kb


def from_urls_to_kb(urls):
    kb = KB()

    for count, url in enumerate(urls):
        print(f"Visiting {url} [{count+1}/{len(urls)}]")
        try:
            kb_url = from_url_to_kb(url)
            kb.merge_with_kb(kb_url)
        except ArticleException:
            print(f"  Couldn't download article at url {url}")

    return kb
