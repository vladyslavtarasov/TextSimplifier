import nltk
import spacy
import pyinflect
import os
import re
import torch
import torch.nn.functional as func
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
from nltk.corpus import brown, wordnet, gutenberg, webtext
from nltk.tokenize import sent_tokenize, word_tokenize

nltk_data_path = os.path.join(os.path.expanduser('~'), 'nltk_data')
nltk.data.path.append(nltk_data_path)

combined_words = brown.words() + webtext.words()
word_freqs = nltk.FreqDist(w.lower() for w in combined_words)

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
bart_model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

spacy_nlp = spacy.load("en_core_web_sm")


# Calculates the difficulty of a word based on its frequency in the combined corpus.
# Lower frequency results in higher difficulty. The '+1' in the denominator prevents division by zero.
def get_word_difficulty(word):
    freq = word_freqs[word.lower()]
    difficulty = 1 / (freq + 1)
    return difficulty


# Tokenizes the input sentence, obtains BERT embeddings, and locates the embedding for the specified word.
# If the word is split into subtokens, their embeddings are averaged to obtain a single vector representation.
def get_word_embedding(sentence, word):
    inputs = bert_tokenizer(sentence, return_tensors="pt")
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state

    word_tokens = bert_tokenizer.tokenize(word)
    word_token_ids = bert_tokenizer.convert_tokens_to_ids(word_tokens)

    word_indexes = [i for i, token_id in enumerate(inputs['input_ids'][0].tolist()) if token_id == word_token_ids[0]]

    if word_indexes:
        word_embedding = embeddings[0, word_indexes[0]]
        if len(word_token_ids) > 1:
            all_embeddings = embeddings[0, word_indexes[0]:word_indexes[0]+len(word_token_ids)]
            word_embedding = torch.mean(all_embeddings, dim=0)
        return word_embedding
    else:
        return None


# Retrieves synonyms for the given word using WordNet, ensuring that the synonyms are different from the original word.
def get_word_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word:
                synonyms.add(synonym)
    return synonyms


# For a given difficult word in a sentence, this function computes the cosine similarity
# between the embedding of the original word and each of its synonyms within the context of the sentence.
# This helps in identifying synonyms that are contextually similar to the original word.
def get_synonyms_similarities(original_sentence, difficult_word, synonyms):
    original_embedding = get_word_embedding(original_sentence, difficult_word)
    synonym_similarities = []

    for synonym in synonyms:
        modified_sentence = original_sentence.replace(difficult_word, synonym)
        synonym_embedding = get_word_embedding(modified_sentence, synonym)
        similarity = func.cosine_similarity(original_embedding.unsqueeze(0), synonym_embedding.unsqueeze(0)).item()
        synonym_similarities.append((synonym, similarity))

    synonym_similarities = sorted(synonym_similarities, key=lambda item: item[1], reverse=True)

    return synonym_similarities


# Splits the input text into smaller subtexts that do not exceed the max_token_count.
# This is necessary because language models have token limits on the input they can process.
# The function ensures that the text is split at sentence boundaries to maintain coherence.
def split_text_into_subtexts(text, max_token_count, tokenizer):
    sentences = sent_tokenize(text)

    subtexts = []
    current_subtext = []
    current_token_count = 0

    for sentence in sentences:
        tokens = tokenizer.tokenize(sentence)
        token_count = len(tokens)

        if token_count + current_token_count <= max_token_count:
            current_subtext.append(sentence)
            current_token_count += token_count
        else:
            if current_subtext:
                subtexts.append(' '.join(current_subtext))
                current_subtext = []
                current_token_count = 0

            if token_count > max_token_count:
                subtexts.extend(split_long_sentence(tokens, max_token_count, tokenizer))
            else:
                current_subtext.append(sentence)
                current_token_count = token_count

    if current_subtext:
        subtexts.append(' '.join(current_subtext))

    return subtexts


# Splits a sentence represented as tokens into smaller segments, each not exceeding the max_token_count.
def split_long_sentence(tokens, max_token_count, tokenizer):
    sub_sentences = []
    current_sub_sentence_tokens = []
    current_token_count = 0

    for token in tokens:
        current_token_count += 1

        if current_token_count <= max_token_count:
            current_sub_sentence_tokens.append(token)
        else:
            sub_sentences.append(tokenizer.convert_tokens_to_string(current_sub_sentence_tokens))
            current_sub_sentence_tokens = [token]
            current_token_count = 1

    if current_sub_sentence_tokens:
        sub_sentences.append(tokenizer.convert_tokens_to_string(current_sub_sentence_tokens))

    return sub_sentences


# Checks if the word meets certain criteria (length, pattern) to be considered valid for processing.
def is_valid_word(word):
    contraction_pattern = re.compile(r"\b\w+('(s|re|ve|d|m|ll|t))?")
    return len(word) > 1 and re.match(contraction_pattern, word) is not None and word.isalpha()


# Identifies words in the text that exceed a specified difficulty threshold.
def get_difficult_words(text, threshold):
    words = word_tokenize(text)
    difficult_words = []
    for word in words:
        if not is_valid_word(word):
            continue
        difficulty = get_word_difficulty(word)
        if difficulty >= threshold:
            difficult_words.append(word)

    return difficult_words


# Splits the input text into manageable subtexts and simplifies each subtext by identifying and replacing difficult words.
# Returns the simplified version of the entire text with word mappings and additional synonym suggestions.
def get_text_simplification(text, original_threshold, new_threshold):
    subtexts = split_text_into_subtexts(text, 510, bert_tokenizer)
    simplified_text = ""
    overall_mapping = {}
    additional_synonyms = {}

    for subtext in subtexts:
        processed_subtext, mapping, additional_synonyms = simplify_subtext(subtext, original_threshold, new_threshold)
        simplified_text += processed_subtext + " "
        overall_mapping.update(mapping)

    return simplified_text.strip(), overall_mapping, additional_synonyms


# Splits the input text into manageable subtexts suitable for the model if needed and summarizes the text.
# Returns the summarized version of the entire text.
def get_text_summarization(text, summarize_coefficient):
    subtexts = split_text_into_subtexts(text, 900, bart_tokenizer)
    summarized_text = ""
    for subtext in subtexts:
        processed_subtext = summarize_subtext(subtext, summarize_coefficient)
        summarized_text += processed_subtext + " "

    return summarized_text.strip()


# Simplifies a segment of text by identifying difficult words, finding easier synonyms,
# and replacing the difficult words with their easier synonyms based on the new threshold.
def simplify_subtext(subtext, original_threshold, new_threshold):
    difficult_words = get_difficult_words(subtext, original_threshold)
    word_replacements = {}
    additional_synonyms = {}

    for word in difficult_words:
        synonyms = get_word_synonyms(word)

        if word.istitle():
            synonyms = [capitalize_first_letter(syn) for syn in synonyms]

        adjusted_synonyms = [match_grammar(subtext, word, syn) for syn in synonyms]
        adjusted_synonyms = filter_original_word(adjusted_synonyms, word, subtext)
        additional_synonyms[word] = adjusted_synonyms

        easier_synonyms = [syn for syn in adjusted_synonyms if get_word_difficulty(syn) <= new_threshold]

        if len(easier_synonyms) == 0:
            continue

        synonyms_similarities = get_synonyms_similarities(subtext, word, easier_synonyms)
        best_synonym = synonyms_similarities[0][0]
        subtext = subtext.replace(word, best_synonym, 1)
        additional_synonyms[word] = [syn for syn in adjusted_synonyms if syn != best_synonym]
        word_replacements[word] = best_synonym

    return subtext, word_replacements, additional_synonyms


# Summarizes a segment of text using the BART model, adjusting the length
# of the summary based on the provided coefficient.
def summarize_subtext(subtext, summarize_coefficient):
    inputs = bart_tokenizer(subtext, return_tensors='pt', truncation=True)
    input_length = inputs['input_ids'].shape[1]
    max_length = int(input_length * summarize_coefficient) + 20
    min_length = max(1, int(max_length * 0.5))

    eos_token_id = bart_tokenizer.encode('.', add_special_tokens=False)[0]

    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, min_length=min_length,
                                      max_length=max_length, early_stopping=True, eos_token_id=eos_token_id)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    summary = summary.rstrip()

    if not summary.endswith(('.', '!', '?')):
        summary = summary + '...'

    return summary


# Capitalises the first letter of the given word.
def capitalize_first_letter(word):
    if not word:
        return word
    return word[0].upper() + word[1:]


# Adjusts the grammatical form of the replacement word to match that of the original word in the sentence,
# using part-of-speech tags and inflection.
def match_grammar(sentence, original_word, replacement_word):
    doc = spacy_nlp(sentence)
    replacement_token = spacy_nlp(replacement_word)[0]
    modified_word = replacement_word

    for token in doc:
        if token.text.lower() == original_word.lower():
            if token.pos_ == "VERB":
                inflected_form = replacement_token._.inflect(token.tag_)
                if inflected_form:
                    modified_word = inflected_form

            elif token.pos_ == "NOUN" and token.tag_ in ["NNS", "NNPS"]:
                inflected_form = replacement_token._.inflect("NNS")
                if inflected_form:
                    modified_word = inflected_form

            elif token.pos_ == "ADJ":
                if token.tag_ == "JJR":
                    inflected_form = replacement_token._.inflect("JJR")
                    if inflected_form:
                        modified_word = inflected_form
                elif token.tag_ == "JJS":
                    inflected_form = replacement_token._.inflect("JJS")
                    if inflected_form:
                        modified_word = inflected_form

            break

    return modified_word


# Filters out synonyms that are too similar to the original word,
# to avoid replacing a word with itself or a close variant.
def filter_original_word(synonyms, original_word, sentence):
    doc = spacy_nlp(sentence)

    original_lemmas = [token.lemma_.lower() for token in doc if token.text.lower() == original_word.lower()]

    filtered_synonyms = []
    for syn in synonyms:
        syn_doc = spacy_nlp(syn)
        syn_lemmas = [token.lemma_.lower() for token in syn_doc]

        if not set(syn_lemmas) == set(original_lemmas):
            filtered_synonyms.append(syn)

    return filtered_synonyms
