import unittest
from simplifier import *


class TestGetWordDifficulty(unittest.TestCase):
    def test_known_word(self):
        self.assertLess(get_word_difficulty('the'), 0.01)

    def test_unknown_word(self):
        self.assertEqual(get_word_difficulty('xyzunknown'), 1.0)


class TestGetWordEmbedding(unittest.TestCase):
    def test_embed_known_word(self):
        sentence = "Test sentence with the word."
        word = "sentence"
        result = get_word_embedding(sentence, word)
        self.assertIsNotNone(result)


class TestGetWordSynonyms(unittest.TestCase):
    def test_synonyms_exist(self):
        word = "happy"
        synonyms = get_word_synonyms(word)
        self.assertIn("glad", synonyms)


class TestGetSynonymsSimilarities(unittest.TestCase):
    def test_returns_non_empty_for_known_inputs(self):
        original_sentence = "The quick brown fox jumps over the lazy dog."
        difficult_word = "quick"
        synonyms = {"fast", "swift", "speedy"}

        similarities = get_synonyms_similarities(original_sentence, difficult_word, synonyms)

        self.assertTrue(len(similarities) > 0,
                        "The function should return a non-empty list of synonyms and their similarities.")

        for item in similarities:
            self.assertIsInstance(item, tuple, "Each item in the list should be a tuple.")
            self.assertEqual(len(item), 2, "Each tuple should contain two elements.")
            self.assertIsInstance(item[0], str, "The first element of the tuple should be a string (synonym).")
            self.assertIsInstance(item[1], float, "The second element of the tuple should be a float (similarity).")

        returned_synonyms = {item[0] for item in similarities}
        self.assertTrue(returned_synonyms.issubset(synonyms),
                        "Returned synonyms should be a subset of the expected synonyms.")


class TestSplitTextIntoSubtexts(unittest.TestCase):
    def test_split_simple_text(self):
        text = "This is a test. This is only a test."
        subtexts = split_text_into_subtexts(text, 5, bert_tokenizer)  # Assuming 5 tokens max for simplicity
        self.assertEqual(len(subtexts), 3)


class TestIsValidWord(unittest.TestCase):
    def test_valid_word(self):
        self.assertTrue(is_valid_word("hello"))

    def test_invalid_word(self):
        self.assertFalse(is_valid_word("123"))


class TestGetDifficultWords(unittest.TestCase):
    def test_with_difficult_words(self):
        text = "This uses some complex terminology."
        difficult_words = get_difficult_words(text, 0.1)
        self.assertIn("terminology", difficult_words)


class TestTextSimplificationAndSummarization(unittest.TestCase):
    def test_text_simplification(self):
        text = "The cat ambulated deftly, evading the quixotic, luminescent orbs."
        simplified_text, _, _ = get_text_simplification(text, 0.1, 0.2)
        self.assertNotEqual(text, simplified_text)

    def test_text_summarization(self):
        text = ("This is a long text that needs to be summarized. "
                "In order for the function to work properly, the text should be long enough.")
        summary = get_text_summarization(text, 0.5)
        self.assertNotEqual(text, summary)


class TestCapitalizeFirstLetter(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(capitalize_first_letter(""), "")

    def test_already_capitalized(self):
        self.assertEqual(capitalize_first_letter("Python"), "Python")

    def test_lowercase_word(self):
        self.assertEqual(capitalize_first_letter("python"), "Python")

    def test_non_alpha_characters(self):
        self.assertEqual(capitalize_first_letter("123abc"), "123abc")

    def test_null_input(self):
        self.assertIsNone(capitalize_first_letter(None))


class TestMatchGrammar(unittest.TestCase):
    def test_verb_conjugation(self):
        sentence = "The cat sits on the mat."
        self.assertEqual(match_grammar(sentence, "sits", "lie"), "lies")

    def test_noun_pluralization(self):
        sentence = "There are two cats."
        self.assertEqual(match_grammar(sentence, "cats", "dog"), "dogs")

    def test_adjective_comparative(self):
        sentence = "This puzzle is harder than the other."
        self.assertEqual(match_grammar(sentence, "harder", "easy"), "easier")

    def test_adjective_superlative(self):
        sentence = "This is the highest point."
        self.assertEqual(match_grammar(sentence, "highest", "low"), "lowest")


class TestFilterOriginalWord(unittest.TestCase):
    def test_filtering(self):
        synonyms = {"cats", "felines", "kittens"}
        self.assertCountEqual(filter_original_word(synonyms, "cat", "The cat sat on the mat."), {"felines", "kittens"})

    def test_no_filtering_needed(self):
        synonyms = {"run", "sprint", "jog"}
        self.assertCountEqual(filter_original_word(synonyms, "run", "I will run to the store."), {"sprint", "jog"})

    def test_filtering_with_no_matches(self):
        synonyms = {"puppies", "dogs"}
        self.assertCountEqual(filter_original_word(synonyms, "cat", "The cat sat on the mat."), {"puppies", "dogs"})


if __name__ == '__main__':
    unittest.main()