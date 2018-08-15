import unittest
import crawl_stat as cs

class CounterFunctionTestCase(unittest.TestCase):
    '''
    using https://wordcounter.net/ as the gold standard, except for characters

    in which I only counted word characters.

    document is in ./examples

    '''
    def test_word_counter(self):
        large_count = cs.count_words('./examples/should_be_3232_words.html')
        print('verifying word counter')
        self.assertEqual(large_count,3232)

    def test_sentence_counter(self):
        large_count = cs.count_sentences('./examples/should_be_3232_words.html')
        print('verifying sentence counter')
        self.assertEqual(large_count,104)

    def test_char_counter(self):
        large_count = cs.count_characters('./examples/should_be_3232_words.html')
        print('verifying chracter counter')
        self.assertEqual(large_count,16698)

if __name__ == '__main__':
    unittest.main()

