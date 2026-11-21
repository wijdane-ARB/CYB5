"""Offline smoke tests for the documented fixed attack transformations."""
import unittest
from src.evaluation.experiment38_distilbert_pilot import (
    lexical_proxy, neutral_hidden_html, url_defang, URL_RE
)

class Experiment38AttackTests(unittest.TestCase):
    def test_lexical_proxy_only_matches_known_tokens(self):
        self.assertEqual(lexical_proxy('Please VERIFY immediately.'),
                         'Please confirm promptly.')
        self.assertEqual(lexical_proxy('Nothing to modify.'), 'Nothing to modify.')

    def test_hidden_comment_is_neutral_and_appended(self):
        original = 'Example email.'
        modified = neutral_hidden_html(original)
        self.assertTrue(modified.startswith(original))
        self.assertIn('<!--', modified)

    def test_defang_does_not_fabricate_url(self):
        self.assertEqual(url_defang('No URL here.'), 'No URL here.')

    def test_defang_urls_and_preserve_sentence_period(self):
        transformed = url_defang('See https://example.com/login. Thanks.')
        self.assertIn('hxxps://example[.]com/login.', transformed)
        self.assertTrue(transformed.endswith(' Thanks.'))

    def test_url_detection(self):
        self.assertTrue(URL_RE.search('Go to http://example.com'))
        self.assertFalse(URL_RE.search('No URL in this message'))

if __name__ == '__main__':
    unittest.main()
