import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "models"))
sys.path.insert(0, str(ROOT / "src" / "evaluation"))
from url_features import FEATURE_NAMES, extract_urls, message_features, featurize

I = FEATURE_NAMES.index


def test_no_url_is_all_zero():
    v, n = message_features("Hello, no link here.")
    assert n == 0 and all(x == 0 for x in v)


def test_trailing_punctuation_removed():
    assert extract_urls("visit http://a.com/x.") == ["http://a.com/x"]


def test_ip_at_punycode():
    v, _ = message_features("http://192.168.1.1/login")
    assert v[I("any_ip")] == 1
    v, _ = message_features("http://user@evil.com/")
    assert v[I("any_at")] == 1
    v, _ = message_features("http://xn--80ak6aa92e.com/")
    assert v[I("any_punycode")] == 1


def test_defang_makes_url_invisible():
    from experiment38_distilbert_pilot import url_defang
    t = "Please go to https://secure-bank.example.com/login now."
    assert message_features(t)[1] == 1
    assert message_features(url_defang(t))[1] == 0   # model is blinded by representation


def test_text_attacks_do_not_change_url_features():
    from experiment38_distilbert_pilot import lexical_proxy, neutral_hidden_html
    t = "Click here to verify http://a.example.com/x immediately"
    base = featurize([t])
    assert (featurize([lexical_proxy(t)]) == base).all()
    assert (featurize([neutral_hidden_html(t)]) == base).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("OK", name)