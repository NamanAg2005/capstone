from app.static_check import is_safe, check


def test_safe_pandas_code_passes():
    code = "result = df['x'].sum()\nprint(result)\n"
    assert is_safe(code)


def test_disallowed_import_rejected():
    assert not is_safe("import os\nresult = os.listdir('.')\n")


def test_disallowed_open_rejected():
    assert not is_safe("f = open('secret.txt')\nresult = f.read()\n")


def test_disallowed_eval_rejected():
    assert not is_safe("result = eval('1+1')\n")


def test_dunder_attribute_rejected():
    assert not is_safe("result = ().__class__.__bases__\n")


def test_syntax_error_reported():
    violations = check("def f(:\n")
    assert violations and "syntax error" in violations[0]
