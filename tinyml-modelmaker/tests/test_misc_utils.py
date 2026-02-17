"""Tests for tinyml_modelmaker.utils.misc_utils."""

import json
import os

import pytest
import yaml

from tinyml_modelmaker.utils.config_dict import ConfigDict
from tinyml_modelmaker.utils import misc_utils


class TestAbsolutePath:
    """Tests for absolute_path and _absolute_path."""

    def test_relative_path(self):
        result = misc_utils.absolute_path('foo/bar')
        assert os.path.isabs(result)
        assert result.endswith('foo/bar') or result.endswith('foo\\bar')

    def test_none(self):
        assert misc_utils.absolute_path(None) is None

    def test_url_http(self):
        url = 'http://example.com/file.zip'
        assert misc_utils.absolute_path(url) == url

    def test_url_https(self):
        url = 'https://example.com/file.zip'
        assert misc_utils.absolute_path(url) == url

    def test_list(self):
        result = misc_utils.absolute_path(['a', 'b'])
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(os.path.isabs(p) for p in result)

    def test_tuple(self):
        result = misc_utils.absolute_path(('a', 'b'))
        assert isinstance(result, list)
        assert len(result) == 2

    def test_expanduser(self):
        result = misc_utils.absolute_path('~/foo')
        assert '~' not in result
        assert os.path.isabs(result)

    def test_already_absolute(self):
        abs_path = '/tmp/test/file.txt'
        result = misc_utils.absolute_path(abs_path)
        assert result == abs_path


class TestStr2Bool:
    """Tests for str2bool."""

    @pytest.mark.parametrize('value', ['true', 'True', 'TRUE', 'yes', 'Yes', '1'])
    def test_true_values(self, value):
        assert misc_utils.str2bool(value) is True

    @pytest.mark.parametrize('value', ['false', 'False', 'FALSE', 'no', 'No', '0', '', 'none', 'None'])
    def test_false_values(self, value):
        assert misc_utils.str2bool(value) is False

    def test_none(self):
        assert misc_utils.str2bool(None) is False

    def test_int_truthy(self):
        assert misc_utils.str2bool(1) is True

    def test_int_falsy(self):
        assert misc_utils.str2bool(0) is False


class TestIsUrl:
    """Tests for is_url."""

    def test_https_url(self):
        assert misc_utils.is_url('https://example.com/file.zip') is True

    def test_http_url(self):
        assert misc_utils.is_url('http://example.com/file.zip') is True

    def test_local_path(self):
        assert misc_utils.is_url('/path/to/file') is False

    def test_relative_path(self):
        assert misc_utils.is_url('relative/path') is False

    def test_none(self):
        assert misc_utils.is_url(None) is False

    def test_integer(self):
        assert misc_utils.is_url(42) is False


class TestSimplifyDict:
    """Tests for simplify_dict."""

    def test_configdict_to_dict(self):
        cfg = ConfigDict({'a': 1, 'b': {'c': 2}})
        result = misc_utils.simplify_dict(cfg)
        assert isinstance(result, dict)
        assert not isinstance(result, ConfigDict)
        assert isinstance(result['b'], dict)
        assert not isinstance(result['b'], ConfigDict)

    def test_tuples_to_lists(self):
        cfg = ConfigDict({'items': (1, 2, 3)})
        result = misc_utils.simplify_dict(cfg)
        assert result['items'] == [1, 2, 3]
        assert isinstance(result['items'], list)

    def test_preserves_values(self):
        cfg = ConfigDict({'a': 1, 'b': 'hello', 'c': True})
        result = misc_utils.simplify_dict(cfg)
        assert result == {'a': 1, 'b': 'hello', 'c': True}


class TestWriteDict:
    """Tests for write_dict."""

    def test_writes_json(self, tmp_path):
        data = ConfigDict({'key': 'value', 'num': 42})
        filepath = str(tmp_path / 'output.txt')
        misc_utils.write_dict(data, filepath, write_json=True, write_yaml=False)

        json_path = str(tmp_path / 'output.json')
        assert os.path.exists(json_path)
        with open(json_path) as fp:
            loaded = json.load(fp)
        assert loaded['key'] == 'value'
        assert loaded['num'] == 42

    def test_writes_yaml(self, tmp_path):
        data = ConfigDict({'key': 'value', 'num': 42})
        filepath = str(tmp_path / 'output.txt')
        misc_utils.write_dict(data, filepath, write_json=False, write_yaml=True)

        yaml_path = str(tmp_path / 'output.yaml')
        assert os.path.exists(yaml_path)
        with open(yaml_path) as fp:
            loaded = yaml.safe_load(fp)
        assert loaded['key'] == 'value'
        assert loaded['num'] == 42

    def test_writes_both(self, tmp_path):
        data = ConfigDict({'key': 'value'})
        filepath = str(tmp_path / 'output.txt')
        misc_utils.write_dict(data, filepath)

        assert os.path.exists(str(tmp_path / 'output.json'))
        assert os.path.exists(str(tmp_path / 'output.yaml'))


class TestDeepUpdateDict:
    """Tests for deep_update_dict."""

    def test_overwrites_scalar(self):
        d1 = {'a': 1}
        d2 = {'a': 2}
        result = misc_utils.deep_update_dict(d1, d2)
        assert result['a'] == 2

    def test_merges_nested(self):
        d1 = {'a': {'b': 1, 'c': 2}}
        d2 = {'a': {'b': 10, 'd': 4}}
        result = misc_utils.deep_update_dict(d1, d2)
        assert result['a']['b'] == 10  # overwritten
        assert result['a']['c'] == 2   # preserved
        assert result['a']['d'] == 4   # added

    def test_adds_new_keys(self):
        d1 = {'a': 1}
        d2 = {'b': 2}
        result = misc_utils.deep_update_dict(d1, d2)
        assert result['a'] == 1
        assert result['b'] == 2

    def test_returns_first_dict(self):
        d1 = {'a': 1}
        d2 = {'b': 2}
        result = misc_utils.deep_update_dict(d1, d2)
        assert result is d1
