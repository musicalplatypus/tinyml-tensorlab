"""Tests for tinyml_modelmaker.utils.config_dict.ConfigDict."""

import os

import pytest
import yaml

from tinyml_modelmaker.utils.config_dict import ConfigDict


class TestConfigDictInit:
    """Tests for ConfigDict initialization."""

    def test_init_from_dict(self, sample_config_dict):
        cfg = ConfigDict(sample_config_dict)
        assert cfg.a == 1
        assert cfg.b.c == 2
        assert cfg.b.d == 'hello'
        assert cfg.e == [1, 2, 3]
        assert cfg.f is True

    def test_init_from_yaml(self, tmp_yaml_file, sample_config_dict):
        cfg = ConfigDict(tmp_yaml_file)
        assert cfg.a == sample_config_dict['a']
        assert cfg.b.c == sample_config_dict['b']['c']
        assert cfg.b.d == sample_config_dict['b']['d']

    def test_init_from_none(self):
        cfg = ConfigDict(None)
        assert len(cfg) == 0

    def test_init_empty(self):
        cfg = ConfigDict()
        assert len(cfg) == 0

    def test_init_from_kwargs(self):
        cfg = ConfigDict(None, key1='val1', key2=42)
        assert cfg.key1 == 'val1'
        assert cfg.key2 == 42

    def test_init_merging_order(self):
        """kwargs override args which override the input dict."""
        base = {'x': 1, 'y': 2}
        override = {'x': 10, 'z': 30}
        cfg = ConfigDict(base, override, y=200)
        assert cfg.x == 10   # override from args
        assert cfg.y == 200  # override from kwargs
        assert cfg.z == 30   # added by args

    def test_init_invalid_input(self):
        with pytest.raises(TypeError):
            ConfigDict(42)

    def test_init_non_yaml_file(self, tmp_path):
        txt_file = tmp_path / 'config.txt'
        txt_file.write_text('key: value')
        with pytest.raises(ValueError):
            ConfigDict(str(txt_file))


class TestConfigDictAccess:
    """Tests for attribute and key access."""

    def test_attribute_access(self):
        cfg = ConfigDict({'key': 'value'})
        assert cfg.key == 'value'
        assert cfg['key'] == 'value'

    def test_attribute_error(self):
        cfg = ConfigDict({'key': 'value'})
        with pytest.raises(AttributeError):
            _ = cfg.nonexistent

    def test_nested_dicts_become_configdict(self):
        cfg = ConfigDict({'outer': {'inner': 42}})
        assert isinstance(cfg.outer, ConfigDict)
        assert cfg.outer.inner == 42

    def test_setattr(self):
        cfg = ConfigDict()
        cfg.new_key = 'new_value'
        assert cfg['new_key'] == 'new_value'


class TestConfigDictUpdate:
    """Tests for the update method."""

    def test_update_deep_merge(self):
        cfg = ConfigDict({'b': {'c': 2, 'existing': 'keep'}})
        cfg.update({'b': {'d': 3}})
        assert cfg.b.c == 2          # preserved
        assert cfg.b.d == 3          # added
        assert cfg.b.existing == 'keep'  # preserved

    def test_update_overwrites_non_dict(self):
        cfg = ConfigDict({'a': 1})
        cfg.update({'a': 99})
        assert cfg.a == 99

    def test_update_adds_new_keys(self):
        cfg = ConfigDict({'a': 1})
        cfg.update({'b': 2})
        assert cfg.b == 2

    def test_update_returns_self(self):
        cfg = ConfigDict({'a': 1})
        result = cfg.update({'b': 2})
        assert result is cfg


class TestConfigDictIncludeFiles:
    """Tests for YAML include file functionality."""

    def test_include_files(self, tmp_path):
        # Create the included file
        include_file = tmp_path / 'extra.yaml'
        with open(include_file, 'w') as fp:
            yaml.safe_dump({'extra_key': 'extra_value', 'number': 42}, fp)

        # Create the main file that references it
        main_file = tmp_path / 'main.yaml'
        with open(main_file, 'w') as fp:
            yaml.safe_dump({
                'include_files': ['extra.yaml'],
                'main_key': 'main_value',
            }, fp)

        cfg = ConfigDict(str(main_file))
        assert cfg.main_key == 'main_value'
        assert cfg.extra_key == 'extra_value'
        assert cfg.number == 42
