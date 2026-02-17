"""Tests for tinyml_modelmaker.ai_modules.common.datasets.dataset_utils."""

import json
import os

import pytest

from tinyml_modelmaker.ai_modules.common.datasets import dataset_utils


class TestCreateFilelist:
    """Tests for create_filelist."""

    def test_creates_filelist(self, tmp_data_dir, tmp_path):
        output_dir = str(tmp_path / 'annotations')
        result = dataset_utils.create_filelist(
            tmp_data_dir, output_dir, ignore_str_list=[r'\.md$', r'LICENSE']
        )
        assert os.path.exists(result)
        with open(result) as fp:
            files = [line.strip() for line in fp.readlines() if line.strip()]
        # Should find 6 CSV files (3 per class, 2 classes)
        assert len(files) == 6
        # Each file should include the class directory prefix
        assert any('classA' in f for f in files)
        assert any('classB' in f for f in files)

    def test_ignore_pattern(self, tmp_path):
        """Files matching ignore patterns are excluded."""
        data_dir = tmp_path / 'data'
        data_dir.mkdir()
        (data_dir / 'good.csv').write_text('data')
        (data_dir / 'README.md').write_text('readme')
        (data_dir / 'LICENSE').write_text('license')

        output_dir = str(tmp_path / 'out')
        result = dataset_utils.create_filelist(
            str(data_dir), output_dir, ignore_str_list=[r'\.md$', r'LICENSE']
        )
        with open(result) as fp:
            files = [line.strip() for line in fp.readlines() if line.strip()]
        assert len(files) == 1
        assert 'good.csv' in files[0]


class TestCreateInterFileSplit:
    """Tests for create_inter_file_split."""

    def _setup_filelist(self, tmp_path, tmp_data_dir):
        """Helper to create a file list and split output paths."""
        annotations_dir = tmp_path / 'annotations'
        annotations_dir.mkdir(exist_ok=True)

        # Create the file list
        file_list_path = dataset_utils.create_filelist(
            tmp_data_dir, str(annotations_dir), ignore_str_list=[]
        )

        split_files = [
            str(annotations_dir / 'train_list.txt'),
            str(annotations_dir / 'val_list.txt'),
            str(annotations_dir / 'test_list.txt'),
        ]
        return file_list_path, split_files

    def test_split_with_float_factor_bug(self, tmp_path, tmp_data_dir):
        """Passing a bare float as split_factor triggers a bug (len() on float).

        This documents an existing bug in create_inter_file_split at line 107
        where it calls len(split_factor) instead of len(split_factors). In practice,
        the codebase always passes tuples like (0.6, 0.3, 0.1).
        """
        file_list_path, split_files = self._setup_filelist(tmp_path, tmp_data_dir)
        with pytest.raises(TypeError, match="object of type 'float' has no len"):
            dataset_utils.create_inter_file_split(
                file_list_path, split_files, split_factor=0.6, shuffle_items=False, random_seed=42
            )

    def test_split_with_tuple_factor(self, tmp_path, tmp_data_dir):
        """Tuple split_factor works correctly (this is the production usage)."""
        file_list_path, split_files = self._setup_filelist(tmp_path, tmp_data_dir)
        dataset_utils.create_inter_file_split(
            file_list_path, split_files, split_factor=(0.6, 0.2, 0.2), shuffle_items=False, random_seed=42
        )
        # All split files should exist
        for sf in split_files:
            assert os.path.exists(sf), f'{sf} should exist'

        # Total files across all splits should equal original count
        total = 0
        for sf in split_files:
            with open(sf) as fp:
                total += len([l for l in fp.readlines() if l.strip()])
        assert total == 6

    def test_split_with_list_factor(self, tmp_path, tmp_data_dir):
        file_list_path, split_files = self._setup_filelist(tmp_path, tmp_data_dir)
        dataset_utils.create_inter_file_split(
            file_list_path, split_files, split_factor=[0.6, 0.2, 0.2], shuffle_items=False, random_seed=42
        )
        total = 0
        for sf in split_files:
            with open(sf) as fp:
                total += len([l for l in fp.readlines() if l.strip()])
        assert total == 6

    def test_deterministic_with_seed(self, tmp_path, tmp_data_dir):
        """Same seed produces identical splits."""
        file_list_path, split_files1 = self._setup_filelist(tmp_path, tmp_data_dir)

        dataset_utils.create_inter_file_split(
            file_list_path, split_files1, split_factor=(0.6, 0.2, 0.2),
            shuffle_items=True, random_seed=42
        )
        contents1 = []
        for sf in split_files1:
            with open(sf) as fp:
                contents1.append(fp.read())

        # Create second set of splits with same seed
        annotations_dir2 = tmp_path / 'annotations2'
        annotations_dir2.mkdir()
        split_files2 = [
            str(annotations_dir2 / 'train_list.txt'),
            str(annotations_dir2 / 'val_list.txt'),
            str(annotations_dir2 / 'test_list.txt'),
        ]
        dataset_utils.create_inter_file_split(
            file_list_path, split_files2, split_factor=(0.6, 0.2, 0.2),
            shuffle_items=True, random_seed=42
        )
        contents2 = []
        for sf in split_files2:
            with open(sf) as fp:
                contents2.append(fp.read())

        assert contents1 == contents2


class TestCategoryUtilities:
    """Tests for category helper functions."""

    @pytest.fixture
    def categories(self):
        return [
            {'id': 1, 'name': 'cat', 'supercategory': 'animal'},
            {'id': 2, 'name': 'dog', 'supercategory': 'animal'},
            {'id': 3, 'name': 'bird', 'supercategory': 'animal'},
        ]

    def test_get_category_names(self, categories):
        names = dataset_utils.get_category_names(categories)
        assert names == ['cat', 'dog', 'bird']

    def test_get_category_ids(self, categories):
        ids = dataset_utils.get_category_ids(categories)
        assert ids == [1, 2, 3]

    def test_get_category_entry_found(self, categories):
        entry = dataset_utils.get_category_entry(categories, 'dog')
        assert entry['id'] == 2
        assert entry['name'] == 'dog'

    def test_get_category_entry_missing(self, categories):
        entry = dataset_utils.get_category_entry(categories, 'fish')
        assert entry is None

    def test_get_new_id_fills_gap(self):
        assert dataset_utils.get_new_id([1, 3]) == 2

    def test_get_new_id_empty(self):
        assert dataset_utils.get_new_id([]) == 1

    def test_get_new_id_contiguous(self):
        assert dataset_utils.get_new_id([1, 2, 3]) == 4

    def test_get_new_category_id(self, categories):
        new_id = dataset_utils.get_new_category_id(categories)
        assert new_id == 4  # next after 1, 2, 3

    def test_add_missing_categories(self):
        cats = [
            {'id': 1, 'name': 'cat', 'supercategory': 'animal'},
            {'id': 3, 'name': 'bird', 'supercategory': 'animal'},
        ]
        result = dataset_utils.add_missing_categories(cats)
        assert len(result) == 3
        ids = [c['id'] for c in result]
        assert ids == [1, 2, 3]
        # The missing one should have a generated name
        assert result[1]['name'].startswith('undefined')

    def test_add_missing_categories_empty(self):
        result = dataset_utils.add_missing_categories([])
        assert result == []


class TestColorUtilities:
    """Tests for color table generation."""

    def test_get_color_table_returns_correct_count(self):
        colors = dataset_utils.get_color_table(10)
        assert len(colors) == 10

    def test_get_color_table_rgb_tuples(self):
        colors = dataset_utils.get_color_table(5)
        for color in colors:
            assert len(color) == 3
            assert all(isinstance(c, (int, float)) for c in color)

    def test_get_color_palette_256_entries(self):
        palette = dataset_utils.get_color_palette(10)
        assert len(palette) == 256

    def test_get_color_palette_single_class(self):
        palette = dataset_utils.get_color_palette(1)
        assert len(palette) == 256


class TestFileUtilities:
    """Tests for file path utility functions."""

    def test_get_file_list(self, tmp_path):
        (tmp_path / 'file1.csv').write_text('data')
        (tmp_path / 'file2.csv').write_text('data')
        files = dataset_utils.get_file_list(str(tmp_path))
        assert len(files) == 2

    def test_get_file_name_partial(self):
        result = dataset_utils.get_file_name_partial(
            '/projects/data/classA/file1.csv', '/projects/data'
        )
        assert result == 'classA/file1.csv' or result == 'classA\\file1.csv'

    def test_get_file_name_partial_none(self):
        result = dataset_utils.get_file_name_partial(None, '/projects/data')
        assert result is None

    def test_get_file_names_partial(self):
        files = ['/projects/data/a.csv', '/projects/data/b.csv']
        result = dataset_utils.get_file_names_partial(files, '/projects/data')
        assert len(result) == 2

    def test_get_file_name_from_partial(self):
        result = dataset_utils.get_file_name_from_partial(
            'classA/file1.csv', '/projects/data'
        )
        expected = os.path.join('/projects/data', 'classA/file1.csv')
        assert result == expected


class TestDatasetSplit:
    """Tests for dataset_split (COCO format)."""

    def test_split_produces_two_splits(self, sample_coco_dataset):
        split_names = ['train', 'val']
        splits = dataset_utils.dataset_split(
            sample_coco_dataset, split_factor=0.7, split_names=split_names, random_seed=42
        )
        assert 'train' in splits
        assert 'val' in splits

    def test_split_preserves_total_count(self, sample_coco_dataset):
        split_names = ['train', 'val']
        splits = dataset_utils.dataset_split(
            sample_coco_dataset, split_factor=0.7, split_names=split_names, random_seed=42
        )
        total_images = len(splits['train']['images']) + len(splits['val']['images'])
        assert total_images == len(sample_coco_dataset['images'])

    def test_split_preserves_categories(self, sample_coco_dataset):
        split_names = ['train', 'val']
        splits = dataset_utils.dataset_split(
            sample_coco_dataset, split_factor=0.7, split_names=split_names, random_seed=42
        )
        assert splits['train']['categories'] == sample_coco_dataset['categories']
        assert splits['val']['categories'] == sample_coco_dataset['categories']

    def test_split_annotations_match_images(self, sample_coco_dataset):
        split_names = ['train', 'val']
        splits = dataset_utils.dataset_split(
            sample_coco_dataset, split_factor=0.7, split_names=split_names, random_seed=42
        )
        for split_name in split_names:
            image_ids = {img['id'] for img in splits[split_name]['images']}
            for ann in splits[split_name]['annotations']:
                assert ann['image_id'] in image_ids


class TestDatasetSplitLimit:
    """Tests for dataset_split_limit."""

    def test_limit_none_returns_same(self, sample_coco_dataset):
        result = dataset_utils.dataset_split_limit(sample_coco_dataset, None)
        assert result is sample_coco_dataset

    def test_limit_reduces_count(self, sample_coco_dataset):
        result = dataset_utils.dataset_split_limit(sample_coco_dataset, 3)
        assert len(result['images']) == 3
        assert len(result['annotations']) == 3
