#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for file utility functions."""

import pytest
import tempfile
from pathlib import Path
from qumat.file_utils import path_exists, ensure_file_exists


class TestPathExists:
    """Tests for path_exists function."""

    def test_path_exists_with_existing_file(self):
        """Test that path_exists returns True for an existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                assert path_exists(temp_path) is True
            finally:
                Path(temp_path).unlink()

    def test_path_exists_with_existing_directory(self):
        """Test that path_exists returns True for an existing directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            assert path_exists(temp_dir) is True

    def test_path_exists_with_nonexistent_path(self):
        """Test that path_exists returns False for a nonexistent path."""
        nonexistent_path = "/tmp/nonexistent_file_12345.txt"
        assert path_exists(nonexistent_path) is False

    def test_path_exists_with_string_path(self):
        """Test that path_exists works with string paths."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                assert path_exists(temp_path) is True
            finally:
                Path(temp_path).unlink()

    def test_path_exists_with_path_object(self):
        """Test that path_exists works with Path objects."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            try:
                assert path_exists(temp_path) is True
            finally:
                temp_path.unlink()


class TestEnsureFileExists:
    """Tests for ensure_file_exists function."""

    def test_ensure_file_exists_with_existing_file(self):
        """Test that ensure_file_exists returns Path for an existing file."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                result = ensure_file_exists(temp_path)
                assert isinstance(result, Path)
                assert result == Path(temp_path)
            finally:
                Path(temp_path).unlink()

    def test_ensure_file_exists_with_nonexistent_file(self):
        """Test that ensure_file_exists raises FileNotFoundError for nonexistent file."""
        nonexistent_path = "/tmp/nonexistent_file_12345.txt"
        with pytest.raises(FileNotFoundError, match="File not found"):
            ensure_file_exists(nonexistent_path)

    def test_ensure_file_exists_with_string_path(self):
        """Test that ensure_file_exists works with string paths."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            try:
                result = ensure_file_exists(temp_path)
                assert isinstance(result, Path)
            finally:
                Path(temp_path).unlink()

    def test_ensure_file_exists_with_path_object(self):
        """Test that ensure_file_exists works with Path objects."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            try:
                result = ensure_file_exists(temp_path)
                assert isinstance(result, Path)
            finally:
                temp_path.unlink()

    def test_ensure_file_exists_with_directory(self):
        """Test that ensure_file_exists works with directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = ensure_file_exists(temp_dir)
            assert isinstance(result, Path)
            assert result == Path(temp_dir)
