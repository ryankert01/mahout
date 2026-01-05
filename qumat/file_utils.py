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
"""File utility functions for Qumat."""

from pathlib import Path
from typing import Union


def path_exists(file_path: Union[str, Path]) -> bool:
    """
    Check if a file or directory exists at the given path.

    This function provides a convenient way to verify file existence
    using Python's pathlib library.

    :param file_path: Path to the file or directory to check.
        Can be either a string or a Path object.
    :type file_path: str | Path
    :returns: True if the path exists, False otherwise.
    :rtype: bool

    :example:
        >>> from qumat.file_utils import path_exists
        >>> path_exists("/path/to/file.txt")
        True
        >>> path_exists("/nonexistent/file.txt")
        False
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    return path.exists()


def ensure_file_exists(file_path: Union[str, Path]) -> Path:
    """
    Verify that a file exists and return its Path object.

    Raises a FileNotFoundError if the file does not exist.

    :param file_path: Path to the file to check.
        Can be either a string or a Path object.
    :type file_path: str | Path
    :returns: Path object for the existing file.
    :rtype: Path
    :raises FileNotFoundError: If the file does not exist.

    :example:
        >>> from qumat.file_utils import ensure_file_exists
        >>> ensure_file_exists("/path/to/existing/file.txt")
        PosixPath('/path/to/existing/file.txt')
        >>> ensure_file_exists("/nonexistent/file.txt")
        Traceback (most recent call last):
            ...
        FileNotFoundError: File not found: /nonexistent/file.txt
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path
