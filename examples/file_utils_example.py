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
"""Example demonstrating the path_exists() and ensure_file_exists() utilities."""

import tempfile
from pathlib import Path
from qumat import path_exists, ensure_file_exists


def main():
    """Demonstrate path existence checking functionality."""
    print("=== Path Existence Checking Example ===\n")

    # Create a temporary file for demonstration
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(b"Hello, Qumat!")

    print(f"Created temporary file: {temp_path}")

    # Example 1: Check if a file exists using path_exists()
    print(f"\n1. path_exists('{temp_path}'): {path_exists(temp_path)}")

    # Example 2: Verify file exists using ensure_file_exists()
    try:
        result = ensure_file_exists(temp_path)
        print(f"2. ensure_file_exists('{temp_path}'): {result}")
    except FileNotFoundError as e:
        print(f"2. Error: {e}")

    # Example 3: Check for a non-existent file
    nonexistent = "/path/to/nonexistent/file.txt"
    print(f"\n3. path_exists('{nonexistent}'): {path_exists(nonexistent)}")

    # Example 4: Try to ensure a non-existent file exists (will raise error)
    try:
        ensure_file_exists(nonexistent)
    except FileNotFoundError as e:
        print(f"4. ensure_file_exists('{nonexistent}'): Raised {type(e).__name__}")

    # Example 5: Works with Path objects too
    path_obj = Path(temp_path)
    print(f"\n5. path_exists(Path object): {path_exists(path_obj)}")

    # Cleanup
    Path(temp_path).unlink()
    print(f"\n6. After deletion, path_exists('{temp_path}'): {path_exists(temp_path)}")

    print("\n=== Example Complete ===")


if __name__ == "__main__":
    main()
