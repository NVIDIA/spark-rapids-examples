#!/bin/bash
#
# Copyright (c) 2026, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

# Script to extract libcudf.so from rapids-4-spark jar
# This allows faster builds by avoiding building cuDF from source

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/target"
NATIVE_DEPS_DIR="$TARGET_DIR/native-deps"
CUDF_REPO_DIR="$TARGET_DIR/cudf-repo"

RAPIDS4SPARK_VERSION="${RAPIDS4SPARK_VERSION:-26.02.0-SNAPSHOT}"
SCALA_VERSION="${SCALA_VERSION:-2.12}"
CUDA_VERSION="${CUDA_VERSION:-cuda12}"
CUDF_BRANCH="${CUDF_BRANCH:-main}"

echo "=================================================="
echo "Extract cuDF Dependencies for UDF Examples"
echo "=================================================="
echo "RAPIDS4SPARK_VERSION: $RAPIDS4SPARK_VERSION"
echo "SCALA_VERSION: $SCALA_VERSION"
echo "CUDA_VERSION: $CUDA_VERSION"
echo "CUDF_BRANCH: $CUDF_BRANCH"
echo "=================================================="

# Create directories
mkdir -p "$NATIVE_DEPS_DIR"
mkdir -p "$CUDF_REPO_DIR"

# Find rapids-4-spark jar in local Maven repository
MAVEN_REPO="${HOME}/.m2/repository"

# Try multiple naming patterns
JAR_PATH_WITH_CLASSIFIER="$MAVEN_REPO/com/nvidia/rapids-4-spark_${SCALA_VERSION}/${RAPIDS4SPARK_VERSION}/rapids-4-spark_${SCALA_VERSION}-${RAPIDS4SPARK_VERSION}-${CUDA_VERSION}.jar"
JAR_PATH_NO_CLASSIFIER="$MAVEN_REPO/com/nvidia/rapids-4-spark_${SCALA_VERSION}/${RAPIDS4SPARK_VERSION}/rapids-4-spark_${SCALA_VERSION}-${RAPIDS4SPARK_VERSION}.jar"

echo "Looking for rapids-4-spark jar..."
echo "  Pattern 1 (with classifier): $JAR_PATH_WITH_CLASSIFIER"
echo "  Pattern 2 (no classifier):   $JAR_PATH_NO_CLASSIFIER"

if [ -f "$JAR_PATH_WITH_CLASSIFIER" ]; then
    JAR_PATH="$JAR_PATH_WITH_CLASSIFIER"
    echo "✓ Found jar (with classifier): $JAR_PATH"
elif [ -f "$JAR_PATH_NO_CLASSIFIER" ]; then
    JAR_PATH="$JAR_PATH_NO_CLASSIFIER"
    echo "✓ Found jar (no classifier): $JAR_PATH"
else
    echo ""
    echo "ERROR: rapids-4-spark jar not found!"
    echo "Tried:"
    echo "  $JAR_PATH_WITH_CLASSIFIER"
    echo "  $JAR_PATH_NO_CLASSIFIER"
    echo ""
    echo "For SNAPSHOT versions:"
    echo "  cd /path/to/spark-rapids"
    echo "  mvn clean install -DskipTests"
    echo ""
    echo "For release versions:"
    echo "  mvn dependency:get -Dartifact=com.nvidia:rapids-4-spark_${SCALA_VERSION}:${RAPIDS4SPARK_VERSION}:jar:${CUDA_VERSION}"
    exit 1
fi

# Extract libcudf.so and dependencies
echo "Extracting native libraries..."
unzip -q -o "$JAR_PATH" "*/libcudf.so*" "*/libnvcomp.so*" -d "$TARGET_DIR/temp"

# Move libraries to native-deps directory, detecting conflicts
echo "Moving extracted libraries..."
CONFLICT_COUNT=0

# Use process substitution to avoid subshell issues
while IFS= read -r source_file; do
    filename=$(basename "$source_file")
    dest_file="$NATIVE_DEPS_DIR/$filename"
    
    if [ -f "$dest_file" ]; then
        # File already exists - check if it's the same
        if ! cmp -s "$source_file" "$dest_file"; then
            echo "WARNING: Conflicting library detected: $filename" >&2
            echo "  Existing: $dest_file" >&2
            echo "  New:      $source_file" >&2
            echo "  Keeping existing file, skipping new one" >&2
            CONFLICT_COUNT=$((CONFLICT_COUNT + 1))
        fi
        # Remove the duplicate source file
        rm -f "$source_file"
    else
        # No conflict, move the file
        mv "$source_file" "$dest_file"
    fi
done < <(find "$TARGET_DIR/temp" -name "*.so*")

if [ "$CONFLICT_COUNT" -gt 0 ]; then
    echo "WARNING: $CONFLICT_COUNT library file(s) had conflicts. Review the warnings above." >&2
fi

rm -rf "$TARGET_DIR/temp"

if [ ! -f "$NATIVE_DEPS_DIR/libcudf.so" ]; then
    echo "ERROR: Failed to extract libcudf.so from jar"
    exit 1
fi

echo "✓ Extracted libraries to: $NATIVE_DEPS_DIR"
ls -lh "$NATIVE_DEPS_DIR"

# Clone cuDF repo for headers (shallow clone)
if [ ! -d "$CUDF_REPO_DIR/.git" ]; then
    echo "Cloning cuDF repository for headers..."
    git clone --depth 1 --branch "$CUDF_BRANCH" https://github.com/rapidsai/cudf.git "$CUDF_REPO_DIR"
    echo "✓ Cloned cuDF repo to: $CUDF_REPO_DIR"
else
    echo "✓ cuDF repo already exists at: $CUDF_REPO_DIR"
    echo "  (Delete it to re-clone: rm -rf $CUDF_REPO_DIR)"
fi

echo ""
echo "=================================================="
echo "Setup complete! You can now build with:"
echo "  mvn clean package -P udf-native-examples"
echo ""
echo "This will use prebuilt libcudf.so and avoid"
echo "building cuDF from source (much faster!)."
echo "=================================================="

