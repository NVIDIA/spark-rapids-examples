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

###############################################################################
# Clone or update cuDF repository for header files
#
# This script is called by Maven during the build process to obtain cuDF
# headers needed for compiling native UDF code.
#
# Usage:
#   clone-cudf-repo.sh <target_directory> <branch_name>
#
# Arguments:
#   target_directory - Directory where cuDF repo will be cloned
#   branch_name      - Git branch to clone/checkout
#
# Exit codes:
#   0 - Success
#   1 - Failed to clone, fetch, or checkout
###############################################################################

set -e
set -o pipefail

# Parse arguments
if [ $# -ne 2 ]; then
    echo "ERROR: Usage: $0 <target_directory> <branch_name>" >&2
    exit 1
fi

CUDF_DIR="$1"
BRANCH="$2"

echo "================================================"
echo "cuDF Repository Management"
echo "  Target directory: $CUDF_DIR"
echo "  Branch: $BRANCH"
echo "================================================"

# Check if repository already exists
if [ ! -d "$CUDF_DIR/.git" ]; then
    # Repository doesn't exist - clone it
    echo "Cloning cuDF repository ($BRANCH branch)..."
    
    git clone --depth 1 --branch "$BRANCH" \
        https://github.com/rapidsai/cudf.git "$CUDF_DIR" || {
        echo "ERROR: Failed to clone cuDF from branch $BRANCH" >&2
        echo "Please check:" >&2
        echo "  1. Network connectivity to GitHub" >&2
        echo "  2. Branch '$BRANCH' exists in cuDF repository" >&2
        exit 1
    }
    
    echo "✓ Successfully cloned cuDF repository"
else
    # Repository exists - verify and update if needed
    echo "cuDF repository exists, verifying branch..."
    cd "$CUDF_DIR" || {
        echo "ERROR: Cannot access directory $CUDF_DIR" >&2
        exit 1
    }
    
    # Get current branch
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")
    
    if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
        # Branch mismatch - fetch and switch to correct branch
        echo "Branch mismatch detected:"
        echo "  Current branch: $CURRENT_BRANCH"
        echo "  Expected branch: $BRANCH"
        echo "Fetching and switching to $BRANCH..."
        
        git fetch --depth 1 origin "$BRANCH" || {
            echo "ERROR: Failed to fetch branch $BRANCH from origin" >&2
            echo "Please check:" >&2
            echo "  1. Network connectivity to GitHub" >&2
            echo "  2. Branch '$BRANCH' exists in cuDF repository" >&2
            exit 1
        }
        
        git checkout "$BRANCH" || {
            echo "ERROR: Failed to checkout branch $BRANCH" >&2
            exit 1
        }
        
        git reset --hard "origin/$BRANCH" || {
            echo "ERROR: Failed to reset to origin/$BRANCH" >&2
            exit 1
        }
        
        echo "✓ Switched to branch $BRANCH"
    else
        echo "✓ Already on correct branch ($BRANCH)"
    fi
fi

echo "================================================"
echo "✓ cuDF repository ready at: $CUDF_DIR"
echo "  Branch: $BRANCH"
echo "================================================"

exit 0

