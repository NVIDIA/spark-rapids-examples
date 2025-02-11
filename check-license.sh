#! /bin/bash -e

LICENSE_PATTERN="Copyright \(c\) .*, NVIDIA CORPORATION."
FILES=($(find . -type f | sed 's|^\./||'))

included_file_patterns="*.sh,
            *.java,
            *.py,
            *.pbtxt,
            *Dockerfile*,
            *Jenkinsfile*,
            *.yml,
            *.yaml,
            *.cpp,
            *.hpp,
            *.txt,
            *.cu,
            *.scala,
            *.ini,
            *.xml"
excluded_file_patterns=""

IFS="," read -r -a INCLUDE_PATTERNS <<< "$(echo "$included_file_patterns" | tr -d ' ' | tr -d '\n')"
IFS="," read -r -a EXCLUDE_PATTERNS <<< "$(echo "$excluded_file_patterns" | tr -d ' ' | tr -d '\n')"
echo "Included file patterns: ${INCLUDE_PATTERNS[@]}"
echo "Excluded file patterns: ${EXCLUDE_PATTERNS[@]}"



NO_LICENSE_FILES=""
for FILE in "${FILES[@]}"; do
    INCLUDE=false
    for INCLUDE_PATTERN in "${INCLUDE_PATTERNS[@]}"; do
        if [[ $FILE == $INCLUDE_PATTERN ]]; then
            INCLUDE=true
            break
        fi
    done
    EXCLUDE=false
    for EXCLUDE_PATTERN in "${EXCLUDE_PATTERNS[@]}"; do
        if [[ $FILE == $EXCLUDE_PATTERN ]]; then
            EXCLUDE=true
            break
        fi
    done
    if [[ $INCLUDE == true && $EXCLUDE == false ]]; then
        echo "Checking $FILE"
        if !(grep -Eiq "$LICENSE_PATTERN" "$FILE"); then
            NO_LICENSE_FILES+="$FILE "
        fi
    fi
done

# Output result
echo "--------- RESULT ---------"
if [ ! -z "$NO_LICENSE_FILES" ]; then
    echo "Following files missed copyright/license header or expired:"
    echo $NO_LICENSE_FILES | tr ' ' '\n'
    echo "If there are files that are not modified by your PR, please try upmerge first."
else
    echo "All files passed the check"
fi

# Add copyright header
COPYRIGHT="# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"


if [ ! -z "$NO_LICENSE_FILES" ]; then
    for FILE in $NO_LICENSE_FILES; do
        # sed -i '1i '"$(echo $COPYRIGHT)" $FILE
        # sed -i '1r /dev/stdin' $FILE <<<"$COPYRIGHT"
        cp -p $FILE temp
        echo "$COPYRIGHT" | cat - $FILE > temp
        cp -p temp $FILE
        rm temp
    done
fi
