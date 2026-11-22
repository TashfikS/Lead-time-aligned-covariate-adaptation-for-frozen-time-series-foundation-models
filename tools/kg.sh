#!/usr/bin/env bash
# usage: tools/kg.sh <CRED_ENV_VAR_NAME> <kaggle cli args...>
# Writes the token from the named env var into a private temp HOME for one call; never echoes it.
V="$1"; shift
tok="${!V}"
if [ -z "$tok" ]; then echo "credential $V not present in env" >&2; exit 2; fi
d=$(mktemp -d -p "$PWD" .kgXXXX); printf "%s" "$tok" > "$d/t"; chmod 600 "$d/t"
env -u KAGGLE_USERNAME -u KAGGLE_KEY HOME="$d" KAGGLE_API_TOKEN="$d/t" timeout 300 kaggle "$@" </dev/null 2>&1 | grep -v "outdated \`kaggle\` version"
rc=${PIPESTATUS[0]}
rm -rf "$d"
exit $rc
