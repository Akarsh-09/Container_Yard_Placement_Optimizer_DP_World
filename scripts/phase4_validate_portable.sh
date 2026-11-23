#!/bin/bash
# macOS-only development portability adapter. The protected validator is run
# unchanged; its two grep -oP expressions are executed by Perl's regex engine.
# All other grep calls use the system binary. No checks are skipped.
set -e
grep() {
    if [ "$1" = "-oP" ]; then
        shift
        local pattern="$1"
        shift
        perl -ne 'BEGIN {$pattern=shift @ARGV;} while (/$pattern/g) {$matched=1; print "$&\n"} END {exit($matched ? 0 : 1)}' "$pattern" "$@"
    else
        command grep "$@"
    fi
}
export -f grep
exec bash validate_submission.sh
