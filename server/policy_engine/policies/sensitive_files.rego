package policy.sensitive_files

import future.keywords.in

default allow := true

# Deny auto-merge if sensitive files are modified
allow := false {
    input.action == "merge"
    some file in input.files
    file in ["secrets.txt", ".env", "config/private.key"]
}