package policy.secrets

default allow := true

# Deny if content contains potential secrets
allow := false {
    contains(input.content, "password")
}

allow := false {
    contains(input.content, "secret")
}

allow := false {
    contains(input.content, "api_key")
}