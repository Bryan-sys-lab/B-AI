package policy.shell_commands

default allow := true

# Deny unsafe shell commands
allow := false {
    input.command in ["rm -rf /", "sudo rm -rf /", "format c:", "del /f /s /q c:\\*"]
}