# Ours, not Doron's: keep pytest from collecting the vendored tests, whose `src.` imports would resolve to our package.
collect_ignore_glob = ["*"]
