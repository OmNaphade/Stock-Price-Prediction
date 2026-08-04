"""Empty on purpose — its presence anchors pytest's rootdir at the project
root so `config`, `auth`, `data_access`, `features`, `models`,
`evaluation`, and `services` import as top-level packages regardless of
the working directory pytest is invoked from."""
