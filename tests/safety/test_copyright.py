"""Copyright rules keep non-permissive and proprietary code out, not BSD files or email."""
import pytest

from forge.safety.copyright import detect_copyright

GPL_HEADER = """\
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License.
"""

BSD_HEADER = """\
// Copyright (c) 2021, Example Org
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
"""


def _flagged(text: str) -> bool:
    return any(finding.rule == "copyright" for finding in detect_copyright(text))


@pytest.mark.parametrize("text", [
    "// SPDX-License-Identifier: GPL-3.0-only\nint main() {}",
    "# SPDX-License-Identifier: AGPL-3.0-or-later",
    "/* SPDX-License-Identifier: MIT AND GPL-2.0-only */",
    "# SPDX-License-Identifier: LicenseRef-Proprietary",
    GPL_HEADER,
    "This library is licensed under the terms of the GNU Lesser General Public License v2.1.",
    "# Proprietary and confidential. Unauthorized copying of this file is strictly prohibited.",
    "// Copyright 2024 Initech Corp. All rights reserved.\nfunc main() {}",
    "/* Do not distribute outside Initech. */",
])
def test_non_permissive_or_proprietary_code_is_flagged(text):
    assert _flagged(text)


@pytest.mark.parametrize("text", [
    "// SPDX-License-Identifier: MIT",
    "# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception",
    "# SPDX-License-Identifier: MIT OR GPL-3.0-or-later",
    "/* SPDX-License-Identifier: BSD-3-Clause AND ISC */",
    BSD_HEADER,
    "# Copyright (c) 2024 Jane Doe\n# Permission is hereby granted, free of charge, to any person. All rights reserved.",
    "© 2026 Acme Inc. All rights reserved. Unsubscribe here.",
    "This email is proprietary and confidential. Do not distribute.",
    "We avoid GPL dependencies because of the GNU license terms.",
    "def close_ticket(ticket_id): return api.close(ticket_id)",
    "",
])
def test_permissive_code_email_footers_and_prose_are_not_flagged(text):
    assert not _flagged(text)


def test_the_finding_names_what_matched():
    (finding,) = detect_copyright("// SPDX-License-Identifier: GPL-3.0-only")
    assert "GPL-3.0-only" in finding.detail
