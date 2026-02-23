from aivis.parser import parse_tool_list, norm_name


def test_parse_numbered_list():
    raw = (
        "1. Asana - Great task management (no citation)\n"
        "2. Jira - Best for agile teams (no citation)\n"
        "3. Monday.com - Visual workflows (no citation)\n"
    )
    tools, meta = parse_tool_list(raw)
    assert meta["parse_success"] is True
    assert len(tools) == 3
    assert tools[0].rank == 1
    assert tools[0].name_norm == "asana"
    assert tools[1].name_norm == "jira"
    assert tools[2].name_norm == "monday.com"


def test_parse_empty_response():
    tools, meta = parse_tool_list("")
    assert meta["parse_success"] is False
    assert "PE-06" in meta["parse_errors"]


def test_parse_prose_response():
    raw = "Project management is important."
    tools, meta = parse_tool_list(raw)
    assert meta["parse_success"] is False
    assert "OCV-01" in meta["violations"]


def test_parse_with_citations():
    raw = (
        "1. Asana - Great tool (asana.com)\n"
        "2. Jira - Agile standard (atlassian.com)\n"
    )
    tools, meta = parse_tool_list(raw)
    assert meta["parse_success"] is True
    assert tools[0].citation_domains == ["asana.com"]
    assert tools[1].citation_domains == ["atlassian.com"]


def test_parse_no_citation_explicit():
    raw = "1. Asana - Great tool (no citation)\n"
    tools, meta = parse_tool_list(raw)
    assert tools[0].citation_domains == []


def test_parse_duplicate_detection():
    raw = "1. Asana - Good\n2. Asana - Also good\n3. Jira - Fine\n"
    tools, meta = parse_tool_list(raw)
    assert meta["has_duplicates"] is True
    assert "PE-04" in meta["parse_errors"]


def test_norm_name_aliases():
    assert norm_name("Jira Software") == "jira"
    assert norm_name("Monday") == "monday.com"
    assert norm_name("  ClickUp  ") == "clickup"
    assert norm_name("MS Project") == "microsoft project"


def test_norm_name_suffix_strip():
    assert norm_name("Wrike Platform") == "wrike"
    assert norm_name("Asana Tool") == "asana"


def test_parse_bold_names():
    raw = (
        "1. **Asana** - Great for teams (no citation)\n"
        "2. **Jira** - Agile standard (no citation)\n"
    )
    tools, meta = parse_tool_list(raw)
    assert meta["parse_success"] is True
    assert tools[0].name_raw == "Asana"
    assert tools[1].name_raw == "Jira"


def test_exceeds_10():
    items = [f"{i}. Tool{i} - reason (no citation)" for i in range(1, 13)]
    raw = "\n".join(items)
    tools, meta = parse_tool_list(raw)
    assert len(tools) == 10
    assert "OCV-02" in meta["violations"]
# === STEP 23: Parser Normalization + Format Hardening Tests ===


def test_norm_name_parenthetical_stripped():
    """Parenthetical qualifiers are removed during normalization."""
    from aivis.parser import norm_name
    assert norm_name("Asana (by Atlassian)") == "asana"
    assert norm_name("Monday.com (formerly DaPulse)") == "monday.com"
    assert norm_name("ClickUp (free tier)") == "clickup"


def test_norm_name_alias_table():
    """Alias table resolves known variants."""
    from aivis.parser import norm_name
    assert norm_name("Jira Software") == "jira"
    assert norm_name("Atlassian Jira") == "jira"
    assert norm_name("Jira by Atlassian") == "jira"
    assert norm_name("Monday") == "monday.com"
    assert norm_name("click up") == "clickup"
    assert norm_name("Ms Project") == "microsoft project"


def test_norm_name_case_and_whitespace():
    """Normalization handles case and extra whitespace."""
    from aivis.parser import norm_name
    assert norm_name("  Asana  ") == "asana"
    assert norm_name("CLICKUP") == "clickup"
    assert norm_name("Monday.com") == "monday.com"
    assert norm_name("  Jira   Software  ") == "jira"


def test_parse_preamble_ignored():
    """Prose preamble lines are skipped, only numbered items parsed."""
    from aivis.parser import parse_tool_list
    text = (
        "Here are the top project management tools:\n"
        "\n"
        "1. Asana - Task management (no citation)\n"
        "2. Monday.com - Visual PM (no citation)\n"
        "3. ClickUp - All-in-one (no citation)"
    )
    tl, meta = parse_tool_list(text)
    assert len(tl) == 3
    assert meta["parse_errors"] == []
    assert tl[0].name_norm == "asana"


def test_parse_trailing_prose_ignored():
    """Trailing non-list prose is skipped."""
    from aivis.parser import parse_tool_list
    text = (
        "1. Asana - Task management (no citation)\n"
        "2. Monday.com - Visual PM (no citation)\n"
        "\n"
        "Note: Rankings vary based on team needs."
    )
    tl, meta = parse_tool_list(text)
    assert len(tl) == 2
    assert meta["parse_errors"] == []


def test_parse_bold_names_stripped():
    """Bold markdown names are cleaned properly."""
    from aivis.parser import parse_tool_list
    text = (
        "1. **Asana** - Task management (no citation)\n"
        "2. **Monday.com** - Visual PM (no citation)"
    )
    tl, meta = parse_tool_list(text)
    assert len(tl) == 2
    assert meta["parse_errors"] == []
    assert tl[0].name_raw == "Asana"
    assert tl[1].name_raw == "Monday.com"


def test_parse_colon_separator():
    """Colon-separated format parses correctly."""
    from aivis.parser import parse_tool_list
    text = (
        "1. Asana: Task management for teams (no citation)\n"
        "2. Monday.com: Visual project management (no citation)"
    )
    tl, meta = parse_tool_list(text)
    assert len(tl) == 2
    assert meta["parse_errors"] == []
    assert tl[0].why != ""
    assert tl[1].why != ""