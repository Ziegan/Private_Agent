from private_agent.agent.project_paths import discover_project_resources


def test_project_resources_are_discovered_from_all_supported_locations(tmp_path):
    (tmp_path / "workspace").mkdir()
    (tmp_path / "resources" / "rag").mkdir(parents=True)
    (tmp_path / "private_agent" / "resources" / "skills").mkdir(parents=True)

    result = discover_project_resources(tmp_path)

    assert result.workspace == (tmp_path / "workspace").resolve()
    assert result.rag == (tmp_path / "resources" / "rag").resolve()
    assert result.skills == (
        tmp_path / "private_agent" / "resources" / "skills"
    ).resolve()


def test_project_root_folder_takes_precedence_over_resource_folders(tmp_path):
    (tmp_path / "rag").mkdir()
    (tmp_path / "resources" / "rag").mkdir(parents=True)
    (tmp_path / "private_agent" / "resources" / "rag").mkdir(parents=True)

    result = discover_project_resources(tmp_path)

    assert result.rag == (tmp_path / "rag").resolve()


def test_project_resources_ignores_symlinks_that_escape_project_root(tmp_path):
    external = tmp_path.parent / f"{tmp_path.name}-external"
    external.mkdir()
    (tmp_path / "resources").mkdir(parents=True)
    (tmp_path / "resources" / "skills").symlink_to(
        external,
        target_is_directory=True,
    )

    try:
        result = discover_project_resources(tmp_path)
    finally:
        (tmp_path / "resources" / "skills").unlink()
        external.rmdir()

    assert result.skills is None


def test_project_resources_are_optional(tmp_path):
    result = discover_project_resources(tmp_path)

    assert result.workspace is None
    assert result.rag is None
    assert result.skills is None
