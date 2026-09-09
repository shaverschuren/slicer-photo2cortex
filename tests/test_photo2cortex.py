import photo2cortex


def test_subject_id_from_dir_uses_final_directory_name(tmp_path):
    subject_dir = tmp_path / "RESP0123"
    subject_dir.mkdir()
    assert photo2cortex.subject_id_from_dir(str(subject_dir) + "\\") == "RESP0123"


def test_discover_batch_subjects_returns_only_matching_directories(tmp_path):
    (tmp_path / "RESP0002").mkdir()
    (tmp_path / "RESP0001").mkdir()
    (tmp_path / "RESP0003").write_text("not a directory")
    (tmp_path / "OTHER").mkdir()

    assert photo2cortex.discover_batch_subjects(str(tmp_path), "RESP*") == [
        str(tmp_path / "RESP0001"),
        str(tmp_path / "RESP0002"),
    ]


def test_parse_args_selects_batch_mode_without_subject():
    args = photo2cortex.parse_args([])
    assert args.subject_dir is None


def test_parse_args_selects_single_subject_mode():
    args = photo2cortex.parse_args([r"D:\other_dataset\RESP1234"])
    assert args.subject_dir == r"D:\other_dataset\RESP1234"


def test_single_subject_config_validation_does_not_require_batch_root(tmp_path):
    slicer = tmp_path / "Slicer.exe"
    slicer.write_text("")
    photo_dir = tmp_path / "photos"
    photo_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "slicer_exe_path: %s\nphoto_data_dir: %s\nbatch:\n  mri_data_dir: %s\n"
        % (slicer, photo_dir, tmp_path / "unavailable")
    )

    config = photo2cortex.load_config(str(config_path), validate_batch=False)
    assert config["batch"]["mri_data_dir"].endswith("unavailable")


def test_process_batch_delegates_each_subject_and_stops_on_abort(monkeypatch, tmp_path):
    subject_dirs = [str(tmp_path / "RESP0001"), str(tmp_path / "RESP0002"), str(tmp_path / "RESP0003")]
    calls = []
    statuses = iter([
        photo2cortex.SubjectProcessStatus.COMPLETED,
        photo2cortex.SubjectProcessStatus.ABORT_BATCH,
    ])
    monkeypatch.setattr(photo2cortex, "discover_batch_subjects", lambda *_: subject_dirs)

    def fake_process_subject(subject_dir, **kwargs):
        calls.append((subject_dir, kwargs))
        return next(statuses)

    monkeypatch.setattr(photo2cortex, "process_subject", fake_process_subject)
    config = {
        "photo_data_dir": "photos",
        "slicer_exe_path": "slicer",
        "batch": {
            "mri_data_dir": "subjects",
            "subject_dir_regex": "RESP*",
            "reprocess": False,
            "process_only_photo": False,
        },
    }

    assert photo2cortex.process_batch(config) == 0
    assert [call[0] for call in calls] == subject_dirs[:2]
    assert all(call[1]["process_only_envelope"] is False for call in calls)


def test_process_subject_envelope_mode_calls_slicer_without_photo_workflow(monkeypatch, tmp_path):
    subject_dir = tmp_path / "RESP0001"
    (subject_dir / "mri").mkdir(parents=True)
    (subject_dir / "surf").mkdir()
    (subject_dir / "mri" / "T1.mgz").write_text("")
    (subject_dir / "surf" / "lh.pial").write_text("")
    (subject_dir / "surf" / "rh.pial").write_text("")
    calls = []

    class Result:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(
        photo2cortex.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)) or Result(),
    )

    status = photo2cortex.process_subject(
        str(subject_dir),
        photo_data_dir=str(tmp_path / "missing_photos"),
        slicer_executable="slicer",
        process_only_envelope=True,
    )

    assert status is photo2cortex.SubjectProcessStatus.COMPLETED
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert "--create_envelope_mode" in command
    assert "--no-main-window" in command
    assert kwargs == {"capture_output": True, "text": True}