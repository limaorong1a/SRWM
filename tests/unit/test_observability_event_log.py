from paradigm_experiments.observability.event_log import TaskFileLogger, safe_filename_fragment


def test_safe_filename_fragment_removes_invalid_path_chars():
    assert safe_filename_fragment('pick/clean:bad*name?') == "pick_clean_bad_name"


def test_task_file_logger_writes_timestamped_lines(tmp_path):
    log_path = tmp_path / "task.log"
    logger = TaskFileLogger(str(log_path), echo=False)
    logger.line("hello")
    logger.blank()
    logger.banner("done")
    logger.close()

    text = log_path.read_text(encoding="utf-8")
    assert "hello" in text
    assert "done" in text
