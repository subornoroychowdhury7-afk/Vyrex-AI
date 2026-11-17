from unittest.mock import patch

import pytest

from vyrex_ai.cli import build_parser, main


def test_evaluate_english_levels_parser_defaults():
    args = build_parser().parse_args([
        "evaluate-english-levels",
        "--dataset", "labels.jsonl",
        "--report", "report.json",
    ])

    assert args.command == "evaluate-english-levels"
    assert args.folds == 5
    assert args.seed == 42


def test_evaluate_english_levels_parser_custom_values():
    args = build_parser().parse_args([
        "evaluate-english-levels",
        "--dataset", "labels.jsonl",
        "--report", "report.json",
        "--folds", "3",
        "--seed", "123",
    ])

    assert args.folds == 3
    assert args.seed == 123


def test_evaluate_english_levels_dispatches_experiment(capsys):
    expected = {
        "status": "completed",
        "sample_count": 100,
        "fold_count": 5,
        "model_names": [
            "logistic_regression",
            "linear_svm",
            "random_forest",
        ],
    }

    with (
        patch(
            "vyrex_ai.data.english_level_ml_experiment.run_ml_experiment",
            return_value=expected,
        ) as experiment,
        patch("vyrex_ai.cli.load_config") as load_config,
    ):
        with patch(
            "sys.argv",
            [
                "vyrex",
                "evaluate-english-levels",
                "--dataset", "labels.jsonl",
                "--report", "report.json",
            ],
        ):
            main()

    experiment.assert_called_once_with(
        dataset_path="labels.jsonl",
        report_path="report.json",
        n_splits=5,
        random_state=42,
    )
    load_config.assert_not_called()

    output = capsys.readouterr().out
    assert "VYREX ENGLISH-LEVEL ML EXPERIMENT COMPLETE" in output
    assert "No production model was approved or deployed." in output


def test_evaluate_english_levels_propagates_failure():
    with patch(
        "vyrex_ai.data.english_level_ml_experiment.run_ml_experiment",
        side_effect=ValueError("Insufficient independent sentence groups"),
    ):
        with patch(
            "sys.argv",
            [
                "vyrex",
                "evaluate-english-levels",
                "--dataset", "labels.jsonl",
                "--report", "report.json",
            ],
        ):
            with pytest.raises(
                ValueError,
                match="Insufficient independent sentence groups",
            ):
                main()


def test_expand_english_levels_parser_defaults():
    args = build_parser().parse_args([
        "expand-english-levels",
        "--input", "accepted.jsonl",
        "--dry-run",
    ])

    assert args.command == "expand-english-levels"
    assert args.input == "accepted.jsonl"
    assert args.output is None
    assert args.manifest is None
    assert args.maximum_records == 10_000
    assert args.dry_run is True


def test_expand_english_levels_parser_custom_values():
    args = build_parser().parse_args([
        "expand-english-levels",
        "--input", "accepted.jsonl",
        "--output", "candidates.jsonl",
        "--manifest", "manifest.jsonl",
        "--maximum-records", "250",
    ])

    assert args.maximum_records == 250
    assert args.dry_run is False


def test_expand_english_levels_dry_run_dispatches(capsys):
    expected = {
        "dry_run": True,
        "records_examined": 10,
        "candidates_created": 8,
        "source_modified": False,
    }

    with (
        patch(
            "vyrex_ai.data.english_level_ml_expansion.expand_tatoeba_candidates",
            return_value=expected,
        ) as expansion,
        patch("vyrex_ai.cli.load_config") as load_config,
        patch(
            "sys.argv",
            [
                "vyrex",
                "expand-english-levels",
                "--input", "accepted.jsonl",
                "--dry-run",
            ],
        ),
    ):
        main()

    expansion.assert_called_once_with(
        input_path="accepted.jsonl",
        output_path=None,
        manifest_path=None,
        maximum_records=10_000,
        dry_run=True,
    )
    load_config.assert_not_called()

    output = capsys.readouterr().out
    assert "VYREX ENGLISH-LEVEL CANDIDATE EXPANSION COMPLETE" in output
    assert "no candidate or manifest files were published" in output


def test_expand_english_levels_normal_export_dispatches():
    with (
        patch(
            "vyrex_ai.data.english_level_ml_expansion.expand_tatoeba_candidates",
            return_value={"candidates_created": 5},
        ) as expansion,
        patch("vyrex_ai.cli.load_config") as load_config,
        patch(
            "sys.argv",
            [
                "vyrex",
                "expand-english-levels",
                "--input", "accepted.jsonl",
                "--output", "candidates.jsonl",
                "--manifest", "manifest.jsonl",
                "--maximum-records", "500",
            ],
        ),
    ):
        main()

    expansion.assert_called_once_with(
        input_path="accepted.jsonl",
        output_path="candidates.jsonl",
        manifest_path="manifest.jsonl",
        maximum_records=500,
        dry_run=False,
    )
    load_config.assert_not_called()


def test_expand_english_levels_requires_output_paths_without_dry_run():
    with patch(
        "sys.argv",
        [
            "vyrex",
            "expand-english-levels",
            "--input", "accepted.jsonl",
        ],
    ):
        with pytest.raises(
            ValueError,
            match="--output and --manifest are required",
        ):
            main()
