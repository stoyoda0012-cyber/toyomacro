"""Public CLI command surface.

The public CLI exposes only implemented, distributed functionality.
``gui`` (private companion layer, not installed by the public package)
and ``fit`` (placeholder pending a data schema) are intentionally not
registered; fitting is available through the Python APIs. Running with
no arguments prints help and exits successfully instead of launching
anything.
"""

import re
import subprocess
import sys


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, '-c',
         'from toyomacro.cli import main; import sys; '
         f'sys.argv={["toyomacro", *argv]!r}; main()'],
        capture_output=True, text=True)


def _help_commands(help_text: str) -> set[str]:
    """Parse the subcommand names out of argparse's help text."""
    m = re.search(r'\{([\w,-]+)\}', help_text)
    assert m, f'no subcommand list found in help output:\n{help_text}'
    return set(m.group(1).split(','))


def test_help_hides_gui_and_fit():
    out = _run_cli('--help')
    assert out.returncode == 0
    commands = _help_commands(out.stdout)
    assert 'gui' not in commands
    assert 'fit' not in commands


def test_help_keeps_import_and_convert():
    out = _run_cli('--help')
    assert out.returncode == 0
    assert {'import', 'convert'} <= _help_commands(out.stdout)


def test_no_args_prints_help_and_exits_zero():
    out = _run_cli()
    assert out.returncode == 0
    assert 'usage: toyomacro' in out.stdout
    # Help only — nothing (GUI or otherwise) may be launched.


def test_gui_and_fit_are_rejected_as_unknown():
    for cmd in ('gui', 'fit'):
        out = _run_cli(cmd)
        assert out.returncode == 2, cmd  # argparse invalid-choice exit
        assert 'invalid choice' in out.stderr, cmd
