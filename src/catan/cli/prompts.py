"""Questions to the user of ``catan match``.

Interactive (a terminal): numbered choices, Enter takes the default, "q"
aborts. Non-interactive (``--non-interactive`` or no terminal): a question
with a default takes it (reported as a note); one without stops the run
with InputRequired, which says what was asked and which option answers it.
"""

import sys
import textwrap


class InputRequired(RuntimeError):
    """A decision is needed, but no one can be asked."""


class Aborted(RuntimeError):
    """The user stopped the run."""


class Prompter:
    def __init__(self, *, interactive=None, stdin=None, stdout=None):
        self.stdin = sys.stdin if stdin is None else stdin
        self.stdout = sys.stdout if stdout is None else stdout
        if interactive is None:
            interactive = bool(getattr(self.stdin, "isatty", lambda: False)())
        self.interactive = interactive
        self.warnings = []

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def _write(self, text=""):
        print(text, file=self.stdout, flush=True)

    def info(self, text=""):
        self._write(text)

    def note(self, text):
        self._write(_indent(text, "  "))

    def warn(self, text):
        """A warning, repeated in the final summary."""
        self.warnings.append(text)
        self._write(_indent(text, "  ! ", "    "))

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------

    def _unanswered(self, question, *, choices=None, option=None):
        lines = [question]
        if choices:
            lines.append("Possible answers: " + "; ".join(label for _, label in choices) + ".")
        if option:
            lines.append(f"Answer it with {option}, or run in a terminal to be asked.")
        return InputRequired("\n".join(lines))

    def _read(self, prompt):
        self.stdout.write(prompt)
        self.stdout.flush()
        line = self.stdin.readline()
        if not line:
            raise Aborted("Input ended.")
        line = line.strip()
        if line.lower() in {"q", "quit"}:
            raise Aborted("Stopped by the user.")
        return line

    def choose(self, question, choices, *, default=None, option=None, details=None):
        """One of ``choices`` [(key, label)]; returns its key. ``default``:
        a key (Enter in a terminal; taken without a terminal). ``option``:
        the command-line option that answers this question. ``details``:
        further lines shown with the question."""
        keys = [key for key, _ in choices]
        if default is not None and default not in keys:
            raise ValueError(f"Default {default!r} is not a choice.")

        if not self.interactive:
            if default is None:
                raise self._unanswered(question, choices=choices, option=option)
            label = dict(choices)[default]
            self.note(f"{question} → {label} (default; no terminal to ask)")
            return default

        self._write()
        self._write(_indent(question, "? ", "  "))
        if details:
            self._write(_indent(details, "    "))
        for index, (key, label) in enumerate(choices, start=1):
            marker = " (default)" if key == default else ""
            self._write(_indent(f"[{index}] {label}{marker}", "    ", "        "))
        hint = "number" + (", Enter for the default" if default is not None else "") + ", q to stop"
        while True:
            answer = self._read(f"  Choice ({hint}): ")
            if not answer and default is not None:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(choices):
                return keys[int(answer) - 1]
            self._write(f"  Please enter a number from 1 to {len(choices)}.")

    def confirm(self, question, *, default=None, option=None, details=None):
        return (
            self.choose(
                question,
                [(True, "yes"), (False, "no")],
                default=default,
                option=option,
                details=details,
            )
        )

    def ask_number(
        self, question, *, default=None, minimum=None, maximum=None, option=None,
        details=None, integer=False,
    ):
        if not self.interactive:
            if default is None:
                raise self._unanswered(question, option=option)
            self.note(f"{question} → {default} (default; no terminal to ask)")
            return default

        self._write()
        self._write(_indent(question, "? ", "  "))
        if details:
            self._write(_indent(details, "    "))
        suffix = f" [{default}]" if default is not None else ""
        while True:
            answer = self._read(f"  Value{suffix} (q to stop): ")
            if not answer and default is not None:
                return default
            try:
                value = int(answer) if integer else float(answer)
            except ValueError:
                self._write("  Please enter a number.")
                continue
            if (minimum is not None and value < minimum) or (
                maximum is not None and value > maximum
            ):
                bounds = (
                    f"between {minimum} and {maximum}"
                    if minimum is not None and maximum is not None
                    else f"at least {minimum}" if minimum is not None
                    else f"at most {maximum}"
                )
                self._write(f"  Please enter a value {bounds}.")
                continue
            return value

    def ask_text(self, question, *, default=None, option=None, details=None, validate=None):
        """Free text; ``validate(text)`` returns an error message or None."""
        if not self.interactive:
            if default is None:
                raise self._unanswered(question, option=option)
            self.note(f"{question} → {default} (default; no terminal to ask)")
            return default

        self._write()
        self._write(_indent(question, "? ", "  "))
        if details:
            self._write(_indent(details, "    "))
        suffix = f" [{default}]" if default is not None else ""
        while True:
            answer = self._read(f"  Answer{suffix} (q to stop): ")
            if not answer and default is not None:
                answer = default
            if not answer:
                continue
            error = validate(answer) if validate is not None else None
            if error is None:
                return answer
            self._write(f"  {error}")


def _indent(text, first, rest=None):
    rest = " " * len(first) if rest is None else rest
    lines = str(text).splitlines() or [""]
    out = []
    for index, line in enumerate(lines):
        wrapped = textwrap.wrap(
            line, width=96, subsequent_indent=rest, break_long_words=False, break_on_hyphens=False
        ) or [""]
        out.append((first if index == 0 else rest) + "\n".join(wrapped))
    return "\n".join(out)
