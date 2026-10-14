"""Handwritten fixtures for offline tests; no text is copied from a model run."""


def synthetic_plan(action: str = "Prepare") -> str:
    """Return an invented text-file task with the workflow parser's five sections."""
    return f"""{action} the synthetic fixture output by writing the requested text to a local file.

1. **Goal & acceptance criteria**
   - `/fixture/output.txt` must contain the exact requested text; the fixture checks the bytes literally.
   - The requested content is `sample text` followed by a newline, with no added commentary in the file.
   - Keep the other fixture files unchanged so the acceptance check only needs to inspect this output.
2. **Environment findings**
   - This invented workspace contains a writable `/fixture` directory and an ordinary command shell.
   - The output file has not been created in this synthetic scenario; no package installation is needed.
3. **Plan**
   - {action}: create `/fixture/output.txt` using a shell command that writes the literal requested content.
   - Read the file back before finishing and compare its contents with the acceptance criteria above.
4. **Execution checkpoints**
   - After writing, check that the output is a regular file in the intended fixture directory.
   - Inspect the contents without changing any neighboring file or adding unrelated workspace state.
5. **Verification**
   - cat /fixture/output.txt -- expect the requested fixture text.
""".strip()
