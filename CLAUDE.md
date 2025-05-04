# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- Run tests: `python -m pytest <test_file.py>` 
- Run single test: `python -m pytest <test_file.py>::<test_function_name>`
- Run module directly: `python -m hongikjiki.<module>`
- Build package: `pip install -e .`
- Format code: `black .` (88 char line limit)
- Sort imports: `isort .`

## Code Style

- Python 3.8+ compatible
- Black formatting (88 char line length)
- Docstrings for all public functions/classes ("""triple quotes""")
- Import order: standard lib, third-party, local (using isort)
- File naming: snake_case for modules, *_test.py for tests
- Function/variable naming: snake_case 
- Class naming: PascalCase
- Korean language comments/strings allowed with UTF-8 encoding
- Log messages for significant operations (using standard logging)
- Exception handling with specific exception types, not bare except