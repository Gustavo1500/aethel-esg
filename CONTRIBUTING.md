# Contributing to Aethel

Thank you for your interest in contributing to Aethel! We welcome contributions from actuarial professionals, developers, mathematicians, and financial planners.

To keep the development process smooth, please follow these guidelines.

## Code of Conduct
By participating in this project, you agree to abide by our Code of Conduct [link to CODE_OF_CONDUCT.md if added].

## Getting Started

1. **Fork the Repository:** Create a personal fork on GitHub.
2. **Clone Locally:**
   ```bash
   git clone https://github.com/Gustavo1500/aethel-esg.git
   cd aethel-esg
3. **Set Up a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
4. **Install Dependencies in Editable Mode:**
   Install development, plotting, and testing tools:
   ```bash
   pip install -e .[numba,plots,docs]
   pip install pytest pytest-cov ruff pre-commit
   ```

## Development Workflow

### Coding Standards
* We use [Ruff](https://github.com/astral-sh/ruff) for linting and code formatting.
* Please ensure your changes comply with formatting standards by running:
  ```bash
  ruff check .
  ruff format .
  ```

### Running Tests
All contributions that modify mathematical logic or utility functions must include corresponding tests. Run the test suite before submitting a Pull Request:
```bash
pytest
```
Ensure your changes do not decrease the project's test coverage.

### Pull Request Process
1. Create a new branch for your change (e.g., `feature/dynamic-erp` or `bugfix/cir-stability`).
2. Make your changes and add corresponding unit or robustness tests.
3. Commit with clear, descriptive commit messages.
4. Push to your fork and submit a Pull Request to the `master` branch.
5. Ensure the automated Continuous Integration (CI) tests pass on your PR.
