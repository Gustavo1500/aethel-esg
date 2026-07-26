<!-- Thank you for contributing to Aethel! Please fill out the sections below to help us review your Pull Request. -->

## Description

Please provide a brief summary of the changes introduced by this Pull Request, the motivation behind them, and how they solve the problem.

### Related Issues
<!-- If this PR addresses an open issue, link it below (e.g., "- Closes #12"). If there are no related issues, you can delete this section. -->
- Closes #

---

## Type of Change
<!-- Please check the options that are relevant (change [ ] to [x]). -->

- [ ] **Bug Fix** (non-breaking change which fixes an issue)
- [ ] **New Feature** (non-breaking change which adds functionality)
- [ ] **Performance Optimization** (changes that improve execution speed or memory usage)
- [ ] **Documentation Update** (modifications to files in `/docs` or docstrings)
- [ ] **Breaking Change** (fix or feature that would cause existing functionality to change)

---

## Mathematical or Logical Modifications
<!-- If this PR modifies any mathematical models (CIR, OU, Merton Jump-Diffusion), policy parameters, or statistical queries, please describe the logic below. -->

- **Model(s) affected:** <!-- e.g., CIR, OU, Merton, Decumulation engine, none -->
- **Formula adjustments:** <!-- If any formulas were changed, provide the math here or reference docs/mathematics.md -->
- **Impact on outcomes:** <!-- How does this affect expected scenario trajectories or decumulation success rates? -->

---

## Testing & Verification
<!-- Please describe the tests you ran to verify your changes. -->

### Automated Tests
- [ ] Added new unit tests in `tests/test_engine.py` or `tests/test_robustness.py`
- [ ] Verified all tests pass locally by running `pytest`

### Numerical Consistency Check (for mathematical modifications)
- [ ] Verified that the outputs contain no `NaN` or `inf` values
- [ ] Checked that seed determinism/reproducibility is preserved

---

## Submission Checklist
<!-- Before submitting, please verify that you have completed the following steps: -->

- [ ] My code follows the code style of this project (run `ruff format .` to format).
- [ ] I have run static analysis checks locally (run `ruff check .` to check).
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have updated the documentation in `/docs` (or inline docstrings) to reflect my changes.
- [ ] My changes generate no new warnings during testing.
