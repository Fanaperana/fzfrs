# Contributing to fuzzly

Thanks for your interest in improving fuzzly.

This repository is both a usable Rust crate and a learning resource for people
who want to understand fuzzy search and Levenshtein distance implementations.
Please keep both goals in mind when contributing.

## Ways to contribute

- Fix bugs or edge cases
- Improve algorithm explanations and examples
- Improve benchmarks and performance investigations
- Improve the docs site in `docs/`
- Add tests for correctness or regressions
- Improve beginner-friendliness in comments and README examples

## Development setup

```sh
git clone https://github.com/Fanaperana/fuzzy-search-rs
cd fuzzy-search-rs
cargo test
cargo run
```

Optional checks:

```sh
cargo bench
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Contribution guidelines

- Keep changes focused and small when possible
- Preserve the educational clarity of the code
- Add or update tests when behavior changes
- Avoid unrelated refactors in the same pull request
- Keep README and docs site examples in sync with the code when relevant

## Pull request checklist

Before opening a PR, please make sure:

- The project builds and tests pass locally
- New behavior is covered by tests where appropriate
- Documentation is updated if user-facing behavior changed
- The change is explained clearly in the PR description

## Good first contributions

- Improve wording in README or docs site explanations
- Add examples for common fuzzy-search use cases
- Improve visual explanations of the algorithm
- Add targeted benchmarks for a new scenario

Thanks for helping make fuzzly easier to learn from and more useful in real
projects.
