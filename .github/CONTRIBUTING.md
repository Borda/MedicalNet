# Contributing to MedicalNet

Welcome to [report Issues](https://github.com/Tencent/MedicalNet/issues)
or [pull requests](https://github.com/Tencent/MedicalNet/pulls). It's recommended to read the following Contributing
Guide first before contributing.

## Issues

We use Github Issues to track public bugs and feature requests.

### Search Known Issues First

Please search the existing issues to see if any similar issue or feature request has already been filed. You should make
sure your issue isn't redundant.

### Reporting New Issues

If you open an issue, the more information the better. Such as detailed description, screenshot or video of your
problem, logcat and xlog or code blocks for your crash.

## Pull Requests

We strongly welcome your pull request to make MedicalNet better.

### Branch Management

There are three main branches here:

1. `master` branch.
   1. It is the latest (pre-)release branch. We use `master` for tags, with version number `1.1.0`, `1.2.0`, `1.3.0`...
   1. **Don't submit any PR on `master` branch.**
1. `develop` branch.
   1. It is our stable developing branch. After full testing, `develop` will be merged to `master` branch for the next
      release.
   1. **You are recommended to submit bugfix or feature PR on `develop` branch.**
1. `hotfix` branch.
   1. It is the latest tag version for hot fix. If we accept your pull request, we may just tag with version
      number `1.1.1`, `1.2.3`.
   1. **Only submit urgent PR on `hotfix` branch for next specific release.**

Normal bugfix or feature request should be submitted to `develop` branch. After full testing, we will merge them
to `master` branch for the next release.

If you have some urgent bugfixes on a published version, but the `master` branch have already far away with the latest
tag version, you can submit a PR on hotfix. And it will be cherry picked to `develop` branch if it is possible.

```
master
 ��
develop        <--- hotfix PR
 ��
feature/bugfix PR
```

### Make Pull Requests

The code team will monitor all pull request, we run some code check and test on it. After all tests passed, we will
accecpt this PR. But it won't merge to `master` branch at once, which have some delay.

Before submitting a pull request, please make sure the followings are done:

1. Fork the repo and create your branch from `master` or `hotfix`.
1. Update code or documentation if you have changed APIs.
1. Add the copyright notice to the top of any new files you've added.
1. Check your code lints and checkstyles.
1. Test and test again your code.
1. Now, you can submit your pull request on `develop` or `hotfix` branch.

## Code Style Guide

Use [Code Style for Python](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)
.

- 4 spaces for indentation rather than tabs
- private function should start with "\_\_".
- function parameters should start with "\_".

## License

By contributing to MedicalNet, you agree that your contributions will be licensed under
its [MIT LICENSE](https://github.com/Tencent/MedicalNet/blob/master/LICENSE)
