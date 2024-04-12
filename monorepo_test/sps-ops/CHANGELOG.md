# Changelog

## [0.4.5](https://github.com/chime-sps/sps-ops/compare/v0.4.4...v0.4.5) (2024-03-07)


### Bug Fixes

* **image.yml:** release-please returing a string and not a boolean ([f55faf4](https://github.com/chime-sps/sps-ops/commit/f55faf4d1d0657acff73dbc4dc96bcbdd833ad0f))

## [0.4.4](https://github.com/chime-sps/sps-ops/compare/v0.4.3...v0.4.4) (2024-03-07)


### Bug Fixes

* **release.yml:** Remove extra step and correct output names in job ([b0dcf16](https://github.com/chime-sps/sps-ops/commit/b0dcf169a8ed4cf701d9e1f5337300e1321d60e2))

## [0.4.3](https://github.com/chime-sps/sps-ops/compare/v0.4.2...v0.4.3) (2024-03-07)


### Bug Fixes

* **release.yml:** Add outputs to reusable workflow with specific syntax ([b4ce500](https://github.com/chime-sps/sps-ops/commit/b4ce5000d58d2d5d4d6d3463951e11605e7354b9))
* **release.yml:** Add outputs to reusable workflow with specific syntax ([6200c89](https://github.com/chime-sps/sps-ops/commit/6200c8979d3109b801417958b7158bfc2c7769fa))

## [0.4.2](https://github.com/chime-sps/sps-ops/compare/v0.4.1...v0.4.2) (2024-03-07)


### Bug Fixes

* **release.yml:** Change way variables are outputted from release.yml job ([baaf023](https://github.com/chime-sps/sps-ops/commit/baaf0235773c6a0ab5aac30ff9717721620e4f01))
* **release.yml:** Change way variables are outputted from release.yml job ([6b60397](https://github.com/chime-sps/sps-ops/commit/6b60397c767e58bb8990a0720fb0155f38d2c7d8))
* **release.yml:** Missing = sign ([495afbd](https://github.com/chime-sps/sps-ops/commit/495afbd46194b823c1d23c5eeebd8048ad01ad1e))

## [0.4.1](https://github.com/chime-sps/sps-ops/compare/v0.4.0...v0.4.1) (2024-03-07)


### Bug Fixes

* **release.py:** Release tag not appearing in next job...? ([92df3d2](https://github.com/chime-sps/sps-ops/commit/92df3d2df1fb18354e5e8b67bf7d2d87eed50c3b))
* **release.py:** Release tag not appearing in next job...? ([141237d](https://github.com/chime-sps/sps-ops/commit/141237dc5f5bbc84d080b771a53b8b5f76e9a627))

## [0.4.0](https://github.com/chime-sps/sps-ops/compare/v0.3.2...v0.4.0) (2024-03-07)


### Features

* **continuous-integration.yml:** Fix CICD for new dispatching of tests to pipeline_batch_db ([59d1e6e](https://github.com/chime-sps/sps-ops/commit/59d1e6e03ecd39907d0c5982dad62e368192e644))


### Bug Fixes

* **continuous-deployment.yml:** Add debug job ([a0a9767](https://github.com/chime-sps/sps-ops/commit/a0a9767e2b7af26387afc4bb9f18663766d0bfb9))
* **continuous-deployment.yml:** Add fallback values ([169e8ba](https://github.com/chime-sps/sps-ops/commit/169e8baa41b4708667c68e1fb8101b42c2af2cbd))
* **continuous-deployment.yml:** Add new debug job ([0c6b191](https://github.com/chime-sps/sps-ops/commit/0c6b19113b3332657f39b04310ab0b66f068d791))
* **continuous-integration.yml:** Add check for GitHub Actor in dispatch tests workflow ([622ffa2](https://github.com/chime-sps/sps-ops/commit/622ffa2e77b7ed479a26cb97a183ae4127a5a1cd))
* **continuous-integration.yml:** Remove dispatch tests trigger from sps-ops ([1a44356](https://github.com/chime-sps/sps-ops/commit/1a4435616149bc4c7f3efc5b9ed9c05e7f5fed4b))

## [0.3.2](https://github.com/chime-sps/sps-ops/compare/v0.3.1...v0.3.2) (2024-03-06)


### Bug Fixes

* **continuous-deployment.yml:** Trigger new release ([12cf333](https://github.com/chime-sps/sps-ops/commit/12cf333cdb1d4ec91a718105f66c88d6f3ece590))

## [0.3.1](https://github.com/chime-sps/sps-ops/compare/v0.3.0...v0.3.1) (2024-03-06)


### Bug Fixes

* **release.yml:** Add path input to release-please action ([6869ba0](https://github.com/chime-sps/sps-ops/commit/6869ba0f02dac439b72b09c3573e5e9b06c4c486))
* **release.yml:** Remove dispatch tests from CI ([fa4b3b1](https://github.com/chime-sps/sps-ops/commit/fa4b3b12989d3538c47b7d145fcd9e414c3ebded))
* **release.yml:** Upgrade release-please action version ([134bd87](https://github.com/chime-sps/sps-ops/commit/134bd8782b60ae77d78d6b89f610a7358fba6c4b))

## [0.3.0](https://github.com/chime-sps/sps-ops/compare/v0.2.0...v0.3.0) (2023-12-10)


### Features

* **pipeline.yml:** Final version of pipeline stack ([78d8c7d](https://github.com/chime-sps/sps-ops/commit/78d8c7d7ece9a066f880820fe7b7fcfc79cf4b3f))

## [0.2.0](https://github.com/chime-sps/sps-ops/compare/v0.1.2...v0.2.0) (2023-11-15)


### Features

* **query_cadvisor.py:** Add script to query cAdvisor from Prometheus… ([#19](https://github.com/chime-sps/sps-ops/issues/19)) ([608b73f](https://github.com/chime-sps/sps-ops/commit/608b73fdf43e0db699065ce32830001bcb2f9cb9))

## [0.1.2](https://github.com/chime-sps/sps-ops/compare/v0.1.1...v0.1.2) (2023-11-08)


### Bug Fixes

* **pipeline.yml:** Add new SPS Pipeline Docker Stack configuration + a bunch of QoL improvements ([94ef7f6](https://github.com/chime-sps/sps-ops/commit/94ef7f6baa57a605079ff8e916c36564f1ee3ab3))

## [0.1.1](https://github.com/chime-sps/sps-ops/compare/v0.1.0...v0.1.1) (2023-11-01)


### Bug Fixes

* **Dockerfile:** Fix Dockerfile, there is no dev group in this pyproject.toml ([5bd3747](https://github.com/chime-sps/sps-ops/commit/5bd3747ced42ec038e8f612fae6f8f0d6c8d87c2))

## 0.1.0 (2023-11-01)


### Features

* **obs:** added pyroscope ([3a61210](https://github.com/chime-sps/sps-ops/commit/3a61210c9a8120005bb41d59caa6eeffd6687348))
* **pipeline.yml:** Adding new SPS pipeline Docker Swarm Stack file ([9be6991](https://github.com/chime-sps/sps-ops/commit/9be69919f1b87c664540529edf59cd2b5c3640f4))
* **stacks:** added obs stack ([db14b1b](https://github.com/chime-sps/sps-ops/commit/db14b1b4365aaaf06854048c7c94ee62a9cf3332))
* **stacks:** started ([b3e2b72](https://github.com/chime-sps/sps-ops/commit/b3e2b72ab9847af978ee090e4309fc76626593f5))


### Bug Fixes

* **.pre-commit-config.yml:** Files/folders starting in . are ignored with git add *, doing git add . now ([9d16470](https://github.com/chime-sps/sps-ops/commit/9d16470d53c031f15c94e94bc9fed9fb09185ec3))
* **continuous-deployment.yml:** Add new CI/CD and PC YML config files and fix Dockerfile ([f6b28b5](https://github.com/chime-sps/sps-ops/commit/f6b28b56ebeea60ac1b9a28c735c3ce8cdef3fdc))
* **obs:** fixed entry config path ([e38561b](https://github.com/chime-sps/sps-ops/commit/e38561b0111d0642bb1aca6787571cc4c32834cb))
* **obs:** fixed port config ([3358a3d](https://github.com/chime-sps/sps-ops/commit/3358a3d99386eff93fc7d0eb1983aefa84f2184f))
* **obs:** fixed with stacks ([127bd4b](https://github.com/chime-sps/sps-ops/commit/127bd4b71a4c3002d2fc11021091c7231f7bb9db))
* **pipeline.yml:** Change syntax error for constraints in compose file ([7d989c8](https://github.com/chime-sps/sps-ops/commit/7d989c807ca41b5b653ba45e6fc50e118c3aa408))
* **pipeline.yml:** Fix incorrect node label constraints ([bfdfd35](https://github.com/chime-sps/sps-ops/commit/bfdfd35d81073be15c179d568be6faf9cc323288))
* **pipeline.yml:** Incorrect "tags" vs "tag" usage on workflow run ([d601c3a](https://github.com/chime-sps/sps-ops/commit/d601c3a37258b0c01d5209f9f4484299c7d8db93))
* **pipeline.yml:** workflow run command missing site flag ([605c9d8](https://github.com/chime-sps/sps-ops/commit/605c9d8d4997e0d54ed4626de861f967ae0f2cb9))
* **stacks:** obs ([30063ad](https://github.com/chime-sps/sps-ops/commit/30063ad442e262845483859b8d9e7b99ed2760e1))
