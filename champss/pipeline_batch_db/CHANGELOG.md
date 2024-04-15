# Changelog

## [1.4.0](https://github.com/chime-sps/pipeline_batch_db/compare/v1.3.3...v1.4.0) (2024-04-12)


### Features

* **pipeline.py:** Update logging folder syntax ([9deb3a7](https://github.com/chime-sps/pipeline_batch_db/commit/9deb3a711bec8d6a01cea91e890a74664efd9d19))
* **sps_config.yml:** Enable fitting ([3235206](https://github.com/chime-sps/pipeline_batch_db/commit/323520659c59347019dad54b16c27933acecc900))
* **sps_config.yml:** Update search and quality parameters ([3ee57aa](https://github.com/chime-sps/pipeline_batch_db/commit/3ee57aa164de42cb66ba3dac2eb58e93887a3903))
* **workflow.py:** Create New Workflow/Swarm API ([555af7b](https://github.com/chime-sps/pipeline_batch_db/commit/555af7b9b3db0b467174b90d413a91a746723a05))


### Bug Fixes

* **cand.py:** Fix .pointing_id ([bb9cde0](https://github.com/chime-sps/pipeline_batch_db/commit/bb9cde0ac59915648d1ea88306afac8bc8b21c78))
* **pipeline.py:** Fix pointing class change ([#168](https://github.com/chime-sps/pipeline_batch_db/issues/168)) ([517c3b1](https://github.com/chime-sps/pipeline_batch_db/commit/517c3b106ccac6d2639c7e60ae374ad3611cef34))

## [1.3.3](https://github.com/chime-sps/pipeline_batch_db/compare/v1.3.2...v1.3.3) (2024-03-25)


### Bug Fixes

* **pyproject.toml:** Add multi-pointing dependency ([cdafd34](https://github.com/chime-sps/pipeline_batch_db/commit/cdafd34161df0d94f523f8882c228772f75627f5))

## [1.3.2](https://github.com/chime-sps/pipeline_batch_db/compare/v1.3.1...v1.3.2) (2024-03-21)


### Bug Fixes

* **pipeline.py:** Try memory reservation // 1.5 instead of // 2 for number of threads used ([0aba59b](https://github.com/chime-sps/pipeline_batch_db/commit/0aba59bb3ab2715cc5cbe2b956dea3564a32e3f1))

## [1.3.1](https://github.com/chime-sps/pipeline_batch_db/compare/v1.3.0...v1.3.1) (2024-03-20)


### Bug Fixes

* **pipeline.py:** Progress one day further in event of an error ([9c0452a](https://github.com/chime-sps/pipeline_batch_db/commit/9c0452ae4d5e0f622344b755b94c275699633b5a))

## [1.3.0](https://github.com/chime-sps/pipeline_batch_db/compare/v1.2.1...v1.3.0) (2024-03-09)


### Features

* **pipeline.py:** Simplify Beamforming ([221521f](https://github.com/chime-sps/pipeline_batch_db/commit/221521f5dfe30a092576e88d0c23187447cf399f))


### Bug Fixes

* **pipeline.py:** Use basepath in beam strategist ([b67ab3b](https://github.com/chime-sps/pipeline_batch_db/commit/b67ab3baf75b5b5942254f7cd012652451d9689a))
* **pipeline.py:** Use basepath in beam strategist ([f4946f2](https://github.com/chime-sps/pipeline_batch_db/commit/f4946f210158528e9d8c5b1945a54a43f37666cc))

## [1.2.1](https://github.com/chime-sps/pipeline_batch_db/compare/v1.2.0...v1.2.1) (2024-03-07)


### Bug Fixes

* **continuous-integration.yml:** update common files, e.g. continuous-integration.yml, continuous-deployment.yml, pre-commit-config.yaml ([10c7b6b](https://github.com/chime-sps/pipeline_batch_db/commit/10c7b6b363e9db31bc78122b32d9e0312ce4afb7))

## [1.2.0](https://github.com/chime-sps/pipeline_batch_db/compare/v1.1.1...v1.2.0) (2024-03-06)


### Features

* **pipeline.py:** add new CICD for benchmarking tests and improve scheduling/alerts ([629ed87](https://github.com/chime-sps/pipeline_batch_db/commit/629ed872177ccee9a88c73de478e1b517a7d87bc))

## [1.1.1](https://github.com/chime-sps/pipeline_batch_db/compare/v1.1.0...v1.1.1) (2024-03-01)


### Bug Fixes

* **Dockerfile:** Restrict Miniconda3 installation to Python3.11 ([a233358](https://github.com/chime-sps/pipeline_batch_db/commit/a23335885dc144355210e130d7cbeec34f5827cf))
* **pipeline.py:** Add QoL improvements to Docker Swarm/Workflow scheduling ([2aa9186](https://github.com/chime-sps/pipeline_batch_db/commit/2aa9186509a1094607c934b53282004120bd0fcc))

## [1.1.0](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.10...v1.1.0) (2024-02-23)


### Features

* **pipeline.py:** Adding new Slack messages for mean pipeline execution time and hours late processing has started ([c6962ff](https://github.com/chime-sps/pipeline_batch_db/commit/c6962fff7b8881d3228dd1e7d484d8839371b640))


### Bug Fixes

* **poetry.lock:** Update poetry lock ([#142](https://github.com/chime-sps/pipeline_batch_db/issues/142)) ([25bd5d6](https://github.com/chime-sps/pipeline_batch_db/commit/25bd5d6a58ed279426913abba03a10f2a2ae32a8))

## [1.0.10](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.9...v1.0.10) (2024-02-16)


### Bug Fixes

* **pipeline.py:** Prevent crash when observation too short ([#140](https://github.com/chime-sps/pipeline_batch_db/issues/140)) ([e08dfe0](https://github.com/chime-sps/pipeline_batch_db/commit/e08dfe0c523df562bfa750b1a90bcd8930c61d78))

## [1.0.9](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.8...v1.0.9) (2024-02-16)


### Bug Fixes

* **pipeline.py:** Fix candidates using stacked spectra ([#138](https://github.com/chime-sps/pipeline_batch_db/issues/138)) ([a052dfe](https://github.com/chime-sps/pipeline_batch_db/commit/a052dfe1381c34f594cf8596a905c8aeb8229f4b))
* **pipeline.py:** Fix stack pipeline crash  ([82a2905](https://github.com/chime-sps/pipeline_batch_db/commit/82a29052d94e2273bbee17cb0d4866afed876826))

## [1.0.8](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.7...v1.0.8) (2023-12-14)


### Bug Fixes

* **pipeline.py:** Slack bug fix, Nproc at start of day's processing ([6997964](https://github.com/chime-sps/pipeline_batch_db/commit/6997964b1cf89c2629e1e18e1d9744a557f56e61))

## [1.0.7](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.6...v1.0.7) (2023-12-14)


### Bug Fixes

* **clear_raw_data.py:** Dummy commitizen style squash commit for docker image ([2389b3f](https://github.com/chime-sps/pipeline_batch_db/commit/2389b3f6d94d92babbdb0d935ca57629e22f92ba))

## [1.0.6](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.5...v1.0.6) (2023-12-13)


### Bug Fixes

* **clear_raw_data.py:** Check if folder exists before posting to Slack and waiting 1 hour ([81d912f](https://github.com/chime-sps/pipeline_batch_db/commit/81d912fb22b41e1495a47ead5bdff72f953653f4))

## [1.0.5](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.4...v1.0.5) (2023-12-10)


### Bug Fixes

* **clear_raw_data.py:** Remove check for date being before given start date ([f2eabf5](https://github.com/chime-sps/pipeline_batch_db/commit/f2eabf54c716c526277188ec3b740d7c0723cd09))

## [1.0.4](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.3...v1.0.4) (2023-12-09)


### Bug Fixes

* **pipeline.py:** Order of subtracting dates in continuous processing was reversed ([afb9142](https://github.com/chime-sps/pipeline_batch_db/commit/afb91428ca5aed1f7b98f2543540087e8b6e4e2d))

## [1.0.3](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.2...v1.0.3) (2023-12-09)


### Bug Fixes

* **pipeline.py:** Add more threads and change Slack alert message about RFI processes ([935cfc4](https://github.com/chime-sps/pipeline_batch_db/commit/935cfc434803856f73524ddd0d3d11a16b5520c0))

## [1.0.2](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.1...v1.0.2) (2023-12-08)


### Bug Fixes

* **poetry.lock:** Use latest folding ([3117852](https://github.com/chime-sps/pipeline_batch_db/commit/31178529744d86407c8a21685c222cce5dc91882))

## [1.0.1](https://github.com/chime-sps/pipeline_batch_db/compare/v1.0.0...v1.0.1) (2023-12-08)


### Bug Fixes

* **pipeline.py:** Add instantiation of coordinates variables ([d0271b7](https://github.com/chime-sps/pipeline_batch_db/commit/d0271b7aaa5c9958cd0debc091e512eacd34cf62))

## [1.0.0](https://github.com/chime-sps/pipeline_batch_db/compare/v0.3.0...v1.0.0) (2023-12-08)


### ⚠ BREAKING CHANGES

* **clear_raw_data.py:** Since these parameters didn't exist before it will break anything using the old PowerSpectraSearch class
* **main:** Since these parameters didn't exist before it will break anything using the old PowerSpectraSearch class

### Features

* **clear_raw_data.py:** Automate Raw Data Deletion ([9c72c53](https://github.com/chime-sps/pipeline_batch_db/commit/9c72c53cc49e29a3dea33a9f9f222f7db19a7f2c))
* **main:** enable new clustering ([8876f3a](https://github.com/chime-sps/pipeline_batch_db/commit/8876f3aff8726cb64e83c26fff19d5d0f5916693))
* **pipeline.py:** Adding better Workflow functionality ([80a95c8](https://github.com/chime-sps/pipeline_batch_db/commit/80a95c8fd7f851587b008696f4a158cfefc35fd6))
* **pipeline.py:** Automate Daily Continuous Processing ([b068b2c](https://github.com/chime-sps/pipeline_batch_db/commit/b068b2cb18fab99ac09db0bb148d43a542dda08d))

## [0.3.0](https://github.com/chime-sps/pipeline_batch_db/compare/v0.2.0...v0.3.0) (2023-11-08)


### Features

* **pipeline.py:** Adding Workflow functionality ([0f5448c](https://github.com/chime-sps/pipeline_batch_db/commit/0f5448c3dfe209e27da9432c47357cb24e7f5c3a))

## [0.2.0](https://github.com/chime-sps/pipeline_batch_db/compare/v0.1.1...v0.2.0) (2023-11-01)


### Features

* **docker:** added linear algebra files ([48aa697](https://github.com/chime-sps/pipeline_batch_db/commit/48aa6977f5bf036c49056b8f7930e9b06f4c26e4))


### Bug Fixes

* **__init__.py:** Adding --db-host option to run-pipeline script to allow choice of sps-archiver, sps-compute1, or sps-compute2 as MongoDB host ([6a27973](https://github.com/chime-sps/pipeline_batch_db/commit/6a2797337c4b28d2aa2f93b9e47ba266a48e5cfa))
* **Dockerfile:** Fix broken Dockerfile ([190db7d](https://github.com/chime-sps/pipeline_batch_db/commit/190db7d6c5cbec056f2e89d9eab14e24fefce9de))
* **docker:** removed build essential after all install steps ([06e0697](https://github.com/chime-sps/pipeline_batch_db/commit/06e06978c1701843b83f20f3075322bc18f06df7))

## [0.1.1](https://github.com/chime-sps/pipeline_batch_db/compare/v0.1.0...v0.1.1) (2023-08-09)


### Bug Fixes

* **Dockerfile:** Fixing docker image not able to access pip ([be9df21](https://github.com/chime-sps/pipeline_batch_db/commit/be9df21d71efb2fa91e84d46d98ef09bbf80ba45))
* **pyproject.toml:** Adding missing console scripts/entry points from setup.cfg to pyproject.toml ([5fa8b07](https://github.com/chime-sps/pipeline_batch_db/commit/5fa8b075566ba0f21b908c852398e5915711f69b))

## 0.1.0 (2023-07-18)


### Bug Fixes

* 30: include package config file in the install ([#32](https://github.com/chime-sps/pipeline_batch_db/issues/32)) ([80ac2d3](https://github.com/chime-sps/pipeline_batch_db/commit/80ac2d3ee2c3e474a30ebcb867c120d0ad8d4732))
* Correctly construct the default value of the date to process ([#39](https://github.com/chime-sps/pipeline_batch_db/issues/39)) ([906b198](https://github.com/chime-sps/pipeline_batch_db/commit/906b198cafc39acfd0602a7c61165f31dadd841b)), closes [#37](https://github.com/chime-sps/pipeline_batch_db/issues/37)
* find the pointings map via beamformer.utilities.common ([1191240](https://github.com/chime-sps/pipeline_batch_db/commit/1191240ee07e4607533f6e86f9c5098112559338))
* Handle NoSuchPointingError from the pointing strategist ([#38](https://github.com/chime-sps/pipeline_batch_db/issues/38)) ([c011cc5](https://github.com/chime-sps/pipeline_batch_db/commit/c011cc5a6d65d53c6ef2b331d107527c5d715ee4))
* **poetry.lock:** Updating software environment to be consistent with other repos ([dc7ecd2](https://github.com/chime-sps/pipeline_batch_db/commit/dc7ecd2d5424ef922f790c43c44323b66b3db5ae))
* use the new `ps-processes` APIs and paths ([#27](https://github.com/chime-sps/pipeline_batch_db/issues/27)) ([07447c9](https://github.com/chime-sps/pipeline_batch_db/commit/07447c94fc049f16e5f72f136aa82a8fd0d6260e))
* use the new archive path structure ([#26](https://github.com/chime-sps/pipeline_batch_db/issues/26)) ([474c75e](https://github.com/chime-sps/pipeline_batch_db/commit/474c75ee3c606f9c74d713368180a0d7f517e30a))
