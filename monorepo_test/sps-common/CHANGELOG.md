# Changelog

## [1.1.0](https://github.com/chime-sps/sps-common/compare/v1.0.2...v1.1.0) (2023-10-19)


### Features

* **plot_candidates-script:** Added a plot_candidates script for conv… ([#147](https://github.com/chime-sps/sps-common/issues/147)) ([bb40045](https://github.com/chime-sps/sps-common/commit/bb400453512ccc684d4cdc230eb7783aff111b6e))
* **plot_candidates:** now uses multiprocessing for candidate plotting ([#144](https://github.com/chime-sps/sps-common/issues/144)) ([c8ab029](https://github.com/chime-sps/sps-common/commit/c8ab02913ff3968c51c0d0ec9d2350bc5bdb9701))


### Bug Fixes

* **plot_candidates:** prevent error message when folder exists ([#146](https://github.com/chime-sps/sps-common/issues/146)) ([b67da9f](https://github.com/chime-sps/sps-common/commit/b67da9fdf73f34218c3495572524b92653965c03))

## [1.0.2](https://github.com/chime-sps/sps-common/compare/v1.0.1...v1.0.2) (2023-07-18)


### Bug Fixes

* **ci:** remove onsite tests ([7b8a0ec](https://github.com/chime-sps/sps-common/commit/7b8a0ec65b1a1900413dbf5d64dd80eb7cdee3b2))
* **poetry.lock:** Updating software environment to be consistent with other repos ([4c3bb1a](https://github.com/chime-sps/sps-common/commit/4c3bb1afcff0a6f2b2b81dca6d1ff5619ed730cb))

## [1.0.1](https://github.com/chime-sps/sps-common/compare/v1.0.0...v1.0.1) (2023-06-28)


### Bug Fixes

* **poetry.lock:** Updating lock file to fix versioning, and adding new steps to CI process, along with updating README accordingly ([f9b698e](https://github.com/chime-sps/sps-common/commit/f9b698ee7794f7362e41593d8d441e11a6c132e5))
* **poetry.lock:** Updating lock file to fix versioning, and fixing CI steps ([879e65a](https://github.com/chime-sps/sps-common/commit/879e65a89fa219225469ac10422d5ba338a1236e))
* **poetry.lock:** Updating lock file to fix versioning, and fixing CI steps ([879e65a](https://github.com/chime-sps/sps-common/commit/879e65a89fa219225469ac10422d5ba338a1236e))

## 0.1.0 (2023-06-22)


### Features

* **pyproject.toml:** Adding Poetry in place of setuptools ([2f698d9](https://github.com/chime-sps/sps-common/commit/2f698d98129fd0dd808ef38308d00777d1fb35ed))


### Bug Fixes

* **.pre-commit-config.yaml:** Adding missing pre-commit configuration file to start new environment ([864c870](https://github.com/chime-sps/sps-common/commit/864c870b765d5633cf8aa61b404a1d5cc633937a))
* add conversion and validators for interfaces in beamformer ([#70](https://github.com/chime-sps/sps-common/issues/70)) ([f6a4100](https://github.com/chime-sps/sps-common/commit/f6a4100aae4798dee945af49a80d32f2ad1580d1))
* add conversion and validators for interfaces in ps-processes ([#63](https://github.com/chime-sps/sps-common/issues/63)) ([568e30f](https://github.com/chime-sps/sps-common/commit/568e30f2bf057e9606bb0deb4df91530d6cbd458))
* convert classification and know_source to objects when reading MultiPointingCandidate file ([#83](https://github.com/chime-sps/sps-common/issues/83)) ([3144f8a](https://github.com/chime-sps/sps-common/commit/3144f8a5d9dc1083f30901010af5cad4a1a59a14))
* convert encoded search algorithm in saved files back to SearchAlgorithm on reading ([#82](https://github.com/chime-sps/sps-common/issues/82)) ([aa7e155](https://github.com/chime-sps/sps-common/commit/aa7e1551c2047dea11b4da95a694a1fb78a0eb67))
* importing Enums in multi_pointing interfaces ([#67](https://github.com/chime-sps/sps-common/issues/67)) ([a9d03b9](https://github.com/chime-sps/sps-common/commit/a9d03b9797622d6a629fc7035e9648a7d8f87628))
* missing import of conversion function used in RFI interface read method ([#52](https://github.com/chime-sps/sps-common/issues/52)) ([f55afde](https://github.com/chime-sps/sps-common/commit/f55afde8134aa0043a4be223236c96a0c7134d1d))
* Move CandidateClassification and KnownSource labels from sps_databases ([#66](https://github.com/chime-sps/sps-common/issues/66)) ([5080c7d](https://github.com/chime-sps/sps-common/commit/5080c7dcc7766fc7bfd985a12fe44b025234b8d7)), closes [#57](https://github.com/chime-sps/sps-common/issues/57)
* typo and missing import in RFI interface ([#53](https://github.com/chime-sps/sps-common/issues/53)) ([0bb65cd](https://github.com/chime-sps/sps-common/commit/0bb65cd8b93bc288c68bfd420ed6e3bb97ce9709))
