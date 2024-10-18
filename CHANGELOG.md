# Changelog

## [0.8.0](https://github.com/chime-sps/champss_software/compare/v0.7.0...v0.8.0) (2024-10-18)


### Features

* Filter detections before clustering ([#80](https://github.com/chime-sps/champss_software/issues/80)) ([f671484](https://github.com/chime-sps/champss_software/commit/f67148416db6568928e250588ee55b82d2eee0c1))


### Bug Fixes

* file reading when full path if given ([#84](https://github.com/chime-sps/champss_software/issues/84)) ([5c426e1](https://github.com/chime-sps/champss_software/commit/5c426e166fc368ce63745bac4a41f95898c22d01))
* Update astropy and minimum python version ([#87](https://github.com/chime-sps/champss_software/issues/87)) ([b91bd25](https://github.com/chime-sps/champss_software/commit/b91bd2503dd831bfdb80450fc6b44933d05666a3))

## [0.7.0](https://github.com/chime-sps/champss_software/compare/v0.6.1...v0.7.0) (2024-10-12)


### Features

* Allow proper prediction of injection sigma ([#74](https://github.com/chime-sps/champss_software/issues/74)) ([6f71fdf](https://github.com/chime-sps/champss_software/commit/6f71fdfac4d7aaf33a0c9138ebd00cf7cfcc6e82))
* Run monthly search without access to database ([#78](https://github.com/chime-sps/champss_software/issues/78)) ([d2419fc](https://github.com/chime-sps/champss_software/commit/d2419fc2d5c20c89eeb46ac387eb72e99f4ae695))


### Bug Fixes

* **processing.py:** Update all refrences of sps-archiver to sps-archiver1 ([ee1d88f](https://github.com/chime-sps/champss_software/commit/ee1d88f5c46b123f343f6cdb2afa22f0dcfc5208))

## [0.6.1](https://github.com/chime-sps/champss_software/compare/v0.6.0...v0.6.1) (2024-09-09)


### Bug Fixes

* candidate writing during benchmark and datpath import ([5befc95](https://github.com/chime-sps/champss_software/commit/5befc95bfd4e6e74738d838b0e31e5084fc89b36))

## [0.6.0](https://github.com/chime-sps/champss_software/compare/v0.5.0...v0.6.0) (2024-09-06)


### Features

* allow custom basepath, remove redundant search for files, move benchmark to site ([#67](https://github.com/chime-sps/champss_software/issues/67)) ([2b9db5f](https://github.com/chime-sps/champss_software/commit/2b9db5f508c57d006251f428dc26f1c2ccbb5fcd))
* Refine clustering ([#60](https://github.com/chime-sps/champss_software/issues/60)) ([6c9ff4c](https://github.com/chime-sps/champss_software/commit/6c9ff4cc89fe2374ce2ff93d47863fb81159ac5b))


### Bug Fixes

* Fixed benchmark ([#64](https://github.com/chime-sps/champss_software/issues/64)) ([e951714](https://github.com/chime-sps/champss_software/commit/e9517140884fe531dedf84c6545d981f93c20154))
* **pyproject.toml:** replace chime-frb-api with workflow-core package ([5500ba7](https://github.com/chime-sps/champss_software/commit/5500ba7fb9d4388659dbbe665db2cd65f773184e))

## [0.5.0](https://github.com/chime-sps/champss_software/compare/v0.4.0...v0.5.0) (2024-08-23)


### Features

* Predict sigma of injection ([#58](https://github.com/chime-sps/champss_software/issues/58)) ([281e982](https://github.com/chime-sps/champss_software/commit/281e9827aa14edfb948e114f585cd6ee998917da))


### Bug Fixes

* injection PR and precommit files ([091e317](https://github.com/chime-sps/champss_software/commit/091e317d5b2b07ed6525dd320dd5af5a415dcbb1))
* ks filter for single day pipeline ([#59](https://github.com/chime-sps/champss_software/issues/59)) ([33a4430](https://github.com/chime-sps/champss_software/commit/33a443091d583f263c8316466bac58c387e77dda))

## [0.4.0](https://github.com/chime-sps/champss_software/compare/v0.3.1...v0.4.0) (2024-07-22)


### Features

* **known_source_sifter.py:** Add quick sanity check before running ks filter ([#23](https://github.com/chime-sps/champss_software/issues/23)) ([c54395e](https://github.com/chime-sps/champss_software/commit/c54395e160b6b19cde0b5fc90a59eff6afd096ec))
* **sps_multi_pointing:** Enable position filtering and setting of used metric in spsmp ([#26](https://github.com/chime-sps/champss_software/issues/26)) ([647c851](https://github.com/chime-sps/champss_software/commit/647c851342b55a2979ae491c2b036b66d53b28aa))
* **workflow.py:** Adding improvements to scheduling ([e3616b1](https://github.com/chime-sps/champss_software/commit/e3616b18908b53750eadecb0bb5f5fc317099ad9))


### Bug Fixes

* reverting spshuff import order ([#50](https://github.com/chime-sps/champss_software/issues/50)) ([ead46fd](https://github.com/chime-sps/champss_software/commit/ead46fdbbddcc71b4e002e4fc458bfb6cc61c786))
* **workflow.py:** Fix bug when microseconds is not defined in Docker Service CreatedAt field ([8bf7297](https://github.com/chime-sps/champss_software/commit/8bf729710b801c3ea113a27b5ac722160fa7a6e6))

## [0.3.1](https://github.com/chime-sps/champss_software/compare/v0.3.0...v0.3.1) (2024-06-07)


### Bug Fixes

* **workflow.py:** Read container log generator into file ([a5ba7a3](https://github.com/chime-sps/champss_software/commit/a5ba7a3ce400225289ec30e020bef9650226e327))

## [0.3.0](https://github.com/chime-sps/champss_software/compare/v0.2.0...v0.3.0) (2024-06-06)


### Features

* **continuous-integration.yml:** Plot candiate plots in benchmark and enable manual run ([91202c5](https://github.com/chime-sps/champss_software/commit/91202c5a84333191861181e9a6120e05a49303a8))
* **run-benchmark.sh:** Refine benchmark ([#15](https://github.com/chime-sps/champss_software/issues/15)) ([70c494d](https://github.com/chime-sps/champss_software/commit/70c494d653d375f5a8311d3fb8872b9bd396fb46))


### Bug Fixes

* **grouper.py:** Disallow delta_ra values above 180 ([#18](https://github.com/chime-sps/champss_software/issues/18)) ([8a1e2b0](https://github.com/chime-sps/champss_software/commit/8a1e2b0051c7afe7ac9d4ad02ff03293da743765))
* **workflow.py:** Adding log dumping of multipointing containers before cleanup and password obfuscation ([e103f5c](https://github.com/chime-sps/champss_software/commit/e103f5c05c12b3d86987e219cb77ab461f992212))

## [0.2.0](https://github.com/chime-sps/champss_software/compare/v0.1.0...v0.2.0) (2024-05-24)


### Features

* **pipeline.py:** Allow alternate config name ([#10](https://github.com/chime-sps/champss_software/issues/10)) ([e830bfe](https://github.com/chime-sps/champss_software/commit/e830bfe22522bb40099f5eab3bca244643c183ee))


### Bug Fixes

* **common.py:** automatic loading of beam-model files ([#9](https://github.com/chime-sps/champss_software/issues/9)) ([fdf30e1](https://github.com/chime-sps/champss_software/commit/fdf30e1857eb1f66eee991f866a20fa31d8ec990))

## 0.1.0 (2024-05-15)


### Features

* **continuous-integration.yml:** Adding new GitHub Actions ([8799978](https://github.com/chime-sps/champss_software/commit/879997803b1b60d2231a76785b32d91cee760139))
