# Changelog

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
