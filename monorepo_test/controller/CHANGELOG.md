# Changelog

## [0.1.3](https://github.com/chime-sps/controller/compare/v0.1.2...v0.1.3) (2023-07-19)


### Bug Fixes

* **Dockerfile:** Adding new Dockerfile, and triggering new release to add new Docker Image to Dockerhub ([8a82267](https://github.com/chime-sps/controller/commit/8a82267c7a40d64c2909d150171f22b4ab5a4d2d))

## [0.1.2](https://github.com/chime-sps/controller/compare/v0.1.1...v0.1.2) (2023-07-18)


### Bug Fixes

* **continuous-deployment.yml:** Fix typo and trigger new DockerHub image creation ([aa5bc27](https://github.com/chime-sps/controller/commit/aa5bc274a3603d6ee1b2e01f3c0ed9b6403fd610))

## [0.1.1](https://github.com/chime-sps/controller/compare/v0.1.0...v0.1.1) (2023-07-18)


### Bug Fixes

* **base-dir:** changed sps data dir to /sps-archiver/chime/sps/raw ([22e7315](https://github.com/chime-sps/controller/commit/22e7315851226c0345c91576266b0a2b6daa5162))
* **cli:** fix for catching errors ([c8fb8b7](https://github.com/chime-sps/controller/commit/c8fb8b735a63f25ebc5c7e53c0eb2f753e11059f))
* **cli:** improved run cmd ([f889d20](https://github.com/chime-sps/controller/commit/f889d20d8050cb99af68c6f2af60ea3e93627374))
* **cli:** working to auto-catch errors when cli fails ([1baab04](https://github.com/chime-sps/controller/commit/1baab0428e989c759cb2db8ea2f7b490cea74b43))
* **updater:** fix for downgrading beam bit depth ([301e9be](https://github.com/chime-sps/controller/commit/301e9bec417a3341c688586d6c06410e5fdd841e))

## 0.1.0 (2022-12-22)


### Features

* **precommit:** adding precommit hooks ([#25](https://github.com/chime-sps/controller/issues/25)) ([6e2d355](https://github.com/chime-sps/controller/commit/6e2d355867e680d37dd3681b58aad88563b8133a))
* schedule calculation and pointing updates ([88f69ef](https://github.com/chime-sps/controller/commit/88f69ef8f165e6831c392f83ab85d675d228b224))
* **workflows/release-please.yml:** setting up release please ([#28](https://github.com/chime-sps/controller/issues/28)) ([5051c32](https://github.com/chime-sps/controller/commit/5051c32f128413e4ac0324994c6594068fd1991c))


### Bug Fixes

* compute the pointing schedule with enough overlap between batches ([#8](https://github.com/chime-sps/controller/issues/8)) ([750cf28](https://github.com/chime-sps/controller/commit/750cf2825c405fcc1e2c2e1a2f12c03988261318))
* **imports:** fixed circular imports ([490f047](https://github.com/chime-sps/controller/commit/490f04760dd2588594fa85fb5d9b740bee36031a))
* **layout:** fixed project layout ([8a44452](https://github.com/chime-sps/controller/commit/8a4445260383d9a2c70d327ab39160451c0d0342))
* **test.yml:** fixing test workflow to install poetry dependencies an… ([#30](https://github.com/chime-sps/controller/issues/30)) ([956f072](https://github.com/chime-sps/controller/commit/956f0720d8bfe78c68d3b3a48bc6ae14fab420e6))
* **tests:** fixed error with test setup ([e5fedba](https://github.com/chime-sps/controller/commit/e5fedba14f5782f02beb89efd246c6f99597e10e))
* **updater:** syntax error ([9a29811](https://github.com/chime-sps/controller/commit/9a298113e3611afb4108ba6c1346bdc4e8f0ba5e))
* **wip:** doing things ([84f9f01](https://github.com/chime-sps/controller/commit/84f9f01f42b53713c95e32b07dfb2e436eb1846d))
