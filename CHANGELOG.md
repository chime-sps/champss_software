# Changelog

## [0.15.0](https://github.com/chime-sps/champss_software/compare/v0.14.0...v0.15.0) (2026-01-16)


### Features

* filter at 9 sigma for website ([#203](https://github.com/chime-sps/champss_software/issues/203)) ([3a8b8ca](https://github.com/chime-sps/champss_software/commit/3a8b8ca9435c2c74d298fd9d12beaec2769fe956))
* Run stack search ([#208](https://github.com/chime-sps/champss_software/issues/208)) ([09468c2](https://github.com/chime-sps/champss_software/commit/09468c2bf6722aaca4fc8e421a00ce3e208a1b71))


### Bug Fixes

* Fix processing bugs ([#201](https://github.com/chime-sps/champss_software/issues/201)) ([5f55277](https://github.com/chime-sps/champss_software/commit/5f55277c960a2ff9070dae3e5608ccd728b9ce01))

## [0.14.0](https://github.com/chime-sps/champss_software/compare/v0.13.0...v0.14.0) (2025-12-23)


### Features

* Allow controller to write fixed nchan and use multiple mounts ([#159](https://github.com/chime-sps/champss_software/issues/159)) ([717d28f](https://github.com/chime-sps/champss_software/commit/717d28f2df5f910619e871e712517d064b3ae10f))
* Allow setting datpath in multiday folding ([#163](https://github.com/chime-sps/champss_software/issues/163)) ([57b7b59](https://github.com/chime-sps/champss_software/commit/57b7b590a7a0f00c3e5bf718974c59d69e86e51d))
* change default database server to sps-archiver1 ([#171](https://github.com/chime-sps/champss_software/issues/171)) ([a3d002d](https://github.com/chime-sps/champss_software/commit/a3d002db74a56bafdddcf93a7055bddbabc2bc72))
* Changed search parameters for holiday processing run ([#199](https://github.com/chime-sps/champss_software/issues/199)) ([af6b674](https://github.com/chime-sps/champss_software/commit/af6b674dc7c73d9c6f484a568a5c9a789add75dd))
* enable saving medians to database ([#160](https://github.com/chime-sps/champss_software/issues/160)) ([4fdd3ae](https://github.com/chime-sps/champss_software/commit/4fdd3aeaa7a207c53b03da942e68fbdfad6f6b8f))
* Enable saving rednoise ([#149](https://github.com/chime-sps/champss_software/issues/149)) ([e62f672](https://github.com/chime-sps/champss_software/commit/e62f672a6c6956b1253102dbc1a4a984a34fa00e))
* Faster loading of injection files, stricter injection identification and loading of empty candidate files ([#158](https://github.com/chime-sps/champss_software/issues/158)) ([595cddd](https://github.com/chime-sps/champss_software/commit/595cddd069ed0466bed03d8e6041fb730e6ca49c))
* Filter known pulsars along arc ([#193](https://github.com/chime-sps/champss_software/issues/193)) ([e0775fb](https://github.com/chime-sps/champss_software/commit/e0775fbc3fe5fbf6f20f3cdb7cf8adf0e9972bd1))
* Filter known source f0 nan in sifter ([#194](https://github.com/chime-sps/champss_software/issues/194)) ([88641b5](https://github.com/chime-sps/champss_software/commit/88641b56d0d01166ea6c906242f335af8c6bf00a))
* Injection patches and cleaning for realtime processing ([#198](https://github.com/chime-sps/champss_software/issues/198)) ([f10a9cd](https://github.com/chime-sps/champss_software/commit/f10a9cd6e3df5f845a060f586743ab2d0e58cf67))
* New scheduling and other changes ([#188](https://github.com/chime-sps/champss_software/issues/188)) ([cf46388](https://github.com/chime-sps/champss_software/commit/cf463886981ad627d40b2658c2babd1dbd12815b))
* Rewrite frequency ranges ([#162](https://github.com/chime-sps/champss_software/issues/162)) ([c183586](https://github.com/chime-sps/champss_software/commit/c1835860118ef7152777ab94da552ea3851f38ce))


### Bug Fixes

* Allow sps-common to be installed as a dependency ([#165](https://github.com/chime-sps/champss_software/issues/165)) ([06f0218](https://github.com/chime-sps/champss_software/commit/06f02187847595934add5ddc0f87d2b8bbcb9ee9))
* Change build node ([#187](https://github.com/chime-sps/champss_software/issues/187)) ([543cfec](https://github.com/chime-sps/champss_software/commit/543cfec7e04785a0e6cedd974d74abd83b18ea41))

## [0.13.0](https://github.com/chime-sps/champss_software/compare/v0.12.1...v0.13.0) (2025-05-16)


### Features

* **continuous-integration.yml:** Make our re-usable GitHub Actions workflows use a local path for ease of development branch testing ([#155](https://github.com/chime-sps/champss_software/issues/155)) ([2dd6f6d](https://github.com/chime-sps/champss_software/commit/2dd6f6da523b297156f7954fd0076408e23fdbc6))
* Enable public install ([#154](https://github.com/chime-sps/champss_software/issues/154)) ([1535e35](https://github.com/chime-sps/champss_software/commit/1535e35ec2ed7748e310a4118e0385ef92940f79))


### Bug Fixes

* Add Slack token from GitHub secrets to Docker image build and use in scheduler module ([#145](https://github.com/chime-sps/champss_software/issues/145)) ([92e0ecb](https://github.com/chime-sps/champss_software/commit/92e0ecb52bacce3369f7940a5b6cf18760b27957))
* continue to next day when no processes available ([#147](https://github.com/chime-sps/champss_software/issues/147)) ([39a3e16](https://github.com/chime-sps/champss_software/commit/39a3e1679e4d56065653e741e5f192fd4a9c39ef))
* **Dockerfile:** version lock dspsr to working popular version ([#150](https://github.com/chime-sps/champss_software/issues/150)) ([af96f8a](https://github.com/chime-sps/champss_software/commit/af96f8a1c0a379bd0639e6f4a025d2ca29ddf1ea))
* enable foldpath in multiday fold ([#151](https://github.com/chime-sps/champss_software/issues/151)) ([04afdc2](https://github.com/chime-sps/champss_software/commit/04afdc2f5097ccf1ad2f7ad00bf1c5b5ccd9357d))
* fix creation of all_processes list ([#148](https://github.com/chime-sps/champss_software/issues/148)) ([261b069](https://github.com/chime-sps/champss_software/commit/261b069cc8fd83f05cc66ed303d4fba97a92f3aa))
* Fix file integrity check and delete remp files ([#141](https://github.com/chime-sps/champss_software/issues/141)) ([d1197bd](https://github.com/chime-sps/champss_software/commit/d1197bdab13cf9d3db75ab929fa988ff6d032daa))
* Fix folding bugs ([#144](https://github.com/chime-sps/champss_software/issues/144)) ([238f89e](https://github.com/chime-sps/champss_software/commit/238f89e9f0a47ba5492b1da55a408302c903882a))
* Fix slack and fold ([#139](https://github.com/chime-sps/champss_software/issues/139)) ([1c1c9d8](https://github.com/chime-sps/champss_software/commit/1c1c9d8fc428efcff6f18d281c876e9e7e91617d))


### Documentation

* removed old frb-api docs from rfi-mitigation ([#143](https://github.com/chime-sps/champss_software/issues/143)) ([5be2f74](https://github.com/chime-sps/champss_software/commit/5be2f748957a05eabdcb84c83dce6bcfb17feb54))

## [0.12.1](https://github.com/chime-sps/champss_software/compare/v0.12.0...v0.12.1) (2025-04-17)


### Bug Fixes

* bump version ([e3415a0](https://github.com/chime-sps/champss_software/commit/e3415a090324f3ab48f8929fb45b3d24313bc6ed))
* **pyproject.toml:** Dummy commit to trigger new release from manually created tag ([4830942](https://github.com/chime-sps/champss_software/commit/4830942f46c78c5256da5a97d66e1d126a31e8d6))

## [0.2.0](https://github.com/chime-sps/champss_software/compare/v0.1.0...v0.2.0) (2025-04-17)


### Features

* Add batch wrapper for spsctl ([#105](https://github.com/chime-sps/champss_software/issues/105)) ([28ccd79](https://github.com/chime-sps/champss_software/commit/28ccd794fc55348580e6fae9b75133aa41317d93))
* add new candidate writing method ([#115](https://github.com/chime-sps/champss_software/issues/115)) ([19587e4](https://github.com/chime-sps/champss_software/commit/19587e41f5505c05ebb0091a803dbe3d997c0fe3))
* allow custom basepath, remove redundant search for files, move benchmark to site ([#67](https://github.com/chime-sps/champss_software/issues/67)) ([43d50a0](https://github.com/chime-sps/champss_software/commit/43d50a0dc47d53286165f347153359d3a6e4ee38))
* Allow proper prediction of injection sigma ([#74](https://github.com/chime-sps/champss_software/issues/74)) ([0d763d9](https://github.com/chime-sps/champss_software/commit/0d763d98d9ff4a9d01d37c10eee0bbcc116361b8))
* **continuous-integration.yml:** Adding new GitHub Actions ([3665af5](https://github.com/chime-sps/champss_software/commit/3665af506490530b1a24ad078e95a1b028d2ad36))
* **continuous-integration.yml:** Plot candiate plots in benchmark and enable manual run ([435e561](https://github.com/chime-sps/champss_software/commit/435e561ff6169b08cdc76cde959cef639543a45b))
* **controller:** Add basepath on L1 nodes as a Click CLI parameter ([48c3f21](https://github.com/chime-sps/champss_software/commit/48c3f211c22cff478c92646a71c1316a0dcb6100))
* Enable finer control of processing and improve RFI ([#131](https://github.com/chime-sps/champss_software/issues/131)) ([9fbc944](https://github.com/chime-sps/champss_software/commit/9fbc944383b1d2209098b4d49a53987e9cd5b1ea))
* Filter detections before clustering ([#80](https://github.com/chime-sps/champss_software/issues/80)) ([ba931ff](https://github.com/chime-sps/champss_software/commit/ba931ffb281d179aa4e6802b8c7f62198dad282d))
* **image.yml:** Add support for new self-hosted Docker Image registry ([2a8e79e](https://github.com/chime-sps/champss_software/commit/2a8e79e8a573a2b316b3defdb122c9b359e7f943))
* **known_source_sifter.py:** Add quick sanity check before running ks filter ([#23](https://github.com/chime-sps/champss_software/issues/23)) ([e736453](https://github.com/chime-sps/champss_software/commit/e7364538d5cc63c1d02cdb7137a90a0217f9e950))
* **pipeline.py:** Allow alternate config name ([#10](https://github.com/chime-sps/champss_software/issues/10)) ([8e0f8b5](https://github.com/chime-sps/champss_software/commit/8e0f8b54b41ddb6bc01802bbf27bb58b0183dae1))
* **plot_candidate:** single day fold cand plot upgrade ([#109](https://github.com/chime-sps/champss_software/issues/109)) ([b91ade4](https://github.com/chime-sps/champss_software/commit/b91ade484de6e7ebf4532381a02398242bc429ae))
* Predict sigma of injection ([#58](https://github.com/chime-sps/champss_software/issues/58)) ([7519b8f](https://github.com/chime-sps/champss_software/commit/7519b8fb140bd34e331417e3ab7d65e9ffc8bff9))
* Refine clustering ([#60](https://github.com/chime-sps/champss_software/issues/60)) ([f4946d1](https://github.com/chime-sps/champss_software/commit/f4946d1b8dbf85b340402870e426c3e6b7e78276))
* Restructure multi-pointing candidate writing ([#117](https://github.com/chime-sps/champss_software/issues/117)) ([0183050](https://github.com/chime-sps/champss_software/commit/0183050a94a2f75840ebdc745a48f2e0481a7415))
* Run monthly search without access to database ([#78](https://github.com/chime-sps/champss_software/issues/78)) ([8d19986](https://github.com/chime-sps/champss_software/commit/8d19986ea7936886728f503f4bcd194c194390f7))
* **run-benchmark.sh:** Refine benchmark ([#15](https://github.com/chime-sps/champss_software/issues/15)) ([f21b640](https://github.com/chime-sps/champss_software/commit/f21b64051f4dcf84ac48b5e120fbd51ec304bb67))
* Speedup process creation ([#111](https://github.com/chime-sps/champss_software/issues/111)) ([e3cc9d3](https://github.com/chime-sps/champss_software/commit/e3cc9d3ac0fd5e3696c3df4cd204350f349e2ea3))
* **sps_multi_pointing:** Enable position filtering and setting of used metric in spsmp ([#26](https://github.com/chime-sps/champss_software/issues/26)) ([3cf2c68](https://github.com/chime-sps/champss_software/commit/3cf2c6886c6625d7879046d3552c643b31434585))
* Update birdies.yaml, expand RFI filtering method and enable birdie report ([#83](https://github.com/chime-sps/champss_software/issues/83)) ([592cd50](https://github.com/chime-sps/champss_software/commit/592cd500734364a240bd170502880ff95ff695ef))
* **workflow.py:** Adding improvements to scheduling ([a1d39e8](https://github.com/chime-sps/champss_software/commit/a1d39e8ea5fb021af43d796fa26963e0e3756a82))
* **workflow.py:** Move workflow.py to its own module ([b87e36f](https://github.com/chime-sps/champss_software/commit/b87e36fd44065633d45934e1174aea8edaa7f3c6))


### Bug Fixes

* benchmark mount ([#99](https://github.com/chime-sps/champss_software/issues/99)) ([7ea3b96](https://github.com/chime-sps/champss_software/commit/7ea3b964ea8007a34278ebe5cddf62e95290bd82))
* candidate writing during benchmark and datpath import ([ffc3175](https://github.com/chime-sps/champss_software/commit/ffc3175f2791f2752bdd9ee012756dbc78dfd084))
* **common.py:** automatic loading of beam-model files ([#9](https://github.com/chime-sps/champss_software/issues/9)) ([7b5e6f9](https://github.com/chime-sps/champss_software/commit/7b5e6f9fdd0a5a68edb71600de976c1000d08979))
* Enable exception for get_observation and update_observation when obs_id does not exist ([#103](https://github.com/chime-sps/champss_software/issues/103)) ([8f257e9](https://github.com/chime-sps/champss_software/commit/8f257e9ddb9a27bf8f78f6ca46548a3513a88807))
* file reading when full path if given ([#84](https://github.com/chime-sps/champss_software/issues/84)) ([df48f50](https://github.com/chime-sps/champss_software/commit/df48f507e80fe0e26bb0bbb0b329581610e2dc2f))
* Fix typo in .ram_requirement() ([#132](https://github.com/chime-sps/champss_software/issues/132)) ([a4df8b8](https://github.com/chime-sps/champss_software/commit/a4df8b80b776641ac19c0e866b72860d7554c189))
* Fixed benchmark ([#64](https://github.com/chime-sps/champss_software/issues/64)) ([23b8309](https://github.com/chime-sps/champss_software/commit/23b83097a397d1c5231d1ea019c37fb5f79ea749))
* fixed stopping beams for high number of beams ([#113](https://github.com/chime-sps/champss_software/issues/113)) ([fa53036](https://github.com/chime-sps/champss_software/commit/fa53036716517b1da01264d508661aa989b1bd3a))
* **grouper.py:** Disallow delta_ra values above 180 ([#18](https://github.com/chime-sps/champss_software/issues/18)) ([11534b6](https://github.com/chime-sps/champss_software/commit/11534b673b4f46d178295ba759ab1bb8fdb053b8))
* injection PR and precommit files ([e215ee6](https://github.com/chime-sps/champss_software/commit/e215ee67625a148f12cf885522d510ff4c921783))
* ks filter for single day pipeline ([#59](https://github.com/chime-sps/champss_software/issues/59)) ([3f620b1](https://github.com/chime-sps/champss_software/commit/3f620b18a48f4321a411081de29ee5cf0224453a))
* Plot refinements ([#100](https://github.com/chime-sps/champss_software/issues/100)) ([f29f36d](https://github.com/chime-sps/champss_software/commit/f29f36ddfa072b01201f81e06c8355b29e9310c0))
* **processing.py:** Update all refrences of sps-archiver to sps-archiver1 ([9375231](https://github.com/chime-sps/champss_software/commit/93752312cb90d7cc368063748a27ecb4888910a1))
* **pyproject.toml:** replace chime-frb-api with workflow-core package ([7a507bd](https://github.com/chime-sps/champss_software/commit/7a507bd495d40efeddbda7b3d122589839882953))
* **pyproject.toml:** Try bumping version to 0.12.0 ([d4e4f55](https://github.com/chime-sps/champss_software/commit/d4e4f554e8bdb6381d77c6e358e736706f87fa36))
* reverting spshuff import order ([#50](https://github.com/chime-sps/champss_software/issues/50)) ([7d611cd](https://github.com/chime-sps/champss_software/commit/7d611cdd530a92be68132b9e871a8251f4452945))
* Update astropy and minimum python version ([#87](https://github.com/chime-sps/champss_software/issues/87)) ([6b266a0](https://github.com/chime-sps/champss_software/commit/6b266a0b93e63e8d4acb2d66324f8f8eb53473bb))
* **workflow.py:** Adding log dumping of multipointing containers before cleanup and password obfuscation ([fb42fe0](https://github.com/chime-sps/champss_software/commit/fb42fe00756edfaf09b33bb6edbc348ac9f08d47))
* **workflow.py:** Constrain /data/ mounts to point directly to sps-archiver1.chime ([f5cb5a5](https://github.com/chime-sps/champss_software/commit/f5cb5a53a2788225af80c49475e1d21e59eb59c8))
* **workflow.py:** Fix bug when microseconds is not defined in Docker Service CreatedAt field ([55366ca](https://github.com/chime-sps/champss_software/commit/55366caff70da5c8aeefdc8d758c13acec436ed3))
* **workflow.py:** Read container log generator into file ([df97c46](https://github.com/chime-sps/champss_software/commit/df97c462a3e8da8419d33bef6150c96fb92f3e79))


### Documentation

* More explanations about setting up a database ([f80ca89](https://github.com/chime-sps/champss_software/commit/f80ca89ce325e6320960a9811164cdba11bcefb3))

## 0.1.0 (2025-04-16)


### Features

* Add batch wrapper for spsctl ([#105](https://github.com/chime-sps/champss_software/issues/105)) ([28ccd79](https://github.com/chime-sps/champss_software/commit/28ccd794fc55348580e6fae9b75133aa41317d93))
* add new candidate writing method ([#115](https://github.com/chime-sps/champss_software/issues/115)) ([19587e4](https://github.com/chime-sps/champss_software/commit/19587e41f5505c05ebb0091a803dbe3d997c0fe3))
* allow custom basepath, remove redundant search for files, move benchmark to site ([#67](https://github.com/chime-sps/champss_software/issues/67)) ([43d50a0](https://github.com/chime-sps/champss_software/commit/43d50a0dc47d53286165f347153359d3a6e4ee38))
* Allow proper prediction of injection sigma ([#74](https://github.com/chime-sps/champss_software/issues/74)) ([0d763d9](https://github.com/chime-sps/champss_software/commit/0d763d98d9ff4a9d01d37c10eee0bbcc116361b8))
* **continuous-integration.yml:** Adding new GitHub Actions ([3665af5](https://github.com/chime-sps/champss_software/commit/3665af506490530b1a24ad078e95a1b028d2ad36))
* **continuous-integration.yml:** Plot candiate plots in benchmark and enable manual run ([435e561](https://github.com/chime-sps/champss_software/commit/435e561ff6169b08cdc76cde959cef639543a45b))
* **controller:** Add basepath on L1 nodes as a Click CLI parameter ([48c3f21](https://github.com/chime-sps/champss_software/commit/48c3f211c22cff478c92646a71c1316a0dcb6100))
* Enable finer control of processing and improve RFI ([#131](https://github.com/chime-sps/champss_software/issues/131)) ([9fbc944](https://github.com/chime-sps/champss_software/commit/9fbc944383b1d2209098b4d49a53987e9cd5b1ea))
* Filter detections before clustering ([#80](https://github.com/chime-sps/champss_software/issues/80)) ([ba931ff](https://github.com/chime-sps/champss_software/commit/ba931ffb281d179aa4e6802b8c7f62198dad282d))
* **image.yml:** Add support for new self-hosted Docker Image registry ([2a8e79e](https://github.com/chime-sps/champss_software/commit/2a8e79e8a573a2b316b3defdb122c9b359e7f943))
* **known_source_sifter.py:** Add quick sanity check before running ks filter ([#23](https://github.com/chime-sps/champss_software/issues/23)) ([e736453](https://github.com/chime-sps/champss_software/commit/e7364538d5cc63c1d02cdb7137a90a0217f9e950))
* **pipeline.py:** Allow alternate config name ([#10](https://github.com/chime-sps/champss_software/issues/10)) ([8e0f8b5](https://github.com/chime-sps/champss_software/commit/8e0f8b54b41ddb6bc01802bbf27bb58b0183dae1))
* **plot_candidate:** single day fold cand plot upgrade ([#109](https://github.com/chime-sps/champss_software/issues/109)) ([b91ade4](https://github.com/chime-sps/champss_software/commit/b91ade484de6e7ebf4532381a02398242bc429ae))
* Predict sigma of injection ([#58](https://github.com/chime-sps/champss_software/issues/58)) ([7519b8f](https://github.com/chime-sps/champss_software/commit/7519b8fb140bd34e331417e3ab7d65e9ffc8bff9))
* Refine clustering ([#60](https://github.com/chime-sps/champss_software/issues/60)) ([f4946d1](https://github.com/chime-sps/champss_software/commit/f4946d1b8dbf85b340402870e426c3e6b7e78276))
* Restructure multi-pointing candidate writing ([#117](https://github.com/chime-sps/champss_software/issues/117)) ([0183050](https://github.com/chime-sps/champss_software/commit/0183050a94a2f75840ebdc745a48f2e0481a7415))
* Run monthly search without access to database ([#78](https://github.com/chime-sps/champss_software/issues/78)) ([8d19986](https://github.com/chime-sps/champss_software/commit/8d19986ea7936886728f503f4bcd194c194390f7))
* **run-benchmark.sh:** Refine benchmark ([#15](https://github.com/chime-sps/champss_software/issues/15)) ([f21b640](https://github.com/chime-sps/champss_software/commit/f21b64051f4dcf84ac48b5e120fbd51ec304bb67))
* Speedup process creation ([#111](https://github.com/chime-sps/champss_software/issues/111)) ([e3cc9d3](https://github.com/chime-sps/champss_software/commit/e3cc9d3ac0fd5e3696c3df4cd204350f349e2ea3))
* **sps_multi_pointing:** Enable position filtering and setting of used metric in spsmp ([#26](https://github.com/chime-sps/champss_software/issues/26)) ([3cf2c68](https://github.com/chime-sps/champss_software/commit/3cf2c6886c6625d7879046d3552c643b31434585))
* Update birdies.yaml, expand RFI filtering method and enable birdie report ([#83](https://github.com/chime-sps/champss_software/issues/83)) ([592cd50](https://github.com/chime-sps/champss_software/commit/592cd500734364a240bd170502880ff95ff695ef))
* **workflow.py:** Adding improvements to scheduling ([a1d39e8](https://github.com/chime-sps/champss_software/commit/a1d39e8ea5fb021af43d796fa26963e0e3756a82))
* **workflow.py:** Move workflow.py to its own module ([b87e36f](https://github.com/chime-sps/champss_software/commit/b87e36fd44065633d45934e1174aea8edaa7f3c6))


### Bug Fixes

* benchmark mount ([#99](https://github.com/chime-sps/champss_software/issues/99)) ([7ea3b96](https://github.com/chime-sps/champss_software/commit/7ea3b964ea8007a34278ebe5cddf62e95290bd82))
* candidate writing during benchmark and datpath import ([ffc3175](https://github.com/chime-sps/champss_software/commit/ffc3175f2791f2752bdd9ee012756dbc78dfd084))
* **common.py:** automatic loading of beam-model files ([#9](https://github.com/chime-sps/champss_software/issues/9)) ([7b5e6f9](https://github.com/chime-sps/champss_software/commit/7b5e6f9fdd0a5a68edb71600de976c1000d08979))
* Enable exception for get_observation and update_observation when obs_id does not exist ([#103](https://github.com/chime-sps/champss_software/issues/103)) ([8f257e9](https://github.com/chime-sps/champss_software/commit/8f257e9ddb9a27bf8f78f6ca46548a3513a88807))
* file reading when full path if given ([#84](https://github.com/chime-sps/champss_software/issues/84)) ([df48f50](https://github.com/chime-sps/champss_software/commit/df48f507e80fe0e26bb0bbb0b329581610e2dc2f))
* Fix typo in .ram_requirement() ([#132](https://github.com/chime-sps/champss_software/issues/132)) ([a4df8b8](https://github.com/chime-sps/champss_software/commit/a4df8b80b776641ac19c0e866b72860d7554c189))
* Fixed benchmark ([#64](https://github.com/chime-sps/champss_software/issues/64)) ([23b8309](https://github.com/chime-sps/champss_software/commit/23b83097a397d1c5231d1ea019c37fb5f79ea749))
* fixed stopping beams for high number of beams ([#113](https://github.com/chime-sps/champss_software/issues/113)) ([fa53036](https://github.com/chime-sps/champss_software/commit/fa53036716517b1da01264d508661aa989b1bd3a))
* **grouper.py:** Disallow delta_ra values above 180 ([#18](https://github.com/chime-sps/champss_software/issues/18)) ([11534b6](https://github.com/chime-sps/champss_software/commit/11534b673b4f46d178295ba759ab1bb8fdb053b8))
* injection PR and precommit files ([e215ee6](https://github.com/chime-sps/champss_software/commit/e215ee67625a148f12cf885522d510ff4c921783))
* ks filter for single day pipeline ([#59](https://github.com/chime-sps/champss_software/issues/59)) ([3f620b1](https://github.com/chime-sps/champss_software/commit/3f620b18a48f4321a411081de29ee5cf0224453a))
* Plot refinements ([#100](https://github.com/chime-sps/champss_software/issues/100)) ([f29f36d](https://github.com/chime-sps/champss_software/commit/f29f36ddfa072b01201f81e06c8355b29e9310c0))
* **processing.py:** Update all refrences of sps-archiver to sps-archiver1 ([9375231](https://github.com/chime-sps/champss_software/commit/93752312cb90d7cc368063748a27ecb4888910a1))
* **pyproject.toml:** replace chime-frb-api with workflow-core package ([7a507bd](https://github.com/chime-sps/champss_software/commit/7a507bd495d40efeddbda7b3d122589839882953))
* reverting spshuff import order ([#50](https://github.com/chime-sps/champss_software/issues/50)) ([7d611cd](https://github.com/chime-sps/champss_software/commit/7d611cdd530a92be68132b9e871a8251f4452945))
* Update astropy and minimum python version ([#87](https://github.com/chime-sps/champss_software/issues/87)) ([6b266a0](https://github.com/chime-sps/champss_software/commit/6b266a0b93e63e8d4acb2d66324f8f8eb53473bb))
* **workflow.py:** Adding log dumping of multipointing containers before cleanup and password obfuscation ([fb42fe0](https://github.com/chime-sps/champss_software/commit/fb42fe00756edfaf09b33bb6edbc348ac9f08d47))
* **workflow.py:** Constrain /data/ mounts to point directly to sps-archiver1.chime ([f5cb5a5](https://github.com/chime-sps/champss_software/commit/f5cb5a53a2788225af80c49475e1d21e59eb59c8))
* **workflow.py:** Fix bug when microseconds is not defined in Docker Service CreatedAt field ([55366ca](https://github.com/chime-sps/champss_software/commit/55366caff70da5c8aeefdc8d758c13acec436ed3))
* **workflow.py:** Read container log generator into file ([df97c46](https://github.com/chime-sps/champss_software/commit/df97c462a3e8da8419d33bef6150c96fb92f3e79))


### Documentation

* More explanations about setting up a database ([f80ca89](https://github.com/chime-sps/champss_software/commit/f80ca89ce325e6320960a9811164cdba11bcefb3))

## [0.11.0](https://github.com/chime-sps/champss_software/compare/v0.10.0...v0.11.0) (2025-04-08)


### Features

* Add batch wrapper for spsctl ([#105](https://github.com/chime-sps/champss_software/issues/105)) ([494118a](https://github.com/chime-sps/champss_software/commit/494118a8e4979edfa7f5dd39c9914a2e28e92084))
* add new candidate writing method ([#115](https://github.com/chime-sps/champss_software/issues/115)) ([1f80524](https://github.com/chime-sps/champss_software/commit/1f80524df35e523bc733d7656824e3681bcc8f15))
* Enable finer control of processing and improve RFI ([#131](https://github.com/chime-sps/champss_software/issues/131)) ([f6fac7d](https://github.com/chime-sps/champss_software/commit/f6fac7d21bbae47958212e1577941f80ffb558c0))
* **plot_candidate:** single day fold cand plot upgrade ([#109](https://github.com/chime-sps/champss_software/issues/109)) ([b835704](https://github.com/chime-sps/champss_software/commit/b835704608bce2e0675666a893a5fbae30fc4173))
* Restructure multi-pointing candidate writing ([#117](https://github.com/chime-sps/champss_software/issues/117)) ([a6323d6](https://github.com/chime-sps/champss_software/commit/a6323d61cd0140fd3df6a595304554dd58523c5c))
* Speedup process creation ([#111](https://github.com/chime-sps/champss_software/issues/111)) ([dc39604](https://github.com/chime-sps/champss_software/commit/dc3960483790856088eae7da7a0789d176364a18))


### Bug Fixes

* fixed stopping beams for high number of beams ([#113](https://github.com/chime-sps/champss_software/issues/113)) ([f3f112d](https://github.com/chime-sps/champss_software/commit/f3f112d2615244a64d696a77139aec41cf32bd3e))

## [0.10.0](https://github.com/chime-sps/champss_software/compare/v0.9.0...v0.10.0) (2025-01-31)


### Features

* **controller:** Add basepath on L1 nodes as a Click CLI parameter ([a8cefab](https://github.com/chime-sps/champss_software/commit/a8cefabacb63875a044dd5e6ab9fe22bff3b21ff))
* **image.yml:** Add support for new self-hosted Docker Image registry ([d0e9c26](https://github.com/chime-sps/champss_software/commit/d0e9c26af52ea4e06f9f3e56c1bc5cbbc88f9499))
* Update birdies.yaml, expand RFI filtering method and enable birdie report ([#83](https://github.com/chime-sps/champss_software/issues/83)) ([c3cb022](https://github.com/chime-sps/champss_software/commit/c3cb0229fc136f4188e673709251f5ba3c63c433))


### Bug Fixes

* benchmark mount ([#99](https://github.com/chime-sps/champss_software/issues/99)) ([9b2f9d8](https://github.com/chime-sps/champss_software/commit/9b2f9d855d6576713f7081377a83ad6c28b91aa9))
* Plot refinements ([#100](https://github.com/chime-sps/champss_software/issues/100)) ([d220027](https://github.com/chime-sps/champss_software/commit/d220027b656ed411fd3dfe9be21994ef8c1debe7))


### Documentation

* More explanations about setting up a database ([6f2153b](https://github.com/chime-sps/champss_software/commit/6f2153b48963330d338af36c0c9baa0757baefa9))

## [0.9.0](https://github.com/chime-sps/champss_software/compare/v0.8.0...v0.9.0) (2024-10-25)


### Features

* **workflow.py:** Move workflow.py to its own module ([337473d](https://github.com/chime-sps/champss_software/commit/337473d214ab98f4c6d5330a78dbb375224fb4ba))


### Bug Fixes

* **workflow.py:** Constrain /data/ mounts to point directly to sps-archiver1.chime ([f0539cf](https://github.com/chime-sps/champss_software/commit/f0539cf8543584e06146e3b6fe7df7f7a1d797f8))

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
