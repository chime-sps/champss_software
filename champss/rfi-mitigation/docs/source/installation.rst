.. _installation:

Installation
------------

To install the latest release of chime-frb-api, you can use ``pip``::

	pip install chime-frb-api

Alternatively, you can use ``pip`` to install the development version directly from github::

	pip install git+https://github.com/CHIMEFRB/frb-api.git

Another option would be to to clone the `github repository <https://github.com/CHIMEFRB/frb-api>`_ and install from your local copy::

	git clone https://github.com/CHIMEFRB/frb-api.git
	cd frb-api
	pip install .


Dependencies
~~~~~~~~~~~~

- Python 3.6+

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

- `requests <https://requests.readthedocs.io/en/master/>`_ (>2.22)
- `pyjwt <https://pyjwt.readthedocs.io/en/latest/>`_ (>1.7)
- `python-dateutil <https://dateutil.readthedocs.io/en/stable/>`_ (>2.8)

Developer dependencies
^^^^^^^^^^^^^^^^^^^^^^

- `pytest <https://docs.pytest.org/en/latest/>`_
- `pytest-cov <https://github.com/pytest-dev/pytest-cov>`_
- `coverage <https://coverage.readthedocs.io/en/coverage-5.0.3/>`_
- `pyyaml <https://github.com/yaml/pyyaml>`_
- `sphinx <https://www.sphinx-doc.org/en/master/>`_
- `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html>`_
- `sphinx_bootstrap_theme <https://github.com/ryan-roemer/sphinx-bootstrap-theme>`_

Bugs
~~~~

Please report any bugs you encounter through the github `issue tracker <https://github.com/CHIMEFRB/frb-api/issues>`_ . It will be most helpful to
include a reproducible example of the issue with information on expected behavior. It is difficult to debug any issues without knowing the versions of `chime-frb-api` and the backend version you are connected to. These can be found through:

.. code-block:: python

   from chime_frb_api import __version__
   from chime_frb_api.backends import frb_master
   master = frb_master.FRBMaster(base_url="https://frb.chimenet.ca/frb-master")
   print("Backend Version: {}".format(master.version()))
   print("API Version: {}".format(__version__))


Developer Resources
~~~~~~~~~~~~~~~~~~~

To contribute to the code base, visit the :doc:`developer section<developer>`.

