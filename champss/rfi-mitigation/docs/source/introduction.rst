.. _introduction:

Introduction
------------

chime-frb-api is a python library for connecting with the CHIME/FRB backends. It is built on top of the `requests <https://requests.readthedocs.io/en/master/>`_ python package and provides a secure access through the use of *JSON Web Tokens (JWT)*.

Here is some of the functionality access *chime-frb-api* offers:

- Realtime pipeline triggers
- Injection backend
- Verified astrophysical sources
- General software parameters
- Calibrated data products
- Support for inter-pipeline communication via distributors

chime-frb-api aims to keep *easy-of-use* for scientific developers as its core foundation. Here is an example of what that means:

.. code-block:: python

	from chime_frb_api.backends import frb_master
	master = frb_master.FRBMaster(base_url="https://frb.chimenet.ca/frb-master")
	master.events.get_event(9386707)
	[2020-03-01 16:48:17,265] INFO Authorization Status: None
	[2020-03-01 16:48:17,266] INFO Authorization Method: Username/Password
	Username: shiny
	Password: ###############
	[2020-03-01 16:48:22,617] INFO Authorization Result: Passed
	[2020-03-01 16:48:22,617] INFO Authorization Expiry: Sun Mar  1 17:18:22 2020

	{'beam_numbers': [166, 1166],
 	 'event_type': 'EXTRAGALACTIC',
 	 'fpga_time': 39927134208,
 	 id': 9386707,
 	 ...}

A few things happened here so lets break them down:

1. We connect to the *frb_master* backend from *chime-frb-api*

.. code-block:: python

	from chime_frb_api.backends import frb_master
	master = frb_master.FRBMaster(base_url="https://frb.chimenet.ca/frb-master")

Behind the scences, *chime-frb-api* uses the the `base_url` argument to connect to the CHIME/FRB DRAO Backend by  starting a `requests session <https://requests.readthedocs.io/en/master/user/advanced/#session-objects>`_ . 

2. We ask the Events API to fetch us the parameters for the event id `9386707`.

.. code-block:: python

	master.events.get_event(9386707)

3. *chime-frb-api* then determines the best possible way to authenticate with the backend, in this case it determines to ask for the user for their credentials.

.. code-block:: python

	[2020-03-01 16:48:17,265] INFO Authorization Status: None
	[2020-03-01 16:48:17,266] INFO Authorization Method: Username/Password
	Username: shiny
	Password: ###############
	[2020-03-01 16:48:22,617] INFO Authorization Result: Passed
	[2020-03-01 16:48:22,617] INFO Authorization Expiry: Sun Mar  1 17:18:22 2020

Underneath the hood, *chime-frb-api* contacts the backend with your credentials, if validated, the backend replies with an `access_token` and a `refresh_token`. It then uses these tokens to authenticate the actual API request made to the backend to ask for events parameters. For in-depth documentation on authenticaion, see :doc:`here <authentication>`.

4. Once authenticated, the CHIME/FRB backend replies with the event parameters,

.. code-block:: python

	{'beam_numbers': [166, 1166],
 	 'event_type': 'EXTRAGALACTIC',
 	 'fpga_time': 39927134208,
 	 id': 9386707,
 	 ...}


For advanced examples check out the :doc:`tutorials <tutorials>` section.



	

