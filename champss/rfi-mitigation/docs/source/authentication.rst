.. _authentication:

Authentication
--------------

*chime-frb-api* uses an `access_token` and an optional `refresh_token` to authorize HTTP Requests against the CHIME/FRB backends. For a deeper look at these *tokens* visit the :doc:`developer<developer>` section.

The authentication process can be reduced to the following steps:

- Generate tokens with proper credentials
- Attach the `access_token` with HTTP Requests to validate them
- Regenerate the `access_tokens` using the `refresh_token` when it expires


How to generate tokens?
^^^^^^^^^^^^^^^^^^^^^^^
*chime-frb-api* manages the creation, use and refreshing of tokens automatically and always prefers to use *tokens* before *credentials*. In the order of decreasing prefernce, *chime-frb-api* looks for authentication components in the following locations:

1. Arguments passed when creating the backend object

.. code-block:: python

	from chime_frb_api.backends import frb_master
	# Option A
	master = frb_master.FRBMaster(base_url="https://some-url", username=username, password=password)
	# Option B
	master = frb_master.FRBMaster(base_url="https://some-url", access_token=access_token, refresh_token=refresh_token)

2. If no arguments are passed, *chime-frb-api* will look in the local user environment via for the following variables

.. code-block::

	FRB_MASTER_ACCESS_TOKEN=<TOKEN>
	FRB_MASTER_REFRESH_TOKEN=<TOKEN>
	FRB_MASTER_USERNAME=debug
  	FRB_MASTER_PASSWORD=debug

3. Finally, if no arguments are passed or are available via the local environment, *chime-frb-api* will ask the user for credentials

.. code-block::
	
	[2020-03-01 16:48:17,265] INFO Authorization Status: None
	[2020-03-01 16:48:17,266] INFO Authorization Method: Username/Password
	Username: shiny
	Password: ###############
	[2020-03-01 16:48:22,617] INFO Authorization Result: Passed
	[2020-03-01 16:48:22,617] INFO Authorization Expiry: Sun Mar  1 17:18:22 2020


**Generating a new access token using your credentials invalidates the currently issued refresh token.**


User Workflow
^^^^^^^^^^^^^
Ideally, you should only need to generate your token **once**, and then save it to your environment. 

.. code-block:: python

	from chime_frb_api.backends import frb_master
	master = frb_master.FRBMaster(base_url="https://frb.chimenet.ca/frb-master")
	master.API.authorize()
	
	# Enter Credentials
	[2020-03-01 23:26:57,047] INFO Authorization Method: Username/Password
	Username: shiny
	Password:
	[2020-03-01 23:27:01,983] INFO Authorization Result: Passed
	[2020-03-01 23:27:01,983] INFO Authorization Expiry: Sun Mar  1 23:57:01 2020
	
	# Find your tokens
	master.API.access_token
	"your.very.long.access.token"
	master.API.refresh_token
	"your-shorter-refresh-token"


Now you can copy the tokens into your `.bashrc` or `.bash_profile` files:

.. code-block::

	export FRB_MASTER_ACCESS_TOKEN="your.very.long.access.token"
	export FRB_MASTER_REFRESH_TOKEN="your-shorter-refresh-token"
