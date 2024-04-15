.. CHIME/FRB API documentation master file, created by
   sphinx-quickstart on Tue Feb 18 13:38:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CHIME/FRB API
=============

.. raw:: html

	<div class="container-fluid">
    	<div class="row">
       		<div class="col-md-9">

chime-frb-api is a Python library to interact with the Canadian Hydrogen Intensity Mapping Experiment' (CHIME) Fast Radio Burst (FRB) backend. It provides a high-level interface for accessing data products, compute hardware, analysis jobs and controlling the backend itself.

For a brief introduction to the ideas behind the library, you can read the :doc:`introduction notes <introduction>`. Visit the :doc:`installation page <installation>` to see how to download and install the package. You can browse through the :doc:`tutorials <tutorials>` to see basic examples of what you can do with chime-frb-api.

To contribute or report a bug, please visit the :doc:`developer section <developer>`. 

.. raw:: html

    		</div>
			<div class="col-md-3">
    			<div class="panel panel-default">   
        			<div class="panel-heading">
            			<h3 class="panel-title">Contents</h3>
           			</div>
       			<div class="panel-body">

.. toctree::
   :maxdepth: 1

   Installation <installation>
   Introduction <introduction>
   Tutorials <tutorials>
   Authentication <authentication>
   Developer <developer>

.. raw:: html

       			</div>
     		</div>
   		</div>
   </div>

.. code-block:: python

	>>> from chime_frb_api.backends import frb_master
	>>> master = frb_master.FRBMaster(base_url="https://frb.chimenet.ca/frb-master")
	>>> master.events.get_event(71780218)
	{'beam_numbers': [185],
 	 'event_type': 'EXTRAGALACTIC',
 	 'fpga_time': 1838000314368,
 	 'id': 71780218,
 	 ... }
