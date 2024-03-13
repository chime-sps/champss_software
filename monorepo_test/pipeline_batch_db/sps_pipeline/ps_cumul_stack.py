import logging

from omegaconf import OmegaConf
from ps_processes.ps_pipeline import StackSearchPipeline

log = logging.getLogger(__package__)


def run(pointing, ps_cumul_stack_processor, monthly_power_spectra=None):
    """
    Run the power spectra stacking and searching process.

    Parameters
    =======
    ps_stack: PsStack
        PsStack object from sps-databases with information about the power spectra stack.

    ps_cumul_stack_processor: Wrapper
        A wrapper object containing the StackSearchPipeline configured to sps_config.yml

    monthly_power_spectra: PowerSpectra
        The monthly power spectra if they have been loaded already.

    Returns
    =======
    ps_detections: PowerSpectraDetections
        The PowerSpectraDetections interface storing the detections from the power spectra search.
    """
    if ps_cumul_stack_processor.pipeline.run_ps_stack:
        log.info(
            f"Performing stacking of {pointing.ra:.2f} {pointing.dec:.2f} into its"
            " cumulative stack"
        )
    if ps_cumul_stack_processor.pipeline.run_ps_search:
        log.info(
            "Performing searching on cumulative stack"
            f" {pointing.ra:.2f} {pointing.dec:.2f}"
        )
    ps_detections, power_spectra = ps_cumul_stack_processor.pipeline.stack_and_search(
        pointing.pointing_id, monthly_power_spectra
    )
    return ps_detections, power_spectra


def initialise(configuration, stack, search, search_monthly):
    class Wrapper:
        def __init__(self, config):
            self.config = config
            self.pipeline = StackSearchPipeline(
                run_ps_stack=stack,
                run_ps_search=search,
                run_ps_search_monthly=search_monthly,
                **OmegaConf.to_container(config.ps_cumul_stack),
            )

    return Wrapper(configuration)
