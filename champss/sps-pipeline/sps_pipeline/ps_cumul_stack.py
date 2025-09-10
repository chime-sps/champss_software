import logging

import numpy as np
from omegaconf import OmegaConf
from ps_processes.ps_pipeline import StackSearchPipeline

log = logging.getLogger(__package__)


def run(
    pointing,
    ps_cumul_stack_processor,
    monthly_power_spectra=None,
    injection_path=None,
    injection_idx=None,
    only_injections=False,
    scale_injections=False,
):
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
    injection_path: str
        Path to injection file or string describing default injection type
    injection_idx: list
        Indices of injection file entries that are injected
    only_injections: bool
        Whether non-injections are filtered out. Default: False
    cutoff_frequency: float
        Highest frequency allowed for a candidate/detection. Default: 100
    scale_injection: bool
        Whether to scale the injection so that the detected sigma should be
        the same as the input sigma. Default: False

    Returns
    =======
    ps_detections: PowerSpectraDetections
        The PowerSpectraDetections interface storing the detections from the power spectra search.

    power_spectra: PowerSpectra
        The power spectra containing detections
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

        (
            ps_detections,
            power_spectra,
        ) = ps_cumul_stack_processor.pipeline.stack_and_search(
            pointing._id,
            monthly_power_spectra,
            injection_path,
            injection_idx,
            only_injections,
            scale_injections,
        )
    return ps_detections, power_spectra


def initialise(
    configuration, stack, search, search_monthly, known_source_threshold=np.inf
):
    class Wrapper:
        def __init__(self, config):
            self.config = config
            self.pipeline = StackSearchPipeline(
                run_ps_stack=stack,
                run_ps_search=search,
                run_ps_search_monthly=search_monthly,
                **OmegaConf.to_container(config.ps_cumul_stack),
                known_source_threshold=known_source_threshold,
            )

    return Wrapper(configuration)
