"""Script to clear SPS Workflow Results/Buckets."""
import click
from chime_frb_api.modules.buckets import Buckets
from chime_frb_api.modules.results import Results


@click.command()
@click.option(
    "--workflow-results-name",
    type=str,
    required=True,
    help="Name of the Workflow Results collection to delete",
)
def clear_workflow_results(workflow_results_name):
    """Function to empty given SPS Results collection on-site."""
    results_api = Results()
    results_list = ["pass"]
    while len(results_list) != 0:
        # Results API only allows 10 deletes per request
        results_list = results_api.view(
            query={}, pipeline=workflow_results_name, limit=10, projection={"id": 1}
        )
        result_ids_to_delete = [result["id"] for result in results_list]
        results_api.delete_ids(pipeline=workflow_results_name, ids=result_ids_to_delete)


@click.command()
@click.option(
    "--workflow-buckets-name",
    type=str,
    required=True,
    help="Name of the Workflow Buckets collection to delete",
)
def clear_workflow_buckets(workflow_buckets_name):
    """Function to empty given SPS Buckets collection on-site."""
    buckets_api = Buckets()
    buckets_list = ["pass"]
    while len(buckets_list) != 0:
        # Bucket API only allows 100 deletes per request
        buckets_list = buckets_api.view(
            query={"pipeline": workflow_buckets_name},
            limit=100,
            projection={"id": True},
        )
        bucket_ids_to_delete = [bucket["id"] for bucket in buckets_list]
        buckets_api.delete_ids(ids=bucket_ids_to_delete)


@click.command()
@click.option(
    "--workflow-name",
    type=str,
    required=True,
    help="Name of the Workflow Buckets/Results collection to delete",
)
def clear_workflow(workflow_name):
    """Function to empty given SPS Buckts AND Results collection on-site."""
    try:
        clear_workflow_buckets(
         ["--workflow-buckets-name", workflow_name], standalone_mode=False
        )
    except Exception as e:
        pass
    try:
        clear_workflow_results(
            ["--workflow-results-name", workflow_name], standalone_mode=False
        )
    except Exception as e:
        pass
