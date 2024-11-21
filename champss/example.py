import os

from scheduler.workflow import (
        schedule_workflow_job,
        wait_for_no_tasks_in_states,
        clear_workflow_buckets,
        get_work_from_results,
        docker_swarm_pending_states,
        docker_swarm_running_states,
)

def example_workflow_function(my_input_value):
    # This function is what you want to run in your container
    print("Running my Workflow function.")
    node_name = os.environ.get("NODE_NAME", "")
    random_plot_path = f"/champss_module/plots/J1629+4639_candidate.png"
    # Must return {}, [], []
    # The first dictionary is what will be saved in Results
    # The last dictionary is the path to any plots that will be visible in the Workflow UI
    # (the plots must be under /data/chime/sps/, and thus that path must be mounted in the Docker container!)
    return {"my_input_value": my_input_value, "node_name": node_name}, [random_plot_path], [random_plot_path]


# See logs in /data/chime/sps/logs/processing-<docker_name>.log
# Be careful ls'ing into here though...it will be very slow, it's full of logs
def example_submit_job():
    workflow_buckets_name = "my-bucket" # Anything you want, preferably keep similar jobs in the same bucket

    # Since clear_workflow_buckets is a Click CLI command, we need to call it this strange way
    # with .main(), args, and standalone_mode=False
    clear_workflow_buckets.main(
        args=["--workflow-buckets-name", workflow_buckets_name],
        standalone_mode=False,
    )
    
    docker_service_prefix = "chris" # Identifier to make your container name unique
    docker_registry_name = "sps-archiver1.chime:5000" # DockerHub = chimefrb -> where the image is stored
    github_repo_name = "champss_software" # Name of the GitHub repository you're working on
    github_branch_name = "adapt-code-for-local-docker-image-registry" # Name of your development branch

    work_id = schedule_workflow_job( # Not a CLI command (yet) so we can call it directly
        # Format: <docker_registry_name>/<github_repo_name>:<github_branch_name>
        docker_image=f"{docker_registry_name}/{github_repo_name}:{github_branch_name}",
        # Format: <path_on_host>:<path_in_container>
        docker_mounts=[
            "/home/candrade:/champss_module/my_home",
            "/data/chime/sps/pulsars/plots:/champss_module/plots",
        ],
        # Format: <docker_service_prefix>-<anything>
        docker_name=f"{docker_service_prefix}-my-workflow", # Will be prepended with "processing-"
        # In gigabytes
        docker_memory_reservation=5,
        workflow_buckets_name=workflow_buckets_name,
        # Path to your function in the same format as the import statement
        workflow_function="champss.example.example_workflow_function",
        # Parameters to pass to your function as a dictionary
        workflow_params={"my_input_value": 42},
        # Any string tags to allow for filtering on the Workflow UI
        workflow_tags=["chris", "my-workflow"],
        # Owner of the Workflow Job in Buckets/Results
        workflow_user="candrade"
    )

    # This function will pause the script until all tasks are no longer in the pending state
    # that have the same docker_service_name_prefix in their container name
    wait_for_no_tasks_in_states(
        states_to_wait_for_none=docker_swarm_pending_states, docker_service_name_prefix=docker_service_prefix
    )
    print("Jobs are now running.")
    
    # This function will pause the script until all tasks are no longer in the running state
    # that have the same docker_service_name_prefix in their container name
    wait_for_no_tasks_in_states(
        states_to_wait_for_none=docker_swarm_running_states, docker_service_name_prefix=docker_service_prefix
    )
    print("Jobs are now complete.")

    # This function will get the results of the work_id from the workflow_results_name Results
    # (Results are under the same name as the Bucket, with the same Work, just moved there after completion)
    work_result = get_work_from_results(
        workflow_results_name=workflow_buckets_name,
        work_id=work_id,
        failover_to_buckets=True,
    )
    print(work_result)

if __name__ == "__main__":
    # When evoking this script with python3 example.py, example_submit_job() will run
    example_submit_job()
