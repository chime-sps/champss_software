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
    print("Running my Workflow function.")
    node_name = os.environ.get("NODE_NAME", "")
    return {"my_input_value": my_input_value, "node_name": node_name}, [], []


# See error logs in /data/chime/sps/logs/processing-<docker_name>.log
def example_submit_job():
    workflow_buckets_name = "my-bucket"

    clear_workflow_buckets.main(
        args=["--workflow-buckets-name", workflow_buckets_name],
        standalone_mode=False,
    )
    
    docker_service_prefix = "chris"
    docker_registry_name = "sps-archiver1.chime:5000" # DockerHub = chimefrb
    github_repo_name = "champss_software"
    github_branch_name = "adapt-code-for-local-docker-image-registry"

    work_id = schedule_workflow_job(
        docker_image=f"{docker_registry_name}/{github_repo_name}:{github_branch_name}",
        docker_mounts=["/home/candrade:/champss_module/my_home"],
        docker_name=f"{docker_service_prefix}-my-workflow", # Will be prepended with "processing-"
        docker_memory_reservation=5,
        workflow_buckets_name=workflow_buckets_name,
        workflow_function="champss.example.example_workflow_function",
        workflow_params={"my_input_value": 42},
        workflow_tags=["chris", "my-workflow"],
        workflow_user="candrade"
    )

    wait_for_no_tasks_in_states(
        states_to_wait_for_none=docker_swarm_pending_states, docker_service_name_prefix=docker_service_prefix
    )
    print("Jobs are now running.")
    
    wait_for_no_tasks_in_states(
        states_to_wait_for_none=docker_swarm_running_states, docker_service_name_prefix=docker_service_prefix
    )
    print("Jobs are now complete.")

    work_result = get_work_from_results(
        workflow_results_name=workflow_buckets_name,
        work_id=work_id,
        failover_to_buckets=True,
    )
    print(work_result)

if __name__ == "__main__":
    example_submit_job()
