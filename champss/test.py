from scheduler.workflow import (
        schedule_workflow_job,
        wait_for_no_tasks_in_states,
        clear_workflow_buckets,
        docker_swarm_running_states
)

def my_workflow_function()
    print("Running my Workflow function.")
    return {"some_value": True}, [], []

def test_submit_job():
    workflow_buckets_name = ""

    clear_workflow_buckets.main(
        args=["--workflow-buckets-name", workflow_buckets_name],
        standalone_mode=False,
    )

    work_id = schedule_workflow_job(
        docker_image_name="",
        docker_mounts=[],
        docker_name="",
        docker_memory_reservation=0,
        workflow_buckets_name=workflow_buckets_name,
        workflow_function="",
        workflow_params={},
        workflow_tags=[],
    )

    wait_for_no_tasks_in_states(
        states_to_wait_for_none=docker_swarm_pending_states, docker_service_name_prefix=""
    )
    print("Jobs are now running.")
    wait_for_no_tasks_in_states(
        states_to_wait_for_none=docker_swarm_running_states, docker_service_name_prefix=""
    )
    print("Jobs are now complete.")

    work_result = get_work_from_results(
        workflow_results_name=workflow_buckets_name,
        work_id=work_id,
        failover_to_buckets=True,
    )
    print

if __name__ == "__main__":
    test_submit_job()
