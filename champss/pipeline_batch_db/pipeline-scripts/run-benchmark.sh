ADDITIONAL_PIPELINE_PARAMS="$1" # First argument
ADDITIONAL_STACK_PARAMS="$2" # Second argument

rm -r /data/chime/sps/benchmark/stack /data/chime/sps/benchmark/2022/06/*/*/*_obs_id.txt

run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220618 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220619 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220620 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220621 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220622 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220623 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220624 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220625 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220626 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01
run-pipeline $ADDITIONAL_PIPELINE_PARAMS --plot --plot-threshold 6.5 --date 20220627 --stack --db-host ss1 --db-name sps_benchmark --basepath /data/chime/sps/benchmark --stackpath /data/chime/sps/benchmark 317.21 50.01

run-stack-search-pipeline $ADDITIONAL_STACK_PARAMS --plot --plot-threshold 13 --db-host ss1 --db-name sps_benchmark --path-cumul-stack /data/chime/sps/benchmark 317.21 50.01 all

print_candidates --threshold 6.5 /data/chime/sps/benchmark/2022/06/*/*/*_power_spectra_candidates.npz
print_candidates --threshold 13  /data/chime/sps/benchmark/*/*_candidates.npz


rm -r /data/chime/sps/benchmark/stack /data/chime/sps/benchmark/2022/06/*/*/*_obs_id.txt
