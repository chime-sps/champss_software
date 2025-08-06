ADDITIONAL_PIPELINE_PARAMS="$1" # First argument
ADDITIONAL_STACK_PARAMS="$2" # Second argument

rm -r ./benchmark/stack ./benchmark/2022/06/*/*/*_obs_id.txt

run-pipeline --plot --plot-threshold 6.5 --date 20220618 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220619 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220620 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220621 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220622 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220623 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220624 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220625 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220626 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01
run-pipeline --plot --plot-threshold 6.5 --date 20220627 --stack --db-host sps-archiver1 --db-name sps_benchmark --basepath ./benchmark/ --datpath /data/chime/sps/raw_backup/ --stackpath ./benchmark/ $ADDITIONAL_PIPELINE_PARAMS 317.21 50.01

run-stack-search-pipeline --plot --plot-threshold 10 --db-host sps-archiver1 --db-name sps_benchmark --path-cumul-stack ./benchmark/ $ADDITIONAL_STACK_PARAMS 317.21 50.01 all

print_candidates --threshold 6.5 ./benchmark/2022/06/*/*/*_power_spectra_candidates.npz
print_candidates --threshold 10  ./benchmark/*/*_candidates.npz


rm -r ./benchmark/stack ./benchmark/2022/06/*/*/*_obs_id.txt
