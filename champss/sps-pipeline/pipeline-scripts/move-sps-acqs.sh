#!/bin/bash
#
# Moves source intensity data acquired through the SPS script into SPS-compliant
# hierarchy at /data/frb-archiver/SPS/intensity/<yyyy>/<mm>/<dd>/<beam>

set -euo pipefail            # strict bash mode: runtime errors cause error exit

# Example acq directory: /data/frb-archiver/acq_data/sps_B0136+57_2020-06-10-09-15_3150
#
# Explanation: source B0136+57, transiting on June 10, 2020 through beam 3150, with
# acquisition started at 09:15 am UTC (!! since 16:20 on June 14, 2020)
#
# L1 nodes writing out the required beam actually wants subdirectories of the
# acq directory for all four beams it is receiving, each name beam_<beam>.
# Normally, three of them will be empty, and only the fourth one will have
# "chunk_<NNN>.msg" files, but this script does not assume that and will copy
# all files written into the beams subdirectories.

# Because we use a fairly generous 30-minute data acq around the transit, we
# will avoid touching acquisitions less than two hours old:
acq_cutoff_ts=$(date --date="2 hours ago" --utc +%Y%m%d%H%M)

# We want wildcard with no matches to expand to nothing, so the loop has nothing to ru
shopt -s nullglob

# Collect "acquisition runs", and process them as a unit
acq_runs=$(
    for d in "/data/frb-archiver/acq_data/sps_"*_20??-*; do
        echo "$d"
    done \
    | cut -d_ -f1-4 \
    | uniq
)
[[ -z "${acq_runs}" ]] && echo "Nothing to process"

# Back to normal wilcard expansion behaviour, so we don't get tripped by
# expanding on an empty variable
shopt -u nullglob
shopt -s failglob

for acq_run in ${acq_runs}; do
    acq_ts=$(echo "${acq_run}" | cut -d_ -f4 | tr -d -)
    if [[ ${acq_ts} -gt ${acq_cutoff_ts} ]]; then
       echo "Skip ${acq_run}"
       continue
    fi

    acq_source=$(echo "${acq_run}" | cut -d_ -f3)

    acq_date=$(echo "${acq_run}" | cut -d_ -f4 | cut -d- -f1-3 --output-delimiter=/)
    echo "${acq_date} ${acq_run}"

    # Process all L1 node directories created for this acq run
    acq_files=0
    for d in "${acq_run}_"[0-9][0-9][0-9][0-9]; do
        # Remove empty acq subdirs
        find "$d" -type d -empty -delete

        # If the acquisition directory itself was removed, then there were no
        # saved intensity files and the acquisition failed
        if [[ ! -d "$d" ]]; then
            echo "Acquisition failed!"
        else
            # Handle remaining directories: these will all contain chunks to move
            for beam_dir in "$d/beam"_*; do
                beam=$(echo "${beam_dir}" | cut -d_ -f 6)
                dest_dir="/data/frb-archiver/SPS/intensity/${acq_date}/${beam}"
                mkdir -p "${dest_dir}"
                echo "$beam $(find "${beam_dir}" -type f | wc -l)"
                #rsync -rt --remove-source-files "${beam_dir}/" "${dest_dir}"
                mv "${beam_dir}/"*.msg "${dest_dir}"
                echo "$beam $(find "${dest_dir}" -type f | wc -l)"
                (( acq_files+=$(find "${dest_dir}" -type f | wc -l) ))
            done
        fi

        # Clean up the acquisition directory
        find "$d" -type d -empty -delete
    done

    if (( acq_files > 0 )); then
        curl --data-urlencode "payload={\"text\": \"Data for source ${acq_source} for ${acq_date} are now ready\"}" https://hooks.slack.com/services/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
        # Notify the worker pipeline
        /usr/bin/python -c "f=open('/tmp/davor-pipeline', 'w+'); print >> f, '${acq_source} ${acq_date}'; f.flush(); f.close()"
        # `echo` or `cat` aren't a good choice because they will block when there is no reader
        #echo "${acq_source} ${acq_date}" > /tmp/davor-pipeline
    else
        curl --data-urlencode "payload={\"text\": \"No data were acquired for source ${acq_source} for ${acq_date}\"}" https://hooks.slack.com/services/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
    fi

done
