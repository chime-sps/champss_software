#!/bin/bash
#
# Reads sources to analyze from named pipe `/tmp/davor-pipeline`, and runs the
# v0 pipeline on the source. If the analysis is successful, candidate clusters
# plot at duty cycle=0.4 is uploaded to Slack channel `#slow-pulsar-alerts`.
#
# The input data is in the form "SOURCE-NAME DATE", and if the input source is not one of the five whose intensity data is being regularly saved, it is silently ignored.

set -euo pipefail            # strict bash mode: runtime errors cause error exit

# Use the Python venv for the pipeline
source ~/.local/share/virtualenvs/sps-v0/bin/activate

while :; do
    while read -r src_task; do
        echo ${src_task}
        src=$(echo "$src_task" | cut -d" " -f1)
        src_date=$(echo "$src_task" | cut -d" " -f2)
        transit_date=$(echo "$src_date" | tr / -)
        candidate=""

        case ${src} in
            "B0136+57")
                ra="24.83"
                dec="58.24"
                candidate="150"
                dm="73.811"
                f0="3.670"
            ;;
            "J2111+2106")
                ra="317.89"
                dec="21.10"
                candidate="057"
                dm="59.296"
                f0="0.253"
            ;;
            "J2129+1210A")
                ra="322.49"
                dec="12.17"
                candidate="038"
                dm="67.31"
                f0="9.036"
            ;;
            "J0857+3349")
                ra="134.28"
                dec="33.82"
                candidate="087"
                dm="23.998"
                f0="4.116"
            ;;
            "J1957-0002")
                ra="299.43"
                dec="-0.035"
                candidate="015"
                dm="38.31"
                f0="1.036"
            ;;
        esac
        if run-pipeline --date="$transit_date" -- "$ra" "$dec" all; then
            if [[ -n "${candidate}" ]]; then
                curl \
                    -H "Authorization: Bearer ${SLACK_SPS_BOT_ACCESS_TOKEN}" \
                    -F file=@${src_date}/${candidate}/clusters_4.png \
                    -F channels=C015BAVJ3UN \
                    -F "initial_comment=Candidate clusters for ${src_task} (true DM: ${dm}; F0: ${f0})" \
                    https://slack.com/api/files.upload
            fi
        else
            curl -H "Authorization: Bearer ${SLACK_SPS_BOT_ACCESS_TOKEN}" \
                 -H "Content-type: application/json; charset=us-ascii" \
                 --data "{\"channel\": \"C015BAVJ3UN\", \"icon_emoji\": \"butterfly\", \"username\": \"SPS test source analysis\", \"text\": \"Analysis pipeline for source ${src_task} failed.\"}" \
                 https://slack.com/api/chat.postMessage
        fi

    done < /tmp/davor-pipeline
done
