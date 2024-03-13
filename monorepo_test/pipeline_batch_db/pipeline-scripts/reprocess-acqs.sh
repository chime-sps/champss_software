#!/bin/bash
#
# Moves source intensity data acquired through the SPS script into SPS-compliant
# hierarchy at /data/frb-archiver/SPS/intensity/<yyyy>/<mm>/<dd>/<beam>

set -euo pipefail            # strict bash mode: runtime errors cause error exit

for d in $(find /data/sps-data/intensity/2020 -mindepth 3 -maxdepth 3 -name '0???' | sort -t/ -k 8,8 -k 5,5 -k 6,6 -k 7,7); do
    date=$(echo "$d" | cut -d/ -f 5-7)
    beam0=$(basename "$d")
    case ${beam0} in
        "0150")
            src="B0136+57"
            ra="24.83"
            dec="58.24"
            candidate="150"
            dm="73.811"
            f0="3.670"
        ;;
        "0057")
            src="J2111+2106"
            ra="317.89"
            dec="21.10"
            candidate="057"
            dm="59.296"
            f0="0.253"
        ;;
        "0038")
            src="J2129+1210A"
            ra="322.49"
            dec="12.17"
            candidate="038"
            dm="67.31"
            f0="9.036"
        ;;
        "0087")
            src="J0857+3349"
            ra="134.28"
            dec="33.82"
            candidate="087"
            dm="23.998"
            f0="4.116"
        ;;
        "0015")
            src="J1957-0002"
            ra="299.43"
            dec="-0.035"
            candidate="015"
            dm="38.31"
            f0="1.036"
        ;;
        *)
            >&2 echo "Unknown source ${beam0} on ${date}"
            continue
    esac

    row=$(echo "$beam0" | cut -c 2-4)
    if [[ -z $(find "/data/sps-data/intensity/${date}/"?"$row" -type f) ]]; then
        >&2 echo "No data for ${src} on ${date}"
        continue
    fi

    echo "Run the pipeline for ${src} on ${date}: (${ra} ${dec}), beam $row / $candidate"
    if perl -e "alarm shift; exec @ARGV" 5400 run-pipeline --date="$date" -- "$ra" "$dec" all; then
        hhat_candidates="${date}/${candidate}/clusters_4.png"
        [[ -f ${hhat_candidates} ]] && \
        curl \
            -H "Authorization: Bearer ${SLACK_SPS_BOT_ACCESS_TOKEN}" \
            -F file=@${hhat_candidates} \
            -F channels=C015BAVJ3UN \
            -F "initial_comment=Hhat candidate clusters for ${src} (true DM: ${dm}; F0: ${f0}) - ${date}" \
            https://slack.com/api/files.upload

        ps_candidates="${date}/${candidate}/candidates.png"
        [[ -f ${ps_candidates} ]] && \
        curl \
            -H "Authorization: Bearer ${SLACK_SPS_BOT_ACCESS_TOKEN}" \
            -F file=@${ps_candidates} \
            -F channels=C015BAVJ3UN \
            -F "initial_comment=PS candidates for ${src} (true DM: ${dm}; F0: ${f0}) - ${date}" \
            https://slack.com/api/files.upload
    else
        if [[ $? -eq 142 ]]; then
                curl -H "Authorization: Bearer ${SLACK_SPS_BOT_ACCESS_TOKEN}" \
                -H "Content-type: application/json; charset=utf8" \
                --data "{\"channel\": \"C015BAVJ3UN\", \"icon_emoji\": \"butterfly\", \"username\": \"SPS test source analysis\", \"text\": \"Analysis pipeline for source ${src} on ${date} ran overtime.\"}" \
                https://slack.com/api/chat.postMessage > /dev/null
        else
            curl -H "Authorization: Bearer ${SLACK_SPS_BOT_ACCESS_TOKEN}" \
                    -H "Content-type: application/json; charset=utf8" \
                    --data "{\"channel\": \"C015BAVJ3UN\", \"icon_emoji\": \"butterfly\", \"username\": \"SPS test source analysis\", \"text\": \"Analysis pipeline for source ${src} on ${date} failed.\"}" \
                    https://slack.com/api/chat.postMessage > /dev/null
        fi
        rm -rf "${date}/${row}" "${date}/0${row}" "${date}/1${row}" "${date}/2${row}" "${date}/3${row}"
    fi
done
