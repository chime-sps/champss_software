#Step 1 — Query candidates from database
#Connects to the CHAMPSS database using CandidateViewerQuery, loops over a date range (folders), and collects all candidates matching #specific classifications (<faint>, NEW CANDIDATE) into a single list (all_candidates).
#Step 2 — Run multi-day folding pipeline
#For each candidate, builds arguments and runs multidayfold_pipeline to process data across multiple days; skips candidates already #processed (based on existing output files) and stores results in all_outputs.
#Step 3 — Save results
#Serializes the collected outputs (all_outputs) into a file using pickle, allowing the results to be reloaded later without recomputing.

import sps_databases
import subprocess
from cfbm.bm_data import get_data
import os
import scipy
from datetime import datetime, timedelta
from scheduler.run_as_service import run_as_service
import pickle
from sps_pipeline.candidate_viewer import CandidateViewerQuery
from sps_pipeline.candidate_viewer import CandidateViewerRegistrar
from multiday_search import multidayfold_pipeline
import traceback
from dataclasses import dataclass


#All harcoded info should be in the CONFIG dict, so that it is easier to change if needed and to avoid hardcoding in the code itself.
config = {
# Database configuration
    "db_config": {
        "host": "sps-archiver1",
        "user": "automation",
        "port": 3306,
        "password": "",
        "database": "champss",
    },
    "start_date": datetime(2026, 4, 9),#3,22
    "end_date": datetime(2026, 6, 15),#4,8
    "classifications": ["<faint>", "NEW CANDIDATE"],
    "outfile_dir": "/mnt/beegfs-client/processed/multiday/",
    "raw_dir": "/mnt/beegfs-client/raw/",
    "archive_dir": "/mnt/beegfs-client/processed/archives/",
    "docker_image": "sps-archiver1.chime:5000/champss_software:run_on_compute1",
    "filename": 'result_single_day4.pkl',
    "debug": True,
}

#This is to debug, allows to control printing. Put True to print, put falso when everything works
def log(config, msg):
    if config.get("debug", False):
        print(msg, flush=True)


#Define fct so that main() is clean

def build_folders(start_date, end_date):
    folders = []
    current = start_date
    
    # Loop through the date range and create folder names
    while current <= end_date:
        folders.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    return folders


def query_candidates(config, folders):
    all_candidates = []
    #query = CandidateViewerQuery(survey="stackcands", db_config=db_config)
    #candidates = query.get_metadata(folder="stack_0")
    classifications = config["classifications"]

    
    with CandidateViewerQuery(survey='dailycands', db_config=config["db_config"]) as query:
        for folder in folders:
            log(config, f"\n[STEP 1] Processing folder: {folder}")

            for cls in classifications:
                try:
                    candidates = query.get_ratings(
                        folder=folder,
                        classification=cls,
                        with_metadata=True
                    )
                except Exception:
                    print(f"No data for {folder}")
                    continue  # skip this classification if query fails

            
                log(config, f"[STEP 1] Found {len(candidates)} candidates for {cls} in {folder}")
                all_candidates.extend(candidates)

    log(config, "[STEP 1] Query finished")

    return all_candidates



def run_multiday_fold(all_candidates, config):
    
    #Step_2-Run the multi-day fold
    all_outputs = []

    for i, cand in enumerate(all_candidates):
        log(config, f"\n[STEP 2] Candidate {i+1}/{len(all_candidates)}")

        metadata = cand['metadata']
        input_file = metadata['input_file']
        outfile = os.path.join(config["outfile_dir"], metadata["file"])
        log(config, f"[STEP 2] File: {metadata['file']}")
        log(config, f"[STEP 2] Input: {input_file}")
        log(config, f"[STEP 2] Output: {outfile}")
        print(input_file)

        # Skip if already folded
        if os.path.exists(outfile):
            log(config, f"[STEP 2] SKIP already exists: {metadata['file']}")
            continue

        log(config, f"[STEP 2] RUNNING multiday-fold pipeline: {metadata['file']}")

        #to run in terminal(need conversion to python file also:)command = f"multidayfold_pipeline --candpath {input_file}
        #--db-name champss_processing --nday 0 --datpath /mnt/beegfs-client/raw/ --foldpath /mnt/beegfs-client/processed/archives/ --           use-workflow"
        
        args = [
        "--candpath",
        input_file,
        "--db-name",
        "champss_processing",
        "--nday",
        "0",
        "--datpath",
        config["raw_dir"],
        "--foldpath",
        config["archive_dir"],
       # "--use-workflow",
        "--docker-image-name",
        config["docker_image"],
        ]
        #add workflow option to the command to avoid error

        #nday=0 is to run over all available days(did not work last time)
        #This is starting a service that starts another service. 
        #The first one will be on sps-compute1 if you start it from my branch, but the second not
        #so we put the docker command so that both are on compute1

        
        #We want/need to add something to avoid rerunning the fold on candidate we did on previous day(flag!!)
        # Running the command
        try:
            log(config, f"[STEP 2] Pipeline started: {metadata['file']}")
            fold_output = multidayfold_pipeline.main(
            args=args,
            standalone_mode=False
        )
            all_outputs.append([cand,fold_output[0]])

            log(config, f"[STEP 2] SUCCESS, Fold finished: {metadata['file']}")
            
        except Exception as e: #e = the error object
            log(config, f"[STEP 2] FAILED: {metadata['file']}")
            log(config, f"[STEP 2] ERROR TYPE: {type(e)}")
            log(config, f"[STEP 2] ERROR MSG: {e}")
            log(config, traceback.format_exc())
            
            
    return all_outputs


def save_results(all_outputs, filename):
    
    #Step_3-Retrieve the relevant parameters from the multiday fold        
    log(config, f"[STEP 3] Saving to {filename}")

    with open(filename, 'wb') as outp:
        pickle.dump(all_outputs, outp, pickle.HIGHEST_PROTOCOL)

    log(config, f"[STEP 3] Saved {len(all_outputs)} outputs")



# In main remove all loops by creating fct predefined so that main is just like reading a recipe not the actual cooking

def main():
    
    #Step_1-Query website candidates and put them in a list
    folders = build_folders( 
        config["start_date"],
        config["end_date"]
    )

    all_candidates = query_candidates(config, folders)
    
    #Step_2-Run multiday_fold
    all_outputs = run_multiday_fold(all_candidates, config)

    
    #Step_3-Save output
    save_results(
        all_outputs,
        config["filename"]
    )
    
#This runs if this file being run directly by Python and not if it is being imported as a module?”
if __name__ == "__main__":
    main()

# How to run that code
#We run that from the terminal in /data/rtellier/Automation Part_1, then run  docker command and then:python Query(main)_website_multidayfold.py