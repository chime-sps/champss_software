#!/usr/bin/env python
# coding: utf-8
#to run that need to run "cd /data/rtellier/Automation Part_1" and then "conda activate champss2"
import sps_pipeline.candidate_viewer
import sps_databases
import subprocess
from cfbm.bm_data import get_data
import os
import click
import scipy
from datetime import datetime, timedelta
from sps_pipeline.candidate_viewer import CandidateViewerQuery
from multiday_search import multidayfold_pipeline



# Database configuration
db_config = {
    'host': 'sps-archiver1',
    'user': 'automation',
    'port': 3306,
    'password': '',
    'database': 'champss'
}
classifications = ['<faint>', 'NEW CANDIDATE']


# In[20]:


# Click (to run in terminal)
@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--nday",
    default=10,
    type=int,
    help="Number of days to fold and search. Default will fold and search all available days.",
)
@click.option(
    "--start-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="Start date of data to process (inclusive)."
)
@click.option(
    "--end-date",
    type=click.DateTime(["%Y%m%d", "%Y-%m-%d", "%Y/%m/%d"]),
    required=True,
    help="End date of data to process (inclusive)."
)
def main(nday, start_date, end_date):
#Multiday folding pipeline CLI. Generates folder list from start_date to end_date, queries candidates, and runs multiday folds.
    
    # Generate folder strings from start_date to end_date
    folders = []
    current = start_date
    while current <= end_date:
        folders.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)

    print(f"Processing folders: {folders}")
    print(f"Nday: {nday}")
    print(f"Classifications: {classifications}")

  
    all_candidates = []
    with CandidateViewerQuery(survey='dailycands', db_config=DB_CONFIG) as query:
        for folder in folders:
            print(f"\nProcessing folder: {folder}")
            for cls in classifications:
                try:
                    candidates = query.get_ratings(
                        folder=folder,
                        classification=cls,
                        with_metadata=True
                    )
                except Exception as e:
                    print(f"No data for {folder} / {cls}: {e}")
                    continue
                print(f"Found {len(candidates)} candidates for {cls} in {folder}")
                all_candidates.extend(candidates)
    print("\nQuery finished")

    # Step 2: Run multiday folds
    for cand in all_candidates:
        metadata = cand['metadata']
        input_file = metadata['input_file']
        outfile = f"/mnt/beegfs-client/processed/multiday/{metadata['file']}"
        
        # Skip if already folded
        if os.path.exists(outfile):
            print(f"Skipping {metadata['file']} — already folded")
            continue
        

        args = [
            "--candpath", input_file,
            "--db-name", "champss_processing",
            "--nday", str(nday),
            "--datpath", "/mnt/beegfs-client/raw/",
            "--foldpath", "/mnt/beegfs-client/processed/archives/",
            "--use-workflow"
        ]

        try:
            multidayfold_pipeline.main(args=args, standalone_mode=False)
            print(f"Fold finished: {metadata['file']}")
        except Exception as e:
            print(f"Folding failed for {metadata['file']}: {e}")

#In order to only run the code when running it in the terminal
if __name__ == "__main__":# if this wasn't there each time I would import the command in another file it would run
    main()

