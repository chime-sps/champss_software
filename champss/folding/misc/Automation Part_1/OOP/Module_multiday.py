# Module_multiday.py
from datetime import timedelta
import os
from sps_pipeline.candidate_viewer import CandidateViewerQuery
from multiday_search import multidayfold_pipeline #import the command from git repo


class MultidayCandidatePipeline:
    
    #Pipeline for running multiday candidate folding over a date range.
    
#all of those are the inputs needed for pipeline of 
    def __init__(self, db_config, start_date, end_date, classifications):
        self.db_config = db_config
        self.start_date = start_date
        self.end_date = end_date
        self.classifications = classifications

        # Paths
        self.raw_path = "/mnt/beegfs-client/raw/"
        self.fold_path = "/mnt/beegfs-client/processed/archives/"
        self.output_path = "/mnt/beegfs-client/processed/multiday/"

    def generate_folders(self):
        #Generate a list of folder names (YYYY-MM-DD) for each day in the range
        folders = []
        current = self.start_date
        while current <= self.end_date:
            folders.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return folders

    def run_fold(self, cand):
        #Run the multiday folding for a single candidate
        metadata = cand['metadata']
        input_file = metadata['input_file']
        outfile = os.path.join(self.output_path, metadata['file'])

        if os.path.exists(outfile):
            print(f"Skipping {metadata['file']} — already folded")
            return

        args = [
            "--candpath", input_file,
            "--db-name", "sps-processing",
            "--nday", "2",
            "--datpath", self.raw_path,
            "--foldpath", self.fold_path,
            "--use-workflow"
        ]

        try:
            multidayfold_pipeline.main(
                args=args,
                standalone_mode=False
            )
            print(f"Fold finished: {metadata['file']}")

        except Exception as e:
            print(f"Folding failed for {metadata['file']}")
            print(e)

    def run(self):
        #Run the pipeline over all folders and classifications.
        folders = self.generate_folders()

        with CandidateViewerQuery(
            survey='dailycands',
            db_config=self.db_config
        ) as query:

            for folder in folders:
                print(f"\n===== Processing {folder} =====")

                for cls in self.classifications:
                    try:
                        candidates = query.get_ratings(
                            folder=folder,
                            classification=cls,
                            with_metadata=True
                        )
                    except Exception as e:
                        print(f"No data for {folder}: {e}")
                        continue

                    print(f"Found {len(candidates)} candidates for {cls}")

                    for cand in candidates:
                        self.run_fold(cand)