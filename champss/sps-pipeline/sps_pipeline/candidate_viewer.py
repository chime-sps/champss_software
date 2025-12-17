import os
import json
import tqdm
import datetime
import mysql.connector
from astropy.coordinates import SkyCoord

class CandidateViewerRegistrar:
    def __init__(self, survey, folder, db_config, survey_dir):
        # Initialize registrar
        self.candidates = []
        self.survey = survey
        self.folder = folder
        self.db_config = db_config
        self.survey_dir = survey_dir

        # Sanity check if survey config exists
        self.survey_config_path = f"{self.survey_dir}/{self.survey}.json"
        if not os.path.exists(self.survey_config_path):
            raise FileNotFoundError(f"Survey config file not found: {self.survey_config_path}. Please create the survey first.")

        # Connect to database
        self.cursor = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            database=db_config['database']
        )

    def register_metadata(self, survey, folder, file, input_file, ra_deg, dec_deg, p0_ms, dm_pc_cc, snr):
        # Convert coordinates 
        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit='deg')
        ra_hms = coord.ra.to_string(unit='hourangle', sep=':', pad=True)
        dec_dms = coord.dec.to_string(unit='deg', sep=':', pad=True, alwayssign=True)

        # Gether data
        data = {
            "survey": survey,
            "folder": folder,
            "file": file,
            'input_file': input_file,
            "candidate": "",
            "telescope": "chime",
            "epoch_topo": "",
            "epoch_bary": "",
            "t_sample": "",
            "data_folded": "",
            "data_avg": "",
            "data_stdev": "",
            "profile_bins": "",
            "profile_avg": "",
            "profile_stdev": "",
            "reduce_chi_sqr": "",
            'prob_noise': str(snr),
            'best_dm': str(dm_pc_cc),
            "p_topo": "",
            "p_topo_d1": "",
            "p_topo_d2": "",
            'p_bary': str(p0_ms),
            "p_bary_d1": "0",
            "p_bary_d2": "0",
            "p_orb": "",
            "asin": "",
            "eccentricity": "",
            "w": "",
            "t_peri": "",
            "header_size": "",
            "data_size": "",
            "data_type": "",
            "notes": "",
            "datataking_machine": "champss",
            'source_ra': ra_hms,
            'source_dec': dec_dms,
            "freq": "600",
            "bw": "400",
            "N_channel": "",
            "N_beam": "",
            "beam_number": "",
            "sample_timestamp": "",
            "gregorian_data": "",
            "sample_time": "",
            "N_sample": "",
            "observation_length": "",
            "N_bits_per_sample": "",
            "N_IF": "",
            "source_name": "unknown"
        }

        # Generate SQL query
        keys = ', '.join(data.keys())
        values = ', '.join(['%s'] * len(data))
        sql = f"INSERT INTO profile_cache ({keys}) VALUES ({values})"

        val = tuple(data.values())
        self.cursor.cursor().execute(sql, val)
        self.cursor.commit()

    def generate_survey_config(self):
        config = {}
        for cand in self.candidates:
            config[cand['candname']] = {
                "plot_combined": cand['combined_plot'],
                "plot_stack": cand['stack_plot'],
                "plot_fold": cand['fold_plot'],
                "filename": cand['candname'],
            }

        surve_config = {"files": {}}
        surve_config["files"][self.folder] = config

        return surve_config

    def append_survey_config(self):
        # Generate new survey config
        new_config = self.generate_survey_config()

        # Load existing survey config
        with open(self.survey_config_path, 'r') as f:
            existing_config = json.load(f)

        # Append new config
        if "files" in existing_config:
            for this_new_folder in new_config["files"]:
                if this_new_folder in existing_config["files"]:
                    # Merge entries
                    existing_config["files"][this_new_folder].update(new_config["files"][this_new_folder])
                else:
                    existing_config["files"][this_new_folder] = new_config["files"][this_new_folder]
        else:
            raise Exception("Existing survey config missing 'files' key.")

        # Update the "Updated" field
        existing_config["config"]["Updated"] = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Save updated config
        with open(self.survey_config_path, 'w') as f:
            json.dump(existing_config, f, indent=4)

    def add_candidate(self, candname, ra, dec, f0, dm, snr, stack_plot, fold_plot, combined_plot, input_file=""):
        candidate = {
            'candname': candname,
            'ra': ra,
            'dec': dec,
            'f0': f0,
            'dm': dm,
            'snr': snr,
            'stack_plot': stack_plot,
            'fold_plot': fold_plot,
            'combined_plot': combined_plot, 
            'input_file': input_file
        }
        self.candidates.append(candidate)

    def add_candidates(self, df):
        for row in df.to_dict(orient='records'):
            candname = row['file_name'].split('/')[-1].replace('.npz', '')
            ra = float(row['best_ra'])
            dec = float(row['best_dec'])
            f0 = float(row['mean_freq'])
            dm = float(row['mean_dm'])
            snr = float(row['fs_sigma'])
            stack_plot = row['plot_path']
            fold_plot = row['fold_plot']
            combined_plot = row['combined_plot_path']
            input_file = row.get('file_name', "")

            self.add_candidate(
                candname=candname,
                ra=ra,
                dec=dec,
                f0=f0,
                dm=dm,
                snr=snr,
                stack_plot=stack_plot,
                fold_plot=fold_plot,
                combined_plot=combined_plot,
                input_file=input_file
            )

    def commit(self):
        # Commit candidates into database
        for cand in tqdm.tqdm(self.candidates, desc="Registering candidates"):
            self.register_metadata(
                survey=self.survey,
                folder=self.folder,
                file=cand['candname'],
                input_file=cand['input_file'],
                ra_deg=cand['ra'],
                dec_deg=cand['dec'],
                p0_ms=1000.0 / cand['f0'] if cand['f0'] != 0 else 0,
                dm_pc_cc=cand['dm'],
                snr=cand['snr']
            )

        # Append survey config
        self.append_survey_config()

    def close(self):
        self.cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()