import os
import json
import tqdm
import shutil
import datetime
import mysql.connector
from astropy.coordinates import SkyCoord


class CandidateViewerRegistrar:
    """
    Class to register candidate metadata into the database and update survey config.
    """

    def __init__(self, survey, folder, db_config, survey_dir):
        # Initialize registrar
        self.candidates = []
        self.survey = survey
        self.folder = folder
        self.db_config = db_config
        self.survey_dir = survey_dir

        # Sanity check for survey and folder names
        if (
            " " in self.survey
            or "/" in self.survey
            or "\\" in self.survey
            or "\t" in self.survey
        ):
            raise ValueError(
                "Survey name cannot contain spaces, slashes, backslashes, or tabs."
            )
        if (
            " " in self.folder
            or "/" in self.folder
            or "\\" in self.folder
            or "\t" in self.folder
        ):
            raise ValueError(
                "Folder name cannot contain spaces, slashes, backslashes, or tabs."
            )

        # Sanity check if survey config exists
        self.survey_config_path = f"{self.survey_dir}/{self.survey}.json"
        if not os.path.exists(self.survey_config_path):
            raise FileNotFoundError(
                f"Survey config file not found: {self.survey_config_path}. Please create the survey first."
            )

        # Connect to database
        self.cursor = mysql.connector.connect(
            host=db_config["host"],
            user=db_config["user"],
            port=db_config["port"],
            password=db_config["password"],
            database=db_config["database"],
        )

    def register_metadata(
        self,
        survey,
        folder,
        file,
        input_file,
        ra_deg,
        dec_deg,
        p0_ms,
        dm_pc_cc,
        snr,
        notes,
        commit=True,
    ):
        # Convert coordinates
        coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit="deg")
        ra_hms = coord.ra.to_string(unit="hourangle", sep=":", pad=True)
        dec_dms = coord.dec.to_string(unit="deg", sep=":", pad=True, alwayssign=True)

        # Gether data
        data = {
            "survey": survey,
            "folder": folder,
            "file": file,
            "input_file": input_file,
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
            "prob_noise": str(snr),
            "best_dm": str(dm_pc_cc),
            "p_topo": "",
            "p_topo_d1": "",
            "p_topo_d2": "",
            "p_bary": str(p0_ms),
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
            "notes": json.dumps(notes),
            "datataking_machine": "champss",
            "source_ra": ra_hms,
            "source_dec": dec_dms,
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
            "source_name": "unknown",
        }

        # Generate SQL query
        keys = ", ".join(data.keys())
        values = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO profile_cache ({keys}) VALUES ({values})"

        val = tuple(data.values())
        self.cursor.cursor().execute(sql, val)

        if commit:
            self.cursor.commit()

    def generate_survey_config(self):
        config = {}
        for cand in self.candidates:
            config[cand["candname"]] = {
                "plot_combined": cand["combined_plot"],
                "plot_stack": cand["stack_plot"],
                "plot_fold": cand["fold_plot"],
                "filename": cand["candname"],
            }

        surve_config = {"files": {}}
        surve_config["files"][self.folder] = config

        return surve_config

    def append_survey_config(self):
        """
        Append new candidates to existing survey config.
        """

        # Generate new survey config
        new_config = self.generate_survey_config()

        # Load existing survey config
        with open(self.survey_config_path, "r") as f:
            existing_config = json.load(f)

        # Append new config
        if "files" in existing_config:
            for this_new_folder in new_config["files"]:
                if this_new_folder in existing_config["files"]:
                    # Merge entries
                    existing_config["files"][this_new_folder].update(
                        new_config["files"][this_new_folder]
                    )
                else:
                    existing_config["files"][this_new_folder] = new_config["files"][
                        this_new_folder
                    ]
        else:
            raise Exception("Existing survey config missing 'files' key.")

        # Update the "Updated" field
        existing_config["config"]["Updated"] = datetime.datetime.utcnow().strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Check if output json is valid
        tmp_config_path = (
            "/tmp/" + os.path.basename(self.survey_config_path) + f".{os.getpid()}.tmp"
        )
        with open(tmp_config_path, "w") as f:
            json.dump(existing_config, f, indent=4)
        try:
            with open(tmp_config_path, "r") as f:
                _ = json.load(f)
        except json.JSONDecodeError as e:
            os.remove(tmp_config_path)
            raise Exception(
                f"Generated survey config JSON is invalid: {e}.Is there any None or NaN values in the condidate data?"
            )

        # Save updated config
        shutil.move(tmp_config_path, self.survey_config_path)

    def add_candidate(
        self,
        candname,
        ra,
        dec,
        f0,
        dm,
        snr,
        stack_plot,
        fold_plot,
        combined_plot,
        input_file="",
        fs_id="not_specified",
        fs_sigma="not_specified",
        fs_file="not_specified",
    ):
        """
        Add a candidate to the registrar.

        Parameters:
        - candname: Candidate name
        - ra: Right Ascension in degrees
        - dec: Declination in degrees
        - f0: Frequency in Hz
        - dm: Dispersion Measure in pc/cm^3
        - snr: Signal-to-noise ratio
        - stack_plot: Path to stack plot
        - fold_plot: Path to fold plot
        - combined_plot: Path to combined plot
        - input_file: Original input file path
        """
        candidate = {
            "candname": str(candname),
            "ra": float(ra),
            "dec": float(dec),
            "f0": float(f0),
            "dm": float(dm),
            "snr": float(snr),
            "stack_plot": str(stack_plot),
            "fold_plot": str(fold_plot),
            "combined_plot": str(combined_plot),
            "input_file": str(input_file),
            "notes": {"fs_id": fs_id, "fs_sigma": fs_sigma, "fs_file": fs_file},
        }
        self.candidates.append(candidate)

    def add_candidates(self, df):
        """
        Add multiple candidates from a DataFrame.

        Parameters:
        - df: pandas DataFrame with candidate data
        """

        for row in df.to_dict(orient="records"):
            candname = (
                row["file_name"]
                .split("/")[-1]
                .replace(".npz", "")
                .replace(" ", "_")
                .replace("\t", "_")
                .replace("/", "_")
                .replace("\\", "_")
            )
            ra = float(row["best_ra"])
            dec = float(row["best_dec"])
            f0 = float(row["mean_freq"])
            dm = float(row["mean_dm"])
            snr = float(row["fs_sigma"]) if row.get("fs_sigma") else float(row["sigma"])
            stack_plot = row.get("plot_path", "")
            fold_plot = row.get("fold_plot", "")
            combined_plot = row.get("combined_plot_path", "")
            input_file = row.get("file_name", "")
            fs_id = row.get("fs_id", "unknown")
            fs_sigma = row.get("fs_sigma", "unknown")
            fs_file = row.get("fs_file", "unknown")

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
                input_file=input_file,
                fs_id=fs_id,
                fs_sigma=fs_sigma,
                fs_file=fs_file,
            )

    def commit(self):
        """
        Commit all registered candidates to the database and update survey config.
        """

        # Add candidates into database
        for cand in tqdm.tqdm(self.candidates, desc="Registering candidates"):
            self.register_metadata(
                survey=self.survey,
                folder=self.folder,
                file=cand["candname"],
                input_file=cand["input_file"],
                ra_deg=cand["ra"],
                dec_deg=cand["dec"],
                p0_ms=1000.0 / cand["f0"] if cand["f0"] != 0 else 0,
                dm_pc_cc=cand["dm"],
                snr=cand["snr"],
                notes=cand["notes"],
                commit=False,
            )

        # Append survey config
        self.append_survey_config()

        # Commit all at once
        self.cursor.commit()

    def close(self):
        self.cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class CandidateViewerQuery:
    """
    Class to query candidate metadata and ratings from the database.
    """

    def __init__(self, survey, db_config):
        # Initialize registrar
        self.candidates = []
        self.db_config = db_config
        self.survey = survey
        self.survey_info = None

        # Connect to database
        self.cursor = mysql.connector.connect(
            host=db_config["host"],
            user=db_config["user"],
            port=db_config["port"],
            password=db_config["password"],
            database=db_config["database"],
        )

        # Fetch survey info
        self.survey_info = self.get_survey_info()

    def get_survey_info(self):
        """
        Fetch survey information from the database.
        """

        # Get data
        sql = "SELECT * FROM surveys WHERE survey_id = %s"
        this_cursor = self.cursor.cursor(dictionary=True)
        this_cursor.execute(sql, (self.survey,))
        survey_info = this_cursor.fetchone()

        # Parse db info
        if survey_info is not None:
            survey_info["db_info"] = json.loads(survey_info["db_info"])
        else:
            raise ValueError(f"Survey '{self.survey}' not found in the database.")

        return survey_info

    def get_metadata(self, folder=None, file=None):
        """
        Retrieve metadata from the database based on provided filters.

        Parameters:
        - survey (str): The survey / project name.
        - folder (str, optional): The folder name.
        - file (str, optional): The file name.

        """
        query = "SELECT * FROM profile_cache WHERE survey = %s"
        params = [self.survey]

        if folder:
            query += " AND folder = %s"
            params.append(folder)
        if file:
            query += " AND file = %s"
            params.append(file)

        this_cursor = self.cursor.cursor(dictionary=True)
        this_cursor.execute(query, params)
        results = this_cursor.fetchall()

        # Parse notes field
        for i in range(len(results)):
            if "notes" in results[i]:
                try:
                    results[i]["notes"] = json.loads(results[i]["notes"])
                except:
                    results[i]["notes"] = {
                        "fs_id": "unknown",
                        "fs_sigma": "unknown",
                        "fs_file": "unknown",
                    }  # Placeholder for early entries without notes

        return results

    def get_ratings(
        self, folder=None, file=None, classification=None, with_metadata=False
    ):
        """
        Retrieve ratings from the database based on provided filters.

        Parameters:
        - survey (str): The survey / project name.
        - folder (str, optional): The folder name.
        - file (str, optional): The file name.
        - classification (str, optional): The classification type,
          'NEW CANDIDATE' for New Candidates classification,
          '<faint>' for Faint / Ambiguous classification,
          '<none>' for RFI / None classification,
          '<any_known>' for all known pulsars,
          'B1234+5678' (pulsar name) for a specific known pulsar.
        - with_metadata (bool, optional): Whether to include metadata in the results.

        """
        query = (
            f"SELECT * FROM ratings_{self.survey_info['db_info']['suffix']} WHERE 1=1"
        )
        params = []

        if folder:
            query += " AND folder = %s"
            params.append(folder)
        if file:
            query += " AND file = %s"
            params.append(file)
        if classification:
            if classification == "<any_known>":
                query += " AND result NOT IN (%s, %s, %s)"
                params.extend(["NEW CANDIDATE", "<faint>", "<none>"])
            else:
                query += " AND result = %s"
                params.append(classification)

        print(query, params)
        this_cursor = self.cursor.cursor(dictionary=True)
        this_cursor.execute(query, params)
        results = this_cursor.fetchall()

        if with_metadata:
            for i in range(len(results)):
                metadata = self.get_metadata(
                    folder=results[i]["folder"], file=results[i]["file"]
                )
                results[i]["metadata"] = metadata[0] if metadata else None

        return results

    def close(self):
        self.cursor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
