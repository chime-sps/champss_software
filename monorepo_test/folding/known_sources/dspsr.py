import os
import sys
import datetime
import subprocess
import numpy as np
from math import ceil
import glob
from pathlib import Path

from beamformer.strategist.strategist import PointingStrategist
from beamformer.skybeam import SkyBeamFormer
from sps_databases import db_api

if __name__ == "__main__":
   
    if len(sys.argv) < 5:
        print("usage = fold_psr.py psrname year month day")
   
    psr = str(sys.argv[1])
    year = int(sys.argv[2])
    month = int(sys.argv[3])
    day = int(sys.argv[4])
    print("setting up pointing for {0}...".format(psr))

    ephdir = '/data/adenney/par_files/'
    outdir = '/data/adenney/fil/'
    #ardir = '/data/chime/sps/archives/known_sources/folded_profiles/'+psr
    fname = f'{psr}_{year}-{month}-{day}.fil'
    fil = outdir+fname

    source = db_api.get_known_source_by_name(psr)[0]
    pst = PointingStrategist()
    print(source.pos_ra_deg, source.pos_dec_deg)
    ap = pst.get_single_pointing(source.pos_ra_deg, source.pos_dec_deg, datetime.datetime(year, month, day))
    parfile = f"{ephdir}{psr}.par"
    try:
        open(parfile, 'r')
    except:
        print(f"could not open {parfile}, using psrcat ephemeris")
        subprocess.run(["psrcat",f"{psr}", "-e", ">", f"{psr}.par"], shell=False)
        parfile=f"{psr}.par"
    
    basepath="/data/chime/sps-magnetar/raw/"
    if day<10:
        day_str = '0'+str(day)
    else:
        day_str=str(day)
    ar = str(year)+'-'+str(month)+'-'+day_str
    os.chdir(r'/data/chime/sps/archives/known_sources/folded_profiles/'+psr)
    try:
        archive = glob.glob(f'{ar}-*.ar')[0]
    except: 
        archive = 'null.ar'
    print(archive)
    if Path(archive).is_file() != True:
        
        print("Beamforming...")
        sbf = SkyBeamFormer(
            extn="dat",
            update_db=False,
            min_data_frac=0.5,
            basepath=basepath,
            add_local_median=True,
            detrend_data=True,
            detrend_nsamp=32768,
            masking_timescale=512000,
            #flatten_bandpass=False,
            run_rfi_mitigation=True,
            masking_dict=dict(weights=True, l1=True, badchan=True, kurtosis=False, mad=False, sk=True, powspec=False, dummy=False),
            beam_to_normalise=1,
        )
        sb = sbf.form_skybeam(ap[0], num_threads=32)
    
        print(f"Writing to {fil}")
        sb.write(fil)
        print("Folding...")
        turns = str(ceil(10/source.spin_period_s))
        os.chdir(r'/data/chime/sps/archives/known_sources/folded_profiles/'+psr)
        subprocess.run(["dspsr", "-turns", turns, "-A", "-k", "chime", "-E", f"{parfile}", f"{fil}"])
       
    ra = "{:.2f}".format(ap[0].ra)
    dec = "{:.2f}".format(ap[0].dec)
    os.chdir(r'/data/chime/sps/archives/known_sources/folded_profiles/'+psr)

    try:
        archive = glob.glob(f'{ar}-*.ar')[0]
    except: 
        archive = 'null.ar'
    print('archive after beam ', archive)
    if Path(archive).is_file():
        if Path(psr+'_'+ar+'.png').is_file() != True:
            subprocess.run(["pam", "-T", "-F", f"{ar}-*.ar", "-e","FT"])
            os.system("python /home/adenney/scripts/profile_plot.py "+archive+" "+psr)
        if Path('paas.std').is_file():
            subprocess.run(["pat", "-f", "tempo2", "-s", "paas.std", "*.FT", ">", "name.tim"])
            subprocess.run(["psrstat", "-c", "snr", "-Q", "*.FT", ">", "snr.txt"])
            os.system("pat -f tempo2 -s paas.std *.FT > "+psr+".tim")
            os.system("psrstat -c snr -Q *.FT > snr.txt")
            os.system("python /home/adenney/scripts/toa_plots.py "+psr+" "+ra+" "+dec)
    
    os.chdir(r'/data/adenney/')
    print(f"Finished, removing {fil} from memory")
    os.remove(fil)
    
