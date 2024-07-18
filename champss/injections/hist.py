import numpy as np
import yaml
import subprocess

def call(injection_path, ra, dec, ii = None):


    command = ['run-stack-search-pipeline', str(ra), str(dec), 'search', '--cand-path', './',
            '--injection-path', injection_path, '--only-injections']

    if ii is not None:
        command.extend(['--ii', str(ii)])

    subprocess.run(command)

def get_cand_data(ra, dec, date):
    path = f'./candidates_cumul/{str(ra)}_{str(dec)}_cumulative_power_spectra_stack_{date}_candidates.npz'
    cands = np.load(path, allow_pickle = True)['candidate_dicts']
    return_list = []
    for cand in cands:
        temp_dict = {}
        temp_dict['freq'] = cand['freq']
        temp_dict['dm'] = cand['dm']
        temp_dict['sigma'] = cand['sigma']
        return_list.append(temp_dict)
    return return_list

def run_focus(ra, dec, date, path, focus, step = 1):

    injections = []

    with open(path, 'r') as file:
        data = yaml.safe_load(file)

    i = 0
    while i < len(data):
        print(f'Starting injection {i+1}.')
        inj = data[i]
        call(path, ra, dec, ii = i)
        del inj['profile']
        cands = get_cand_data(ra, dec, date)
        print(f'{len(cands)} candidates returned.')
        injections.append([inj, cands])
        i += step

    output = np.zeros((len(injections), 4))
    multicands = []

    for i in range(len(injections)):
        output[i, 0] = injections[i][0][focus]
        f = injections[i][0]['frequency']
        freq_closest = np.infty
        for cand in injections[i][1]:
            if np.abs(cand['freq'] - f) < np.abs(freq_closest - f):
                output[i, 1] = cand['freq']
                output[i, 2] = cand['dm']
                output[i, 3] = cand['sigma']
                freq_closest = cand['freq']

        if len(injections[i][1]) > 1:
            multicands.append([injections[i][0], len(injections[i][1])])

    return output, multicands
                
