import numpy as np
import os
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from folding.archive_utils import readpsrarch, get_SN, clean_foldspec

def plot_candidate_archive(fn, sigma, dm, f0, ra, dec, coord_path, known=" "):

    data, F, T, psr, tel = readpsrarch(fn)
    print(data.shape)

    fs, flag, mask, bg, bpass = clean_foldspec(data.squeeze())
    taxis = ((T - T[0])*u.day).to(u.min)
    T0 = Time(T[0], format='mjd')
    
    fs_bg = fs - np.median(fs, axis=-1, keepdims=True)
    fs_bg = fs_bg[1:-1]
    
    ngate = fs.shape[-1]
    nc = 256
    binf = fs_bg.shape[1]//nc
    fs_bin = np.copy(fs_bg)
    maskchan = np.argwhere(fs_bin.mean(0).mean(-1)==0).squeeze()
    fs_bin[:,maskchan] = np.nan
    fs_bin = np.nanmean(fs_bin.reshape(fs_bin.shape[0], fs_bin.shape[1]//binf, binf, -1), 2)
    
    profile = np.nanmean(np.nanmean(fs_bin,0), 0)
    SNR_val, SNprof = get_SN(profile, return_profile=True)

    vfmin = np.nanmean(fs_bin) - 1*np.nanstd(np.nanmean(fs_bin, 0))
    vfmax = np.nanmean(fs_bin) + 3*np.nanstd(np.nanmean(fs_bin, 0))
    vtmin = np.nanmean(fs_bin) - 1*np.nanstd(np.nanmean(fs_bin, 1))
    vtmax = np.nanmean(fs_bin) + 3*np.nanstd(np.nanmean(fs_bin, 1))

    fig = plt.figure(figsize=(12, 8))
    ax0 = plt.subplot2grid((3, 4), (0, 0), colspan=2)
    ax1 = plt.subplot2grid((3, 4), (1, 0), colspan=2, rowspan=2)
    ax3 = plt.subplot2grid((3, 4), (1, 2), colspan=2, rowspan=2)

    plt.subplots_adjust(hspace=0.05, wspace=0.05)

    ax0.set_title('{0} {1}'.format(psr, T0.isot[:10]), fontsize=18)

    ax1.imshow(np.nanmean(fs_bin, 0), aspect='auto', interpolation='nearest', vmin=vfmin, vmax=vfmax,
              extent=[0,1, F[-1], F[1]])
    ax3.imshow(np.nanmean(fs_bin,1), aspect='auto', interpolation='nearest', vmin=vtmin, vmax=vtmax, origin='lower',
              extent=[0,1, 0, max(taxis.to(u.min).value)])

    ax1.set_ylabel('Frequency (MHz)', fontsize=18)
    ax1.set_xlabel('Phase', fontsize=18)

    ax3.set_ylabel('Time (min)', fontsize=18)
    ax3.set_xlabel('Phase', fontsize=18)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    phaseaxis = np.linspace(0,1, ngate, endpoint=False)
    ax0.plot(phaseaxis, SNprof)
    ax0.set_xlim(0, 1)
    ax0.set_xticks([])
    if known.strip(): 
        print(f'Known pulsar {known} detected')
        parameters_text = (
        f"{known} \n"    
        f"Incoherent $\sigma$: {sigma}\n"
        f"Folded $\sigma$: {SNR_val}\n"
        f"RA (deg): {ra}\n"
        f"DEC (deg): {dec}\n"
        f"DM: {dm}\n"
        f"f0: {f0}\n"
        )
    else: 
        print('Not a known pulsar')
        parameters_text = (
        f"Incoherent $\sigma$: {sigma}\n"
        f"Folded $\sigma$: {SNR_val}\n"
        f"RA (deg): {ra}\n"
        f"DEC (deg): {dec}\n"
        f"DM: {dm}\n"
        f"f0: {f0}\n"
        )
    
    ax0.text(1.05, 1.0, parameters_text, transform=ax0.transAxes,
             fontsize=18, va='top', ha='left', backgroundcolor='white')
    plt.savefig(coord_path + '/{0}_{1}_{2}_{3}.png'.format(psr, T0.isot[:10], round(dm,2), round(f0,2)))
    
    img_path = "/data/chime/sps/archives/plots/folded_candidate_plots/{0}-plots/".format(T0.isot[:10])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    else:
        print(f"Directory '{img_path}' already exists.")
    plt.savefig(img_path + "{0}_{1}_{2}_{3}.png".format(psr, T0.isot[:10], round(dm,2), round(f0,2)))
    plt.close()

    return SNprof, SNR_val