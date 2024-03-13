import numpy as np
import os
import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

import psrchive

def readpsrarch(fname, dedisperse=True, verbose=True):
    """
    Read pulsar archive directly using psrchive
    Requires the python psrchive bindings

    Parameters
    ----------
    fname: string
    file directory
    dedisperse: Bool
    apply psrchive's by-channel incoherent de-dispersion

    Returns archive data cube, frequency array, time(mjd) array, source name
    """
    
    arch = psrchive.Archive.load(fname)
    source = arch.get_source()
    tel = arch.get_telescope()
    if verbose:
        print("Read archive of {0} from {1}".format(source, fname))

    if dedisperse:
        if verbose:
            print("Dedispersing...")
        arch.dedisperse()
    data = arch.get_data()
    midf = arch.get_centre_frequency()
    bw = arch.get_bandwidth()
    F = np.linspace(midf-bw/2., midf+bw/2., data.shape[2], endpoint=False)
    #F = arch.get_frequencies()

    a = arch.start_time()
    t0 = a.strtempo()
    print(t0)
    t0 = Time(float(t0), format='mjd', precision=9)

    # Get frequency and time info for plot axes
    nt = data.shape[0]
    Tobs = arch.integration_length()
    dt = (Tobs / nt)*u.s
    T = t0 + np.arange(nt)*dt
    T = T.mjd
    
    return data, F, T, source, tel

def clean_foldspec(I, plots=False, apply_mask=True, rfimethod='var', flagval=7, offpulse='True', tolerance=0.7, off_gates=0):
    """
    Clean and rescale folded spectrum
    
    Parameters
    ----------
    I: ndarray [time, pol, freq, phase]
    or [time, freq, phase] for single pol
    plots: Bool
    Create diagnostic plots
    apply_mask: Bool
    Multiply dynamic spectrum by mask
    rfimethod: String
    RFI flagging method, currently only supports var
    tolerance: float
    % of subints per channel to zap whole channel
    off_gates: slice
    manually chosen off_gate region.  If unset, the bottom 50%
    of the profile is chosen

    Returns
    ------- 
    foldspec: folded spectrum, after bandpass division, 
    off-gate subtraction and RFI masking
    flag: std. devs of each subint used for RFI flagging
    mask: boolean RFI mask
    bg: Ibg(t, f) subtracted from foldspec
    bpass: Ibg(f), an estimate of the bandpass
    
    """

    # Sum to form total intensity, mostly a memory/speed concern
    if len(I.shape) == 4:
        print(I.shape)
        I = I[:,(0,1)].mean(1)

    # Use median over time to not be dominated by outliers
    bpass = np.median( I.mean(-1, keepdims=True), axis=0, keepdims=True)
    foldspec = I / bpass
    
    mask = np.ones_like(I.mean(-1))

    prof_dirty = (I - I.mean(-1, keepdims=True)).mean(0).mean(0)
    if not off_gates:
        off_gates = np.argwhere(prof_dirty<np.median(prof_dirty)).squeeze()
        recompute_offgates = 1
    else:
        recompute_offgates = 0

    if rfimethod == 'var':
        if offpulse:
            flag = np.std(foldspec[..., off_gates], axis=-1)
        else:
            flag = np.std(foldspec, axis=-1)
        # find std. dev of flag values without outliers
        flag_series = np.sort(flag.ravel())
        flagsize = len(flag_series)
        flagmid = slice(int(flagsize//4), int(3*flagsize//4) )
        flag_std = np.std(flag_series[flagmid])
        flag_mean = np.mean(flag_series[flagmid])

        # Mask all values over 10 sigma of the mean
        mask[flag > flag_mean+flagval*flag_std] = 0

        # If more than 50% of subints are bad in a channel, zap whole channel
        mask[:, mask.mean(0)<tolerance] = 0
        mask[mask.mean(1)<tolerance] = 0
        if apply_mask:
            I[mask < 0.5] = np.mean(I[mask > 0.5])

    
    profile = I.mean(0).mean(0)

    # redetermine off_gates, if off_gates not specified
    if recompute_offgates:
        off_gates = np.argwhere(profile<np.median(profile)).squeeze()
    
    # renormalize, now that RFI are zapped
    bpass = I[...,off_gates].mean(-1, keepdims=True).mean(0, keepdims=True)
    foldspec = I / bpass
    foldspec[np.isnan(foldspec)] = np.nanmean(foldspec)
    bg = np.mean(foldspec[...,off_gates], axis=-1, keepdims=True)
    foldspec = foldspec - bg
        
    if plots:
        plot_diagnostics(foldspec, flag, mask)

    return foldspec, flag, mask, bg.squeeze(), bpass.squeeze()

def get_SN(profile):
    ngate = len(profile)
    maxbin = int(np.log2(ngate//2))
    
    binning = 2**np.arange(maxbin)
    SNprofs = np.zeros((len(binning), len(profile)))
    
    for i,b in enumerate(binning):
        prof_filtered = uniform_filter(profile, b)
        profsort = np.sort(prof_filtered)
        prof_N = profsort[:3*ngate//4]
        std = np.std(prof_N)
        mean = np.mean(prof_N)
        SNprof = (prof_filtered - mean) / std
        SNprofs[i] = SNprof

    SNmax = np.max(SNprofs)
    return SNprofs, SNmax
    

def plot_foldspec(fn, sigma, dm, f0, ra, dec, coord_path, known=" "):

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
    SNprofs, SNR_val = get_SN(profile)
    SNprof = SNprofs[0]

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
    plt.savefig(coord_path + '/{0}-{1}-{2}-{3}.png'.format(psr, T0.isot[:10], round(dm,2), round(f0,2)))
    
    img_path = "/data/chime/sps/archives/plots/folded_candidate_plots/{0}-plots/".format(T0.isot[:10])
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    else:
        print(f"Directory '{img_path}' already exists.")
    plt.savefig(img_path + "{0}-{1}-{2}-{3}.png".format(psr, T0.isot[:10], round(dm,2), round(f0,2)))

    return SNprof, SNR_val